from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from nvidia_hackathon.prompt_template import (
    STANDARD_CATEGORY_CODES,
    STANDARD_GUARDRAIL,
    build_interval_sft_prompt,
)

RESPONSE_PATTERN = re.compile(
    r"DESCRIPTION:\s*(.*?)\s*GUARDRAIL:\s*(\{.*?\})\s*EXPLANATION:\s*(.*)\Z",
    flags=re.DOTALL,
)
CLIP_RANGE_PATTERN = re.compile(r"(?P<start>\d+)_(?P<end>\d+)\.mp4\Z")


@dataclass(frozen=True)
class ParsedResponse:
    description: str
    guardrail: dict[str, bool]
    explanation: str

    @property
    def is_standard(self) -> bool:
        return tuple(self.guardrail.keys()) == tuple(STANDARD_GUARDRAIL.keys())

    @property
    def has_harm(self) -> bool:
        return any(self.guardrail.values())

    @property
    def has_required_details(self) -> bool:
        return bool(self.description and self.explanation)


def normalize_clip_response(parsed: ParsedResponse) -> ParsedResponse | None:
    if parsed.has_required_details:
        return parsed
    if parsed.has_harm:
        return None
    return ParsedResponse(
        description=parsed.description or "None",
        guardrail=parsed.guardrail,
        explanation=parsed.explanation or "None",
    )


@dataclass(frozen=True)
class ShotRecord:
    start_sec: float
    end_sec: float
    description: str
    guardrail: dict[str, bool]
    explanation: str
    clip_relative_path: str

    def to_output_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("clip_relative_path")
        return payload


IntervalRecord = ShotRecord


@dataclass(frozen=True)
class VideoSample:
    sample_id: str
    video_path: str
    video_relative_path: str
    duration_sec: float
    prompt: str
    assistant_response: str
    video_description: str
    shots: list[ShotRecord]
    source_category: str

    def to_jsonl_record(self) -> dict[str, Any]:
        num_harmful_shots = sum(1 for shot in self.shots if _guardrail_true_codes(shot.guardrail))
        return {
            "id": self.sample_id,
            "task_name": "safewatch-shot-sft",
            "video": self.video_path,
            "videos": [self.video_path],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": self.video_path},
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {"role": "assistant", "content": self.assistant_response},
            ],
            "conversations": [
                {"from": "human", "value": f"<video>\n{self.prompt}"},
                {"from": "gpt", "value": self.assistant_response},
            ],
            "metadata": {
                "video_relative_path": self.video_relative_path,
                "duration_sec": round(self.duration_sec, 3),
                "num_shots": len(self.shots),
                "num_harmful_shots": num_harmful_shots,
                "num_benign_shots": len(self.shots) - num_harmful_shots,
                "num_violations": num_harmful_shots,
                "source_category": self.source_category,
            },
        }


def _guardrail_true_codes(guardrail: dict[str, bool]) -> list[str]:
    true_codes: list[str] = []
    for code in STANDARD_CATEGORY_CODES:
        matched_key = next((key for key in guardrail if key.startswith(f"{code}(")), None)
        if matched_key and guardrail[matched_key]:
            true_codes.append(code)
    return true_codes


def parse_guardrail_json(raw_guardrail: str) -> dict[str, bool]:
    normalized = re.sub(
        r'(:\s*)(True|False|true|false)(?=\s*[,}])',
        lambda match: f"{match.group(1)}{match.group(2).lower()}",
        raw_guardrail,
    )
    payload = json.loads(normalized)
    return {str(key): bool(value) for key, value in payload.items()}


def parse_safewatch_response(text: str) -> ParsedResponse | None:
    match = RESPONSE_PATTERN.search(text.strip())
    if match is None:
        return None
    description = match.group(1).strip()
    explanation = match.group(3).strip()
    try:
        guardrail = parse_guardrail_json(match.group(2))
    except json.JSONDecodeError:
        return None
    return ParsedResponse(description=description, guardrail=guardrail, explanation=explanation)


def annotation_path_to_full_video(relative_video_path: str) -> str | None:
    parts = Path(relative_video_path).parts
    if len(parts) < 5 or parts[0] != "dataset":
        return None
    if parts[1] == "full":
        return relative_video_path
    if parts[1] != "clip":
        return None
    return str(Path("dataset") / "full" / parts[2] / "target" / f"{parts[-2]}.mp4")


def annotation_path_to_absolute_path(dataset_root: Path, relative_video_path: str) -> Path:
    if relative_video_path.startswith("dataset/"):
        return dataset_root / relative_video_path.removeprefix("dataset/")
    return dataset_root / relative_video_path


def clip_time_range(relative_clip_path: str) -> tuple[float, float] | None:
    match = CLIP_RANGE_PATTERN.search(Path(relative_clip_path).name)
    if match is None:
        return None
    start_sec = float(int(match.group("start")))
    end_sec = float(int(match.group("end")))
    if end_sec <= start_sec:
        return None
    return start_sec, end_sec


def probe_video_duration_sec(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid video metadata for {video_path}: fps={fps}, frame_count={frame_count}")
        return frame_count / fps
    finally:
        capture.release()


def is_canonical_annotation_dir(annotation_dir: Path) -> bool:
    return not annotation_dir.name.endswith(("_adaptive", "_binary", "_qa"))


def has_acceptable_shot_coverage(
    shots: list[ShotRecord],
    duration_sec: float,
    *,
    max_gap_sec: float = 5.0,
    end_tolerance_sec: float = 1e-6,
) -> bool:
    if not shots:
        return False
    if shots[0].start_sec > max_gap_sec:
        return False
    previous_end = shots[0].end_sec
    for shot in shots[1:]:
        gap_sec = shot.start_sec - previous_end
        if gap_sec < 0:
            return False
        if gap_sec > max_gap_sec:
            return False
        previous_end = max(previous_end, shot.end_sec)
    tail_gap_sec = max(0.0, duration_sec - shots[-1].end_sec)
    return tail_gap_sec <= max_gap_sec + end_tolerance_sec


def build_assistant_response(video_description: str, shots: list[ShotRecord]) -> str:
    del video_description
    payload = [
        {
            "START_SEC": shot.start_sec,
            "END_SEC": shot.end_sec,
            "DESCRIPTION": shot.description,
            "GUARDRAIL": shot.guardrail,
            "EXPLANATION": shot.explanation,
        }
        for shot in shots
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def choose_full_response(parsed_full_responses: list[ParsedResponse]) -> ParsedResponse | None:
    for response in parsed_full_responses:
        if not response.is_standard:
            continue
        normalized_response = normalize_clip_response(response)
        if normalized_response is not None:
            return normalized_response
    return None


def choose_full_description(parsed_full_responses: list[ParsedResponse]) -> str | None:
    selected_response = choose_full_response(parsed_full_responses)
    if selected_response is not None:
        return selected_response.description
    return None


def build_single_shot_from_full_response(
    full_relative_path: str,
    duration_sec: float,
    parsed_full_response: ParsedResponse,
) -> ShotRecord:
    return ShotRecord(
        start_sec=0.0,
        end_sec=round(duration_sec, 6),
        description=parsed_full_response.description,
        guardrail=parsed_full_response.guardrail,
        explanation=parsed_full_response.explanation,
        clip_relative_path=f"{full_relative_path}#full_fallback",
    )


def _sample_id_from_relative_path(relative_path: str) -> str:
    path = Path(relative_path)
    return f"{path.parts[2]}__{path.stem}"


def summarize_samples(samples: list[VideoSample]) -> dict[str, Any]:
    source_category_counts = Counter(sample.source_category for sample in samples)
    main_category_sample_counts = Counter()
    main_category_shot_counts = Counter()
    num_total_shots = 0
    num_harmful_shots = 0

    for sample in samples:
        sample_codes = sorted(
            {
                code
                for shot in sample.shots
                for code in _guardrail_true_codes(shot.guardrail)
            }
        )
        for code in sample_codes:
            main_category_sample_counts[code] += 1
        num_total_shots += len(sample.shots)
        for shot in sample.shots:
            shot_codes = _guardrail_true_codes(shot.guardrail)
            if shot_codes:
                num_harmful_shots += 1
            for code in shot_codes:
                main_category_shot_counts[code] += 1

    return {
        "num_samples": len(samples),
        "num_harmful_samples": sum(
            1 for sample in samples if any(_guardrail_true_codes(shot.guardrail) for shot in sample.shots)
        ),
        "num_benign_samples": sum(
            1 for sample in samples if all(not _guardrail_true_codes(shot.guardrail) for shot in sample.shots)
        ),
        "num_total_shots": num_total_shots,
        "num_harmful_shots": num_harmful_shots,
        "num_benign_shots": num_total_shots - num_harmful_shots,
        "source_category_counts": dict(sorted(source_category_counts.items())),
        "main_category_sample_counts": {
            code: main_category_sample_counts.get(code, 0) for code in STANDARD_CATEGORY_CODES
        },
        "main_category_shot_counts": {
            code: main_category_shot_counts.get(code, 0) for code in STANDARD_CATEGORY_CODES
        },
        "main_category_interval_counts": {
            code: main_category_shot_counts.get(code, 0) for code in STANDARD_CATEGORY_CODES
        },
    }


def split_train_eval_by_source_category(
    samples: list[VideoSample],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[list[VideoSample], list[VideoSample], dict[str, Any]]:
    rng = random.Random(seed)
    grouped_samples: dict[str, list[VideoSample]] = defaultdict(list)
    for sample in samples:
        grouped_samples[sample.source_category].append(sample)

    train_samples: list[VideoSample] = []
    eval_samples: list[VideoSample] = []
    eval_counts = {}

    for source_category in sorted(grouped_samples):
        bucket = list(grouped_samples[source_category])
        rng.shuffle(bucket)
        raw_eval_count = max(1, math.ceil(len(bucket) * eval_ratio))
        eval_count = min(raw_eval_count, len(bucket) if len(bucket) == 1 else len(bucket) - 1)
        eval_counts[source_category] = eval_count
        eval_samples.extend(bucket[:eval_count])
        train_samples.extend(bucket[eval_count:])

    train_samples.sort(key=lambda sample: sample.sample_id)
    eval_samples.sort(key=lambda sample: sample.sample_id)
    split_summary = {
        "eval_ratio_target": eval_ratio,
        "seed": seed,
        "eval_counts_by_source_category": eval_counts,
        "train_summary": summarize_samples(train_samples),
        "eval_summary": summarize_samples(eval_samples),
    }
    return train_samples, eval_samples, split_summary


def collect_sft_samples(
    *,
    dataset_root: Path,
    project_root: Path,
    max_duration_sec: float = 60.0,
    include_benign: bool = True,
    max_samples: int | None = None,
) -> tuple[list[VideoSample], dict[str, Any]]:
    annotation_root = dataset_root / "annotation"
    if not annotation_root.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_root}")

    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "source_category": None,
            "full_responses": [],
            "parsed_clip_candidates": {},
            "has_incomplete_clip_annotation": False,
        }
    )
    stats = Counter()

    for annotation_dir in sorted(path for path in annotation_root.iterdir() if path.is_dir()):
        if not is_canonical_annotation_dir(annotation_dir):
            continue
        full_json_path = annotation_dir / "full.json"
        if not full_json_path.exists():
            continue
        with full_json_path.open() as handle:
            records = json.load(handle)

        for record in records:
            relative_video_path = record.get("video")
            conversations = record.get("conversations", [])
            if not relative_video_path or len(conversations) < 2:
                stats["skipped_invalid_record"] += 1
                continue

            parsed = parse_safewatch_response(conversations[1].get("value", ""))
            is_clip_record = relative_video_path.startswith("dataset/clip/")
            if parsed is None:
                if is_clip_record:
                    full_relative_path = annotation_path_to_full_video(relative_video_path)
                    if full_relative_path is not None:
                        grouped[full_relative_path]["has_incomplete_clip_annotation"] = True
                stats["skipped_unparseable_response"] += 1
                continue

            full_relative_path = annotation_path_to_full_video(relative_video_path)
            if full_relative_path is None:
                stats["skipped_unrecognized_video_path"] += 1
                continue

            bucket = grouped[full_relative_path]
            bucket["source_category"] = bucket["source_category"] or annotation_dir.name

            if relative_video_path.startswith("dataset/full/"):
                bucket["full_responses"].append(parsed)
                stats["full_records_seen"] += 1
                continue

            if not parsed.is_standard:
                bucket["has_incomplete_clip_annotation"] = True
                stats["skipped_nonstandard_clip_response"] += 1
                continue

            normalized_clip = normalize_clip_response(parsed)
            if normalized_clip is None:
                bucket["has_incomplete_clip_annotation"] = True
                stats["skipped_incomplete_clip"] += 1
                continue

            existing = bucket["parsed_clip_candidates"].get(relative_video_path)
            if existing is None:
                bucket["parsed_clip_candidates"][relative_video_path] = normalized_clip
            stats["clip_records_seen"] += 1

    prompt = build_interval_sft_prompt(project_root)
    samples: list[VideoSample] = []

    for full_relative_path in sorted(grouped):
        bucket = grouped[full_relative_path]
        selected_full_response = choose_full_response(bucket["full_responses"])
        if selected_full_response is None:
            stats["excluded_missing_full_description"] += 1
            continue
        full_description = selected_full_response.description
        if bucket["has_incomplete_clip_annotation"]:
            stats["excluded_incomplete_shot_video"] += 1
            continue

        absolute_video_path = annotation_path_to_absolute_path(dataset_root, full_relative_path)
        if not absolute_video_path.exists():
            stats["excluded_missing_video_file"] += 1
            continue

        try:
            duration_sec = probe_video_duration_sec(absolute_video_path)
        except RuntimeError:
            stats["excluded_unreadable_video"] += 1
            continue

        if duration_sec > max_duration_sec:
            stats["excluded_over_duration"] += 1
            continue

        shots: list[ShotRecord] = []
        for clip_relative_path, parsed_clip in sorted(bucket["parsed_clip_candidates"].items()):
            clip_range = clip_time_range(clip_relative_path)
            if clip_range is None:
                stats["excluded_bad_clip_range_video"] += 1
                shots = []
                break
            start_sec, end_sec = clip_range
            shots.append(
                ShotRecord(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    description=parsed_clip.description,
                    guardrail=parsed_clip.guardrail,
                    explanation=parsed_clip.explanation,
                    clip_relative_path=clip_relative_path,
                )
            )

        if not shots:
            if not bucket["parsed_clip_candidates"]:
                shots = [
                    build_single_shot_from_full_response(
                        full_relative_path,
                        duration_sec,
                        selected_full_response,
                    )
                ]
                stats["used_full_annotation_single_shot_fallback"] += 1
            else:
                stats["excluded_missing_shots"] += 1
                continue

        shots.sort(key=lambda item: (item.start_sec, item.end_sec, item.clip_relative_path))
        if not has_acceptable_shot_coverage(shots, duration_sec):
            stats["excluded_shot_gap_over_limit"] += 1
            continue

        if not include_benign and all(not _guardrail_true_codes(shot.guardrail) for shot in shots):
            stats["excluded_benign_video"] += 1
            continue

        assistant_response = build_assistant_response(full_description, shots)
        samples.append(
            VideoSample(
                sample_id=_sample_id_from_relative_path(full_relative_path),
                video_path=str(absolute_video_path),
                video_relative_path=full_relative_path,
                duration_sec=duration_sec,
                prompt=prompt,
                assistant_response=assistant_response,
                video_description=full_description,
                shots=shots,
                source_category=str(bucket["source_category"]),
            )
        )
        stats["included_samples"] += 1
        if all(not _guardrail_true_codes(shot.guardrail) for shot in shots):
            stats["included_benign_samples"] += 1
        else:
            stats["included_harmful_samples"] += 1

        if max_samples is not None and len(samples) >= max_samples:
            break

    summary = {
        "num_samples": len(samples),
        "num_harmful_samples": sum(
            1 for sample in samples if any(_guardrail_true_codes(shot.guardrail) for shot in sample.shots)
        ),
        "num_benign_samples": sum(
            1 for sample in samples if all(not _guardrail_true_codes(shot.guardrail) for shot in sample.shots)
        ),
        "stats": dict(stats),
    }
    return samples, summary


def write_jsonl(samples: list[VideoSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_jsonl_record(), ensure_ascii=False) + "\n")


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SafeWatch-Bench-200K shot-localization SFT jsonl.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/gpfs/public/datasets/SafeWatch-Bench-200K"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("artifacts/safewatch_interval_sft/safewatch_interval_sft_60s.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("artifacts/safewatch_interval_sft/safewatch_interval_sft_60s.summary.json"),
    )
    parser.add_argument(
        "--train-output-jsonl",
        type=Path,
        default=Path("artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_train.jsonl"),
    )
    parser.add_argument(
        "--eval-output-jsonl",
        type=Path,
        default=Path("artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_eval.jsonl"),
    )
    parser.add_argument(
        "--split-summary-json",
        type=Path,
        default=Path("artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_split.summary.json"),
    )
    parser.add_argument("--max-duration-sec", type=float, default=60.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--exclude-benign",
        action="store_true",
        help="If set, drop videos that have no harmful shots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    samples, summary = collect_sft_samples(
        dataset_root=args.dataset_root,
        project_root=project_root,
        max_duration_sec=args.max_duration_sec,
        include_benign=not args.exclude_benign,
        max_samples=args.max_samples,
    )
    write_jsonl(samples, args.output_jsonl)
    dataset_summary = {
        **summary,
        **summarize_samples(samples),
    }
    train_samples, eval_samples, split_summary = split_train_eval_by_source_category(
        samples,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    write_jsonl(train_samples, args.train_output_jsonl)
    write_jsonl(eval_samples, args.eval_output_jsonl)
    write_summary(dataset_summary, args.summary_json)
    write_summary(split_summary, args.split_summary_json)
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))
    print(json.dumps(split_summary, indent=2, ensure_ascii=False))
    print(f"Wrote {len(samples)} samples to {args.output_jsonl}")
    print(f"Wrote {len(train_samples)} train samples to {args.train_output_jsonl}")
    print(f"Wrote {len(eval_samples)} eval samples to {args.eval_output_jsonl}")


if __name__ == "__main__":
    main()
