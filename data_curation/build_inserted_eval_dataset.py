from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from nvidia_hackathon.prompt_template import STANDARD_GUARDRAIL, build_interval_sft_prompt

EPSILON = 1e-6


@dataclass(frozen=True)
class VideoMeta:
    duration_sec: float
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class HarmfulSource:
    sample_id: str
    source_domain: str
    category: str
    subcategory: str
    video_path: Path
    duration_sec: float
    guardrail: dict[str, bool]


@dataclass(frozen=True)
class Interval:
    start_sec: float
    end_sec: float
    guardrail: dict[str, bool]
    description: str = "None"
    explanation: str = "None"

    def to_dict(self) -> dict[str, Any]:
        return {
            "START_SEC": round(self.start_sec, 3),
            "END_SEC": round(self.end_sec, 3),
            "DESCRIPTION": self.description,
            "GUARDRAIL": self.guardrail,
            "EXPLANATION": self.explanation,
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def zero_guardrail() -> dict[str, bool]:
    return dict(STANDARD_GUARDRAIL)


def has_harm(guardrail: dict[str, bool]) -> bool:
    return any(bool(guardrail.get(key, False)) for key in STANDARD_GUARDRAIL)


def normalize_guardrail(raw_guardrail: dict[str, Any]) -> dict[str, bool]:
    normalized = zero_guardrail()
    for key in normalized:
        normalized[key] = bool(raw_guardrail.get(key, False))
    return normalized


def aggregate_guardrail(shots: list[dict[str, Any]]) -> dict[str, bool]:
    aggregated = zero_guardrail()
    for shot in shots:
        guardrail = normalize_guardrail(shot.get("GUARDRAIL", {}))
        for key, value in guardrail.items():
            aggregated[key] = aggregated[key] or value
    return aggregated


def parse_harmful_source(record: dict[str, Any], *, max_harmful_duration_sec: float) -> HarmfulSource | None:
    video_path = Path(record["video"])
    try:
        benchmark_dir = video_path.parent.name
        category = video_path.parent.parent.name
        source_domain = video_path.parts[video_path.parts.index("SafeWatch-Bench") + 1]
    except (ValueError, IndexError):
        return None

    if not benchmark_dir.endswith("_benchmark"):
        return None

    subcategory = benchmark_dir.removesuffix("_benchmark")
    duration_sec = float(record.get("metadata", {}).get("duration_sec", 0.0))
    if duration_sec <= 0 or duration_sec > max_harmful_duration_sec:
        return None

    conversations = record.get("conversations", [])
    if len(conversations) < 2:
        return None

    try:
        shots = json.loads(conversations[1]["value"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None

    if not isinstance(shots, list):
        return None

    guardrail = aggregate_guardrail(shots)
    if not has_harm(guardrail):
        return None

    return HarmfulSource(
        sample_id=str(record["id"]),
        source_domain=source_domain,
        category=category,
        subcategory=subcategory,
        video_path=video_path,
        duration_sec=duration_sec,
        guardrail=guardrail,
    )


def collect_harmful_sources(
    harmful_jsonl_path: Path,
    *,
    max_harmful_duration_sec: float,
    seed: int,
    max_samples_per_subcategory: int | None,
    max_samples: int | None,
) -> tuple[list[HarmfulSource], dict[str, int]]:
    rng = random.Random(seed)
    stats = Counter()
    sources_by_subcategory: dict[str, list[HarmfulSource]] = defaultdict(list)

    for record in load_jsonl(harmful_jsonl_path):
        parsed = parse_harmful_source(record, max_harmful_duration_sec=max_harmful_duration_sec)
        if parsed is None:
            stats["skipped_non_harmful_or_invalid"] += 1
            continue
        if not parsed.video_path.exists():
            stats["skipped_missing_harmful_video"] += 1
            continue
        sources_by_subcategory[parsed.subcategory].append(parsed)
        stats["included_harmful_sources"] += 1

    sources: list[HarmfulSource] = []
    for subcategory in sorted(sources_by_subcategory):
        bucket = sources_by_subcategory[subcategory]
        rng.shuffle(bucket)
        original_count = len(bucket)
        if max_samples_per_subcategory is not None:
            bucket = bucket[:max_samples_per_subcategory]
        stats[f"selected_{subcategory}"] = len(bucket)
        stats[f"dropped_{subcategory}"] = max(0, original_count - len(bucket))
        sources.extend(bucket)

    rng.shuffle(sources)
    if max_samples is not None:
        sources = sources[:max_samples]
    return sources, dict(stats)


def collect_safe_video_paths(safe_root: Path) -> list[Path]:
    return sorted(path for path in safe_root.glob("*/content.mp4") if path.is_file())


def probe_video_meta(video_path: Path) -> VideoMeta:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid video metadata for {video_path}: fps={fps}, frame_count={frame_count}")
        return VideoMeta(
            duration_sec=frame_count / fps,
            width=width,
            height=height,
            fps=fps,
        )
    finally:
        capture.release()


def pick_safe_video(
    safe_paths: list[Path],
    *,
    rng: random.Random,
    meta_cache: dict[Path, VideoMeta],
    bad_paths: set[Path],
) -> tuple[Path, VideoMeta]:
    if len(bad_paths) >= len(safe_paths):
        raise RuntimeError("No readable safe videos are available.")

    while True:
        safe_path = rng.choice(safe_paths)
        if safe_path in bad_paths:
            continue
        if safe_path not in meta_cache:
            try:
                meta_cache[safe_path] = probe_video_meta(safe_path)
            except RuntimeError:
                bad_paths.add(safe_path)
                continue
        meta = meta_cache[safe_path]
        if meta.duration_sec <= 0:
            bad_paths.add(safe_path)
            continue
        return safe_path, meta


def sanitize_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "item"


def build_sample_id(safe_video_id: str, harmful: HarmfulSource) -> str:
    return "__".join(
        [
            "inserted",
            sanitize_token(safe_video_id),
            sanitize_token(harmful.source_domain),
            sanitize_token(harmful.category),
            sanitize_token(harmful.subcategory),
            sanitize_token(harmful.video_path.stem),
        ]
    )


def choose_insertion_sec(
    *,
    safe_duration_sec: float,
    harmful_duration_sec: float,
    max_output_duration_sec: float,
    rng: random.Random,
) -> float:
    max_start_sec = max(0.0, min(safe_duration_sec, max_output_duration_sec - harmful_duration_sec))
    if max_start_sec <= EPSILON:
        return 0.0
    return round(rng.uniform(0.0, max_start_sec), 3)


def build_inserted_intervals(
    *,
    insert_start_sec: float,
    harmful_duration_sec: float,
    output_duration_sec: float,
    harmful_guardrail: dict[str, bool],
) -> list[Interval]:
    insert_end_sec = min(output_duration_sec, insert_start_sec + harmful_duration_sec)
    if insert_end_sec - insert_start_sec <= EPSILON:
        raise ValueError("Inserted harmful segment is empty after truncation.")

    intervals: list[Interval] = []
    if insert_start_sec > EPSILON:
        intervals.append(
            Interval(
                start_sec=0.0,
                end_sec=insert_start_sec,
                guardrail=zero_guardrail(),
            )
        )

    intervals.append(
        Interval(
            start_sec=insert_start_sec,
            end_sec=insert_end_sec,
            guardrail=dict(harmful_guardrail),
        )
    )

    if output_duration_sec - insert_end_sec > EPSILON:
        intervals.append(
            Interval(
                start_sec=insert_end_sec,
                end_sec=output_duration_sec,
                guardrail=zero_guardrail(),
            )
        )

    return intervals


def build_assistant_response(intervals: list[Interval]) -> str:
    return json.dumps([interval.to_dict() for interval in intervals], indent=2, ensure_ascii=False)


def _even_dimension(value: int, fallback: int) -> int:
    if value <= 0:
        value = fallback
    return value if value % 2 == 0 else value - 1


def _output_fps(fps: float) -> int:
    if fps <= 0:
        return 30
    return max(1, min(60, int(round(fps))))


def render_inserted_video(
    *,
    safe_video_path: Path,
    harmful_video_path: Path,
    safe_meta: VideoMeta,
    harmful_duration_sec: float,
    insert_start_sec: float,
    output_duration_sec: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = _even_dimension(safe_meta.width, 1280)
    height = _even_dimension(safe_meta.height, 720)
    fps = _output_fps(safe_meta.fps)

    segments: list[tuple[int, float, float]] = []
    if insert_start_sec > EPSILON:
        segments.append((0, 0.0, insert_start_sec))
    segments.append((1, 0.0, harmful_duration_sec))
    if safe_meta.duration_sec - insert_start_sec > EPSILON:
        segments.append((0, insert_start_sec, safe_meta.duration_sec))

    filter_parts: list[str] = []
    concat_labels: list[str] = []
    for index, (input_index, start_sec, end_sec) in enumerate(segments):
        label = f"v{index}"
        filter_parts.append(
            (
                f"[{input_index}:v]"
                f"trim=start={start_sec:.3f}:end={end_sec:.3f},"
                f"setpts=PTS-STARTPTS,"
                f"fps={fps},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,"
                f"format=yuv420p"
                f"[{label}]"
            )
        )
        concat_labels.append(f"[{label}]")

    filter_parts.append(
        "".join(concat_labels)
        + f"concat=n={len(concat_labels)}:v=1:a=0,"
        + f"trim=start=0:end={output_duration_sec:.3f},"
        + "setpts=PTS-STARTPTS[outv]"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(safe_video_path),
        "-i",
        str(harmful_video_path),
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[outv]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-20:])
        raise RuntimeError(f"ffmpeg failed for {output_path}:\n{stderr_tail}")


def build_output_record(
    *,
    sample_id: str,
    output_video_path: Path,
    output_video_relative_path: str,
    prompt: str,
    intervals: list[Interval],
    output_duration_sec: float,
    safe_video_path: Path,
    harmful: HarmfulSource,
) -> dict[str, Any]:
    harmful_interval = next(interval for interval in intervals if has_harm(interval.guardrail))
    num_harmful_shots = sum(1 for interval in intervals if has_harm(interval.guardrail))
    assistant_response = build_assistant_response(intervals)
    return {
        "id": sample_id,
        "task_name": "safewatch-shot-sft",
        "video": str(output_video_path),
        "videos": [str(output_video_path)],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(output_video_path)},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": assistant_response,
            },
        ],
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\n{prompt}",
            },
            {
                "from": "gpt",
                "value": assistant_response,
            },
        ],
        "metadata": {
            "video_relative_path": output_video_relative_path,
            "duration_sec": round(output_duration_sec, 3),
            "source_category": f"inserted_{harmful.source_domain}_{harmful.category}_{harmful.subcategory}",
            "num_violations": num_harmful_shots,
            "num_shots": len(intervals),
            "num_harmful_shots": num_harmful_shots,
            "num_benign_shots": len(intervals) - num_harmful_shots,
            "safe_source_video": str(safe_video_path),
            "harmful_source_video": str(harmful.video_path),
            "harmful_source_domain": harmful.source_domain,
            "harmful_source_category": harmful.category,
            "harmful_source_subcategory": harmful.subcategory,
            "insert_start_sec": round(harmful_interval.start_sec, 3),
            "insert_end_sec": round(harmful_interval.end_sec, 3),
        },
    }


def build_summary(records: list[dict[str, Any]], stats: Counter, *, output_jsonl: Path) -> dict[str, Any]:
    by_domain = Counter()
    by_category = Counter()
    by_subcategory = Counter()
    for record in records:
        metadata = record["metadata"]
        by_domain[metadata["harmful_source_domain"]] += 1
        by_category[metadata["harmful_source_category"]] += 1
        by_subcategory[metadata["harmful_source_subcategory"]] += 1

    return {
        "num_records": len(records),
        "output_jsonl": str(output_jsonl),
        "stats": dict(stats),
        "source_domain_counts": dict(sorted(by_domain.items())),
        "category_counts": dict(sorted(by_category.items())),
        "subcategory_counts": dict(sorted(by_subcategory.items())),
    }


def generate_inserted_eval_dataset(
    *,
    safe_root: Path,
    harmful_jsonl_path: Path,
    output_root: Path,
    output_jsonl: Path,
    summary_json: Path,
    max_output_duration_sec: float,
    max_harmful_duration_sec: float,
    seed: int,
    max_samples_per_subcategory: int | None,
    max_samples: int | None,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    safe_paths = collect_safe_video_paths(safe_root)
    if not safe_paths:
        raise FileNotFoundError(f"No safe videos found under {safe_root}")

    harmful_sources, harmful_stats = collect_harmful_sources(
        harmful_jsonl_path,
        max_harmful_duration_sec=max_harmful_duration_sec,
        seed=seed,
        max_samples_per_subcategory=max_samples_per_subcategory,
        max_samples=max_samples,
    )
    if not harmful_sources:
        raise RuntimeError("No eligible harmful source videos were found.")

    stats = Counter(harmful_stats)
    meta_cache: dict[Path, VideoMeta] = {}
    bad_safe_paths: set[Path] = set()
    rng = random.Random(seed)
    prompt = build_interval_sft_prompt(Path(__file__).resolve().parents[1])

    records: list[dict[str, Any]] = []
    for harmful in harmful_sources:
        safe_video_path, safe_meta = pick_safe_video(
            safe_paths,
            rng=rng,
            meta_cache=meta_cache,
            bad_paths=bad_safe_paths,
        )
        safe_video_id = safe_video_path.parent.name
        sample_id = build_sample_id(safe_video_id, harmful)
        output_filename = f"{sample_id}.mp4"
        output_video_path = output_root / harmful.subcategory / output_filename
        output_video_relative_path = str(output_video_path.relative_to(output_root.parent))
        insert_start_sec = choose_insertion_sec(
            safe_duration_sec=safe_meta.duration_sec,
            harmful_duration_sec=harmful.duration_sec,
            max_output_duration_sec=max_output_duration_sec,
            rng=rng,
        )
        output_duration_sec = min(max_output_duration_sec, safe_meta.duration_sec + harmful.duration_sec)
        intervals = build_inserted_intervals(
            insert_start_sec=insert_start_sec,
            harmful_duration_sec=harmful.duration_sec,
            output_duration_sec=output_duration_sec,
            harmful_guardrail=harmful.guardrail,
        )

        if not dry_run and (overwrite or not output_video_path.exists()):
            render_inserted_video(
                safe_video_path=safe_video_path,
                harmful_video_path=harmful.video_path,
                safe_meta=safe_meta,
                harmful_duration_sec=harmful.duration_sec,
                insert_start_sec=insert_start_sec,
                output_duration_sec=output_duration_sec,
                output_path=output_video_path,
            )
            stats["rendered_videos"] += 1
        elif output_video_path.exists():
            stats["reused_existing_videos"] += 1
        else:
            stats["skipped_render_dry_run"] += 1

        records.append(
            build_output_record(
                sample_id=sample_id,
                output_video_path=output_video_path,
                output_video_relative_path=output_video_relative_path,
                prompt=prompt,
                intervals=intervals,
                output_duration_sec=output_duration_sec,
                safe_video_path=safe_video_path,
                harmful=harmful,
            )
        )
        stats["written_records"] += 1

    write_jsonl(records, output_jsonl)
    summary = build_summary(records, stats, output_jsonl=output_jsonl)
    write_summary(summary, summary_json)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert harmful clips into safe videos and build eval jsonl.")
    parser.add_argument(
        "--safe-root",
        type=Path,
        default=Path("/gpfs/public/datasets/Video-MME/processed_data/short/video"),
    )
    parser.add_argument(
        "--harmful-jsonl",
        type=Path,
        default=Path("/gpfs/public/datasets/SafeWatch-Bench/guardrail_safewatch_sft_shot_under60s.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos_eval.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos_eval.summary.json"),
    )
    parser.add_argument("--max-output-duration-sec", type=float, default=60.0)
    parser.add_argument("--max-harmful-duration-sec", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples-per-subcategory", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_inserted_eval_dataset(
        safe_root=args.safe_root,
        harmful_jsonl_path=args.harmful_jsonl,
        output_root=args.output_root,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        max_output_duration_sec=args.max_output_duration_sec,
        max_harmful_duration_sec=args.max_harmful_duration_sec,
        seed=args.seed,
        max_samples_per_subcategory=args.max_samples_per_subcategory,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSample counts by subcategory:")
    for subcategory, count in summary["subcategory_counts"].items():
        print(f"- {subcategory}: {count}")


if __name__ == "__main__":
    main()
