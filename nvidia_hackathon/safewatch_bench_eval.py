from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data_curation.build_safewatch_sft_jsonl import parse_safewatch_response

from .inference import NemotronVideoModerator, extract_assistant_response_text
from .prompt_template import build_interval_sft_prompt, video_moderation_prompt_vg_fine_grained_final
from .utils import ffprobe_duration, load_json, save_json
from .video import extract_frames


STANDARD_CATEGORY_CODES = ("C1", "C2", "C3", "C4", "C5", "C6")
DEFAULT_MODEL_PATH = Path("/gpfs/public/artifacts/models/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
DEFAULT_DATASET_ROOT = Path("/gpfs/public/datasets/SafeWatch-Bench")
DEFAULT_OUTPUT_ROOT = Path("artifacts/safewatch_bench_eval")


@dataclass(frozen=True)
class BenchSample:
    sample_id: str
    video_path: str
    benchmark_json: str
    benchmark_name: str
    source_category: str
    duration_sec: float
    gt_categories: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SafeWatch-Bench real videos <=60s with Nemotron using "
            "1) the original SafeWatch video prompt and "
            "2) the interval-shot prompt used for SafeWatch SFT."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-duration-sec", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--video-pruning-rate", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=("full", "interval"), default=("full", "interval"))
    return parser.parse_args()


def load_safewatch_full_video_prompt(_project_root: Path) -> str:
    return video_moderation_prompt_vg_fine_grained_final


def guardrail_true_codes(guardrail: dict[str, bool]) -> list[str]:
    true_codes: list[str] = []
    for code in STANDARD_CATEGORY_CODES:
        matched_key = next((key for key in guardrail if key.startswith(f"{code}(")), None)
        if matched_key and bool(guardrail[matched_key]):
            true_codes.append(code)
    return true_codes


def parse_interval_json_list(text: str) -> list[dict[str, Any]]:
    assistant_text = extract_assistant_response_text(text).strip()
    candidates: list[str] = []
    if assistant_text.startswith("[") and assistant_text.endswith("]"):
        candidates.append(assistant_text)

    for start_index in [index for index, char in enumerate(assistant_text) if char == "["][::-1]:
        candidate = assistant_text[start_index:].strip()
        if not candidate.endswith("]"):
            end_index = assistant_text.rfind("]")
            if end_index > start_index:
                candidate = assistant_text[start_index : end_index + 1].strip()
        if candidate:
            candidates.append(candidate)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return payload
    raise ValueError(f"Could not parse interval JSON list:\n{assistant_text}")


def parse_interval_method_categories(text: str) -> tuple[list[str], str]:
    shot_payload = parse_interval_json_list(text)
    predicted_codes: set[str] = set()
    for shot in shot_payload:
        if not isinstance(shot, dict):
            continue
        guardrail = shot.get("GUARDRAIL")
        if not isinstance(guardrail, dict):
            continue
        normalized_guardrail = {str(key): bool(value) for key, value in guardrail.items()}
        predicted_codes.update(guardrail_true_codes(normalized_guardrail))
    normalized_payload = json.dumps(shot_payload, ensure_ascii=False, indent=2)
    return sorted(predicted_codes), normalized_payload


def parse_full_method_categories(text: str) -> tuple[list[str], str]:
    parsed = parse_safewatch_response(extract_assistant_response_text(text))
    if parsed is None:
        raise ValueError(f"Could not parse SafeWatch full-video response:\n{text}")
    normalized_payload = json.dumps(
        {
            "DESCRIPTION": parsed.description,
            "GUARDRAIL": parsed.guardrail,
            "EXPLANATION": parsed.explanation,
        },
        ensure_ascii=False,
        indent=2,
    )
    return sorted(guardrail_true_codes(parsed.guardrail)), normalized_payload


def collect_bench_samples(
    dataset_root: Path,
    *,
    max_duration_sec: float,
    limit: int | None = None,
) -> tuple[list[BenchSample], dict[str, Any]]:
    real_root = dataset_root / "real"
    samples: list[BenchSample] = []
    stats: dict[str, int] = {
        "included": 0,
        "excluded_over_duration": 0,
        "excluded_missing_video": 0,
    }

    for category in STANDARD_CATEGORY_CODES:
        json_root = real_root / category
        for json_path in sorted(json_root.glob("*.json")):
            items = load_json(json_path)
            is_benign_benchmark = "benign" in json_path.stem
            for index, item in enumerate(items):
                video_path = dataset_root / item["video_path"]
                if not video_path.exists():
                    stats["excluded_missing_video"] += 1
                    continue

                duration_sec = ffprobe_duration(video_path)
                if duration_sec > max_duration_sec:
                    stats["excluded_over_duration"] += 1
                    continue

                gt_categories = [] if is_benign_benchmark or not item.get("labels") else [category]
                sample = BenchSample(
                    sample_id=f"{category}__{json_path.stem}__{index:04d}",
                    video_path=str(video_path),
                    benchmark_json=str(json_path),
                    benchmark_name=json_path.stem,
                    source_category=category,
                    duration_sec=duration_sec,
                    gt_categories=gt_categories,
                )
                samples.append(sample)
                stats["included"] += 1
                if limit is not None and len(samples) >= limit:
                    break
            if limit is not None and len(samples) >= limit:
                break
        if limit is not None and len(samples) >= limit:
            break

    summary = {
        "num_samples": len(samples),
        "counts_by_source_category": {
            category: sum(sample.source_category == category for sample in samples)
            for category in STANDARD_CATEGORY_CODES
        },
        "num_benign_samples": sum(not sample.gt_categories for sample in samples),
        "num_harmful_samples": sum(bool(sample.gt_categories) for sample in samples),
        "stats": stats,
    }
    return samples, summary


def compute_per_category_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_samples": len(results),
        "per_category": {},
    }
    for category in STANDARD_CATEGORY_CODES:
        tp = tn = fp = fn = 0
        for result in results:
            gt_has = category in result["gt_categories"]
            pred_has = category in result["pred_categories"]
            if gt_has and pred_has:
                tp += 1
            elif gt_has and not pred_has:
                fn += 1
            elif not gt_has and pred_has:
                fp += 1
            else:
                tn += 1

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 0.0
        summary["per_category"][category] = {
            "accuracy": accuracy,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    exact_match_accuracy = (
        sum(sorted(result["gt_categories"]) == sorted(result["pred_categories"]) for result in results) / len(results)
        if results
        else 0.0
    )
    summary["exact_match_accuracy"] = exact_match_accuracy
    return summary


def evaluate_method(
    *,
    method: str,
    samples: list[BenchSample],
    prompt_text: str,
    runner: NemotronVideoModerator,
    output_root: Path,
    fps: float,
    max_frames: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    frames_root = output_root / f"{method}_frames"

    for index, sample in enumerate(samples, start=1):
        print(f"[{method}] [{index}/{len(samples)}] {sample.sample_id}")
        frames_dir = frames_root / sample.sample_id
        frames = extract_frames(Path(sample.video_path), frames_dir, fps, max_frames)
        raw_output_full = runner.infer(frames, sample.duration_sec, prompt_text)

        try:
            if method == "full":
                pred_categories, normalized_output = parse_full_method_categories(raw_output_full)
            elif method == "interval":
                pred_categories, normalized_output = parse_interval_method_categories(raw_output_full)
            else:
                raise ValueError(f"Unsupported method: {method}")
            parse_error = None
        except Exception as exc:
            pred_categories = []
            normalized_output = extract_assistant_response_text(raw_output_full)
            parse_error = str(exc)

        results.append(
            {
                "sample_id": sample.sample_id,
                "video_path": sample.video_path,
                "source_category": sample.source_category,
                "benchmark_name": sample.benchmark_name,
                "duration_sec": sample.duration_sec,
                "gt_categories": list(sample.gt_categories),
                "pred_categories": pred_categories,
                "raw_output": normalized_output,
                "parse_error": parse_error,
            }
        )

    metrics = compute_per_category_accuracy(results)
    metrics["num_parse_errors"] = sum(result["parse_error"] is not None for result in results)
    save_json(output_root / f"{method}_predictions.json", results)
    save_json(output_root / f"{method}_metrics.json", metrics)
    return metrics


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    samples, dataset_summary = collect_bench_samples(
        args.dataset_root,
        max_duration_sec=args.max_duration_sec,
        limit=args.limit,
    )
    save_json(args.output_root / "dataset_summary.json", dataset_summary)
    save_json(args.output_root / "dataset_manifest.json", [sample.to_dict() for sample in samples])

    runner = NemotronVideoModerator(
        model_path=args.model_path,
        device=args.device,
        video_pruning_rate=args.video_pruning_rate,
        max_new_tokens=args.max_new_tokens,
    )

    method_metrics: dict[str, Any] = {}
    if "full" in args.methods:
        full_prompt = load_safewatch_full_video_prompt(args.project_root)
        method_metrics["full"] = evaluate_method(
            method="full",
            samples=samples,
            prompt_text=full_prompt,
            runner=runner,
            output_root=args.output_root,
            fps=args.fps,
            max_frames=args.max_frames,
        )
    if "interval" in args.methods:
        interval_prompt = build_interval_sft_prompt(args.project_root)
        method_metrics["interval"] = evaluate_method(
            method="interval",
            samples=samples,
            prompt_text=interval_prompt,
            runner=runner,
            output_root=args.output_root,
            fps=args.fps,
            max_frames=args.max_frames,
        )

    save_json(args.output_root / "summary.json", {"dataset": dataset_summary, "methods": method_metrics})
    print(json.dumps({"dataset": dataset_summary, "methods": method_metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
