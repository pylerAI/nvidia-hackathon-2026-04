from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
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


def sample_times_1fps(duration_sec: float) -> list[float]:
    if duration_sec <= 0:
        return [0.0]
    sampled_times = []
    current = 0.0
    while current < duration_sec:
        sampled_times.append(round(current, 6))
        current += 1.0
    return sampled_times or [0.0]


def extract_frames_1fps(video_path: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid video metadata for {video_path}: fps={fps}, frame_count={frame_count}")
        duration_sec = frame_count / fps
        frame_paths: list[str] = []
        for index, sample_time in enumerate(sample_times_1fps(duration_sec), start=1):
            frame_index = min(int(round(sample_time * fps)), max(int(frame_count) - 1, 0))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue
            frame_path = output_dir / f"frame_{index:04d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to write frame: {frame_path}")
            frame_paths.append(str(frame_path))
        if not frame_paths:
            raise RuntimeError(f"No frames extracted from {video_path}")
        return frame_paths
    finally:
        capture.release()


def _extract_prompt_from_conversations(record: dict[str, Any]) -> str:
    conversations = record.get("conversations", [])
    if not conversations:
        raise ValueError("Record has no conversations field.")
    human_value = conversations[0].get("value", "")
    if human_value.startswith("<video>\n"):
        return human_value[len("<video>\n") :]
    return human_value.replace("<video>", "", 1).lstrip()


def frame_directory_name(sample_id: str, *, prefix_chars: int = 40) -> str:
    safe_prefix = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in sample_id)
    safe_prefix = safe_prefix[:prefix_chars].rstrip("_") or "sample"
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe_prefix}_{digest}"


def build_image_eval_record(record: dict[str, Any], frame_paths: list[str], frame_dir: str) -> dict[str, Any]:
    prompt = _extract_prompt_from_conversations(record)
    image_tokens = " ".join(["<image>"] * len(frame_paths))
    human_value = f"{image_tokens}\n{prompt}" if prompt else image_tokens
    assistant_value = record["conversations"][1]["value"]
    metadata = dict(record.get("metadata", {}))
    metadata["frame_dir"] = frame_dir
    metadata["num_frames"] = len(frame_paths)
    metadata["source_video"] = record.get("video")
    return {
        "id": record["id"],
        "images": frame_paths,
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": assistant_value},
        ],
        "metadata": metadata,
    }


def convert_eval_videos_to_frames(
    *,
    eval_jsonl_path: Path,
    frames_root: Path,
    output_jsonl_path: Path,
    summary_json_path: Path,
) -> dict[str, Any]:
    records = load_jsonl(eval_jsonl_path)
    image_records: list[dict[str, Any]] = []
    sample_counter = Counter()

    for record in records:
        sample_id = record["id"]
        source_category = record.get("metadata", {}).get("source_category", "unknown")
        video_path = Path(record["video"])
        frame_dir = frames_root / frame_directory_name(sample_id)
        frame_paths = extract_frames_1fps(video_path, frame_dir)
        image_records.append(build_image_eval_record(record, frame_paths, str(frame_dir)))
        sample_counter[source_category] += 1

    write_jsonl(image_records, output_jsonl_path)
    summary = {
        "num_eval_records": len(image_records),
        "num_frame_directories": len(image_records),
        "source_category_counts": dict(sorted(sample_counter.items())),
        "frames_root": str(frames_root),
        "output_jsonl": str(output_jsonl_path),
    }
    write_summary(summary, summary_json_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 1 FPS frames for SafeWatch eval set only.")
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=Path("artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_eval.jsonl"),
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/eval_frames"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_eval_images_1fps.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_eval_images_1fps.summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_eval_videos_to_frames(
        eval_jsonl_path=args.eval_jsonl,
        frames_root=args.frames_root,
        output_jsonl_path=args.output_jsonl,
        summary_json_path=args.summary_json,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
