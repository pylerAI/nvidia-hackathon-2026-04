from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

from .schemas import CATEGORIES, CATEGORY_PAIR_COMBOS, CandidateClip, HarmInterval, InsertedSegment, SyntheticSample
from .utils import ffprobe_duration, load_json, save_json
from .video import (
    clip_video_to_duration,
    concat_videos,
    fps_to_time_step,
    max_video_duration_for_frame_budget,
    sample_insert_positions,
    sample_safe_clip_start,
    snap_interval_to_time_grid,
    trim_and_reencode,
)


def load_harmful_candidates(
    safewatch_root: Path,
    *,
    categories: tuple[str, ...] = CATEGORIES,
    max_insert_duration: float | None = None,
) -> dict[str, list[CandidateClip]]:
    real_root = safewatch_root / "real"
    candidate_pools: dict[str, list[CandidateClip]] = {category: [] for category in categories}

    for category in categories:
        json_root = real_root / category
        for json_path in sorted(json_root.glob("*.json")):
            if "benign" in json_path.stem:
                continue
            items = load_json(json_path)
            for item in items:
                if not item.get("labels"):
                    continue
                video_path = safewatch_root / item["video_path"]
                if not video_path.exists():
                    continue
                duration_sec = ffprobe_duration(video_path)
                if max_insert_duration is not None and duration_sec > max_insert_duration:
                    continue
                candidate_pools[category].append(
                    CandidateClip(
                        category=category,
                        video_path=str(video_path),
                        duration_sec=duration_sec,
                        benchmark_name=json_path.stem,
                        subcategories=list(item.get("subcategories", [])),
                        source_labels=list(item.get("labels", [])),
                        source_description=item.get("video_content"),
                        violate_reason=item.get("violate_reason"),
                    )
                )

        if not candidate_pools[category]:
            raise RuntimeError(f"No harmful candidate clips found for {category}.")

    return candidate_pools


def list_safe_videos(safe_root: Path, min_safe_clip_duration: float) -> list[dict[str, float | str]]:
    safe_videos: list[dict[str, float | str]] = []
    for video_dir in sorted(safe_root.iterdir()):
        content_path = video_dir / "content.mp4"
        if not content_path.exists():
            continue
        duration_sec = ffprobe_duration(content_path)
        if duration_sec < min_safe_clip_duration:
            continue
        safe_videos.append({"video_path": str(content_path), "duration_sec": duration_sec})

    if not safe_videos:
        raise RuntimeError("No usable safe videos were found.")
    return safe_videos


def build_insertion_schedule(
    *,
    num_zero_insert: int,
    num_single_per_category: int,
    num_pair_per_combo: int,
    seed: int,
) -> list[list[str]]:
    rng = random.Random(seed)
    schedule: list[list[str]] = []
    schedule.extend([[] for _ in range(num_zero_insert)])
    for category in CATEGORIES:
        schedule.extend([[category] for _ in range(num_single_per_category)])
    for left, right in CATEGORY_PAIR_COMBOS:
        schedule.extend([[left, right] for _ in range(num_pair_per_combo)])
    rng.shuffle(schedule)
    return schedule


def summarize_schedule(schedule: list[list[str]]) -> dict[str, int]:
    harmful_counter = Counter(category for plan in schedule for category in plan)
    insert_count_counter = Counter(len(plan) for plan in schedule)
    summary = {
        "num_samples": len(schedule),
        "num_zero_insert": insert_count_counter.get(0, 0),
        "num_one_insert": insert_count_counter.get(1, 0),
        "num_two_insert": insert_count_counter.get(2, 0),
    }
    summary.update({f"{category}_segments": harmful_counter.get(category, 0) for category in CATEGORIES})
    return summary


def _build_single_sample(
    *,
    sample_id: str,
    planned_categories: list[str],
    safe_item: dict[str, float | str],
    candidate_pools: dict[str, list[CandidateClip]],
    base_safe_clip_duration_sec: float,
    fps: float,
    max_frames: int,
    output_root: Path,
    rng: random.Random,
) -> SyntheticSample:
    sample_dir = output_root / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = sample_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safe_source_path = Path(str(safe_item["video_path"]))
    safe_source_duration_sec = float(safe_item["duration_sec"])
    base_safe_duration_sec = min(base_safe_clip_duration_sec, safe_source_duration_sec)
    safe_clip_start_sec = sample_safe_clip_start(rng, safe_source_duration_sec, base_safe_duration_sec)
    time_step_sec = fps_to_time_step(fps)
    max_duration_sec = max_video_duration_for_frame_budget(fps, max_frames)

    candidate_clips = [rng.choice(candidate_pools[category]) for category in planned_categories]
    base_insert_positions = sample_insert_positions(
        rng,
        base_safe_duration_sec,
        len(candidate_clips),
        time_step_sec,
    )
    insert_plan = sorted(zip(base_insert_positions, candidate_clips), key=lambda item: item[0])

    segment_paths: list[Path] = []
    raw_intervals: list[tuple[float, float, CandidateClip, float]] = []
    safe_cursor_sec = 0.0
    cumulative_inserted_sec = 0.0

    for insert_index, (base_insert_position_sec, candidate_clip) in enumerate(insert_plan):
        safe_chunk_duration_sec = base_insert_position_sec - safe_cursor_sec
        if safe_chunk_duration_sec > 0.05:
            safe_chunk_path = tmp_dir / f"safe_{insert_index:02d}.mp4"
            trim_and_reencode(
                safe_source_path,
                safe_chunk_path,
                safe_clip_start_sec + safe_cursor_sec,
                safe_chunk_duration_sec,
            )
            segment_paths.append(safe_chunk_path)

        inserted_chunk_path = tmp_dir / f"insert_{insert_index:02d}.mp4"
        trim_and_reencode(Path(candidate_clip.video_path), inserted_chunk_path, 0.0, candidate_clip.duration_sec)
        segment_paths.append(inserted_chunk_path)

        interval_start_sec = base_insert_position_sec + cumulative_inserted_sec
        interval_end_sec = interval_start_sec + candidate_clip.duration_sec
        raw_intervals.append((interval_start_sec, interval_end_sec, candidate_clip, base_insert_position_sec))

        cumulative_inserted_sec += candidate_clip.duration_sec
        safe_cursor_sec = base_insert_position_sec

    trailing_safe_duration_sec = base_safe_duration_sec - safe_cursor_sec
    if trailing_safe_duration_sec > 0.05:
        trailing_safe_path = tmp_dir / "safe_tail.mp4"
        trim_and_reencode(
            safe_source_path,
            trailing_safe_path,
            safe_clip_start_sec + safe_cursor_sec,
            trailing_safe_duration_sec,
        )
        segment_paths.append(trailing_safe_path)

    synthetic_video_path = sample_dir / "mixed.mp4"
    if segment_paths:
        concat_videos(segment_paths, synthetic_video_path, tmp_dir)
    else:
        trim_and_reencode(safe_source_path, synthetic_video_path, safe_clip_start_sec, base_safe_duration_sec)

    was_tail_trimmed = clip_video_to_duration(synthetic_video_path, max_duration_sec)
    clip_duration_sec = ffprobe_duration(synthetic_video_path)

    inserted_segments: list[InsertedSegment] = []
    gt_intervals: list[HarmInterval] = []
    gt_categories: list[str] = []
    for interval_start_sec, interval_end_sec, candidate_clip, base_insert_position_sec in raw_intervals:
        snapped_start_sec, snapped_end_sec = snap_interval_to_time_grid(
            interval_start_sec,
            interval_end_sec,
            clip_duration_sec,
            time_step_sec,
            start_mode="nearest",
            end_mode="ceil",
        )
        if snapped_end_sec <= snapped_start_sec:
            continue

        interval = HarmInterval(
            start_sec=snapped_start_sec,
            end_sec=snapped_end_sec,
            category=candidate_clip.category,
        )
        inserted_segments.append(
            InsertedSegment(
                category=candidate_clip.category,
                source_path=candidate_clip.video_path,
                source_benchmark=candidate_clip.benchmark_name,
                source_subcategories=list(candidate_clip.subcategories),
                source_duration_sec=candidate_clip.duration_sec,
                base_insert_position_sec=base_insert_position_sec,
                interval=interval,
            )
        )
        gt_intervals.append(interval)
        gt_categories.append(candidate_clip.category)

    gt_categories = sorted(set(gt_categories))
    gt_intervals.sort(key=lambda interval: interval.start_sec)
    inserted_segments.sort(key=lambda segment: segment.interval.start_sec)

    return SyntheticSample(
        sample_id=sample_id,
        insert_count=len(planned_categories),
        planned_categories=list(planned_categories),
        synthetic_video_path=str(synthetic_video_path),
        safe_source_path=str(safe_source_path),
        safe_source_duration_sec=safe_source_duration_sec,
        safe_clip_start_sec=safe_clip_start_sec,
        base_safe_clip_duration_sec=base_safe_duration_sec,
        clip_duration_sec=clip_duration_sec,
        max_duration_sec=max_duration_sec,
        was_tail_trimmed=was_tail_trimmed,
        inserted_segments=inserted_segments,
        gt_categories=gt_categories,
        gt_intervals=gt_intervals,
    )


def validate_dataset_balance(samples: list[SyntheticSample], expected_schedule: list[list[str]]) -> dict[str, int]:
    expected_counter = Counter(category for plan in expected_schedule for category in plan)
    actual_counter = Counter(category for sample in samples for category in sample.gt_categories)
    expected_insert_counts = Counter(len(plan) for plan in expected_schedule)
    actual_insert_counts = Counter(sample.insert_count for sample in samples)

    if len(samples) != len(expected_schedule):
        raise RuntimeError("Prepared dataset size does not match the expected schedule size.")
    if actual_insert_counts != expected_insert_counts:
        raise RuntimeError(f"Insert-count balance mismatch: expected {expected_insert_counts}, got {actual_insert_counts}")
    for category in CATEGORIES:
        if actual_counter[category] != expected_counter[category]:
            raise RuntimeError(
                f"Category balance mismatch for {category}: expected {expected_counter[category]}, got {actual_counter[category]}"
            )

    return {
        "num_samples": len(samples),
        "num_zero_insert": actual_insert_counts.get(0, 0),
        "num_one_insert": actual_insert_counts.get(1, 0),
        "num_two_insert": actual_insert_counts.get(2, 0),
        **{f"{category}_segments": actual_counter.get(category, 0) for category in CATEGORIES},
    }


def prepare_dataset(
    *,
    output_root: Path,
    manifest_path: Path,
    schedule_summary_path: Path,
    safe_root: Path,
    safewatch_root: Path,
    num_zero_insert: int,
    num_single_per_category: int,
    num_pair_per_combo: int,
    base_safe_clip_duration_sec: float,
    min_safe_clip_duration: float,
    max_insert_duration: float | None,
    fps: float,
    max_frames: int,
    seed: int,
) -> list[SyntheticSample]:
    rng = random.Random(seed)
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_pools = load_harmful_candidates(
        safewatch_root,
        max_insert_duration=max_insert_duration,
    )
    safe_videos = list_safe_videos(safe_root, min_safe_clip_duration)
    schedule = build_insertion_schedule(
        num_zero_insert=num_zero_insert,
        num_single_per_category=num_single_per_category,
        num_pair_per_combo=num_pair_per_combo,
        seed=seed,
    )

    samples: list[SyntheticSample] = []
    for sample_index, planned_categories in enumerate(schedule):
        sample_id = f"sample_{sample_index:03d}"
        accepted_sample: SyntheticSample | None = None
        for _attempt in range(50):
            safe_item = rng.choice(safe_videos)
            candidate_pools_for_sample = candidate_pools
            sample = _build_single_sample(
                sample_id=sample_id,
                planned_categories=planned_categories,
                safe_item=safe_item,
                candidate_pools=candidate_pools_for_sample,
                base_safe_clip_duration_sec=base_safe_clip_duration_sec,
                fps=fps,
                max_frames=max_frames,
                output_root=output_root,
                rng=rng,
            )
            if sample.gt_categories == sorted(set(planned_categories)) and len(sample.inserted_segments) == len(planned_categories):
                accepted_sample = sample
                break
        if accepted_sample is None:
            raise RuntimeError(f"Could not build a valid sample for plan {planned_categories} after multiple attempts.")
        samples.append(accepted_sample)

    schedule_summary = {
        "planned": summarize_schedule(schedule),
        "prepared": validate_dataset_balance(samples, schedule),
    }
    save_json(manifest_path, [sample.to_dict() for sample in samples])
    save_json(schedule_summary_path, schedule_summary)
    return samples
