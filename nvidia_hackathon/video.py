from __future__ import annotations

import math
import shutil
from pathlib import Path

from PIL import Image

from .utils import ffprobe_duration, run_command


def fps_to_time_step(fps: float) -> float:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return 1.0 / fps


def snap_to_time_grid(value: float, time_step_sec: float, mode: str = "nearest") -> float:
    if time_step_sec <= 0:
        return value

    scaled = value / time_step_sec
    if mode == "floor":
        snapped = math.floor(scaled) * time_step_sec
    elif mode == "ceil":
        snapped = math.ceil(scaled) * time_step_sec
    elif mode == "nearest":
        snapped = round(scaled) * time_step_sec
    else:
        raise ValueError(f"Unsupported snap mode: {mode}")
    return round(snapped, 6)


def snap_interval_to_time_grid(
    start_sec: float,
    end_sec: float,
    clip_duration_sec: float,
    time_step_sec: float,
    *,
    start_mode: str = "nearest",
    end_mode: str = "nearest",
) -> tuple[float, float]:
    start_sec = max(0.0, min(start_sec, clip_duration_sec))
    end_sec = max(0.0, min(end_sec, clip_duration_sec))
    snapped_start = snap_to_time_grid(start_sec, time_step_sec, mode=start_mode)
    snapped_end = snap_to_time_grid(end_sec, time_step_sec, mode=end_mode)
    snapped_start = max(0.0, min(snapped_start, clip_duration_sec))
    snapped_end = max(0.0, min(snapped_end, clip_duration_sec))
    if end_sec > start_sec and snapped_end <= snapped_start:
        snapped_end = min(clip_duration_sec, round(snapped_start + time_step_sec, 6))
    return snapped_start, snapped_end


def sample_safe_clip_start(rng, safe_duration: float, clip_duration: float) -> float:
    max_start = max(safe_duration - clip_duration, 0.0)
    return rng.uniform(0.0, max_start) if max_start > 0 else 0.0


def sample_insert_positions(rng, base_safe_clip_duration_sec: float, insert_count: int, time_step_sec: float) -> list[float]:
    max_insert_index = math.floor(max(base_safe_clip_duration_sec, 0.0) / time_step_sec)
    if insert_count > max_insert_index + 1:
        raise ValueError("Not enough time-grid positions to place inserted clips without overlap.")
    sampled_indices = sorted(rng.sample(range(max_insert_index + 1), insert_count))
    return [round(index * time_step_sec, 6) for index in sampled_indices]


def trim_and_reencode(src: Path, dst: Path, start_sec: float, duration_sec: float) -> None:
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.6f}",
            "-t",
            f"{duration_sec:.6f}",
            "-i",
            str(src),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
    )


def concat_videos(video_paths: list[Path], output_path: Path, work_dir: Path) -> None:
    concat_txt = work_dir / "concat.txt"
    concat_txt.write_text("".join(f"file '{path.as_posix()}'\n" for path in video_paths))
    run_command(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )


def clip_video_to_duration(video_path: Path, max_duration_sec: float) -> bool:
    current_duration_sec = ffprobe_duration(video_path)
    if current_duration_sec <= max_duration_sec + 1e-6:
        return False

    tmp_path = video_path.with_suffix(".trimmed.mp4")
    trim_and_reencode(video_path, tmp_path, 0.0, max_duration_sec)
    tmp_path.replace(video_path)
    return True


def max_video_duration_for_frame_budget(fps: float, max_frames: int) -> float:
    return round(max_frames / fps, 6)


def extract_frames(video_path: Path, frames_dir: Path, fps: float, max_frames: int) -> list[Image.Image]:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-frames:v",
            str(max_frames),
            str(frames_dir / "frame_%05d.jpg"),
        ]
    )

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames were extracted from {video_path}.")

    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))
    return frames
