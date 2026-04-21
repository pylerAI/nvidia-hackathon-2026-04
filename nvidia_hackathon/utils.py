from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _probe_duration_with_cv2(video_path: Path) -> float:
    import cv2

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


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(command)}\n\n"
            f"stdout:\n{exc.stdout}\n\n"
            f"stderr:\n{exc.stderr}"
        ) from exc


def ffprobe_duration(video_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return float(result.stdout.strip())
    except FileNotFoundError:
        return _probe_duration_with_cv2(video_path)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)
