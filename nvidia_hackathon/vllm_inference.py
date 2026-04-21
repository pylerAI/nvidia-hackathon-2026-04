from __future__ import annotations

import argparse
import base64
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal

from .utils import run_command
from .vllm_server import DEFAULT_VLLM_BASE_URL, DEFAULT_VLLM_MODEL

MediaUrlMode = Literal["file_url", "data_url"]


def frame_directory_name(sample_id: str, *, prefix_chars: int = 40) -> str:
    safe_prefix = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in sample_id)
    safe_prefix = safe_prefix[:prefix_chars].rstrip("_") or "sample"
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe_prefix}_{digest}"


def list_frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(path for path in frames_dir.glob("frame_*.jpg") if path.is_file())


def extract_frames_to_directory(
    *,
    video_path: Path,
    frames_dir: Path,
    fps: float,
    max_frames: int,
    overwrite: bool = False,
) -> list[Path]:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    existing_paths = list_frame_paths(frames_dir)
    if existing_paths and not overwrite:
        return existing_paths

    if frames_dir.exists() and overwrite:
        for existing_path in frames_dir.glob("*"):
            if existing_path.is_file():
                existing_path.unlink()
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
    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frame_paths


def ensure_sample_frames(
    *,
    sample_id: str,
    video_path: Path,
    frames_root: Path,
    fps: float,
    max_frames: int,
    overwrite: bool = False,
) -> tuple[Path, list[Path]]:
    frames_dir = frames_root / frame_directory_name(sample_id)
    frame_paths = extract_frames_to_directory(
        video_path=video_path,
        frames_dir=frames_dir,
        fps=fps,
        max_frames=max_frames,
        overwrite=overwrite,
    )
    return frames_dir, frame_paths


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def _data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def frame_paths_to_message_content(frame_paths: list[Path], *, media_url_mode: MediaUrlMode) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for frame_path in frame_paths:
        if media_url_mode == "file_url":
            url = _file_url(frame_path)
        elif media_url_mode == "data_url":
            url = _data_url(frame_path)
        else:
            raise ValueError(f"Unsupported media_url_mode: {media_url_mode}")
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Unexpected vLLM response shape: {json.dumps(payload, ensure_ascii=False)}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_fragments = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        return "\n".join(fragment for fragment in text_fragments if fragment).strip()
    raise RuntimeError(f"Unsupported vLLM message content: {content!r}")


class VLLMFrameClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_VLLM_BASE_URL,
        model_name: str = DEFAULT_VLLM_MODEL,
        api_key: str = "EMPTY",
        timeout_sec: float = 120.0,
        max_completion_tokens: int = 1024,
        temperature: float = 0.0,
        media_url_mode: MediaUrlMode = "file_url",
        include_no_think_system_prompt: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.media_url_mode = media_url_mode
        self.include_no_think_system_prompt = include_no_think_system_prompt

    @property
    def completions_url(self) -> str:
        return self.base_url + "/chat/completions"

    def build_request_payload(self, frame_paths: list[Path], prompt_text: str) -> dict[str, Any]:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        user_content.extend(frame_paths_to_message_content(frame_paths, media_url_mode=self.media_url_mode))
        messages: list[dict[str, Any]] = []
        if self.include_no_think_system_prompt:
            messages.append({"role": "system", "content": "/no_think"})
        messages.append({"role": "user", "content": user_content})
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }

    def infer(self, frame_paths: list[Path], prompt_text: str) -> str:
        payload = self.build_request_payload(frame_paths, prompt_text)
        request = urllib.request.Request(
            self.completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                raw_payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM request failed with status {exc.code} at {self.completions_url}:\n{error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach vLLM server at {self.completions_url}: {exc}") from exc

        response_payload = json.loads(raw_payload)
        return _extract_message_content(response_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a frame directory and prompt to a vLLM multimodal server.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--media-url-mode", choices=("file_url", "data_url"), default="file_url")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_paths = list_frame_paths(args.frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No extracted frames found in {args.frames_dir}")
    client = VLLMFrameClient(
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.api_key,
        timeout_sec=args.timeout_sec,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        media_url_mode=args.media_url_mode,
    )
    print(client.infer(frame_paths, args.prompt))


if __name__ == "__main__":
    main()
