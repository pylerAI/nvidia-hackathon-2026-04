from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_VLLM_MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"


def build_vllm_serve_command(
    *,
    model_name: str = DEFAULT_VLLM_MODEL,
    host: str = "127.0.0.1",
    port: int = 8000,
    quantization: str = "modelopt",
    video_pruning_rate: float = 0.0,
    trust_remote_code: bool = True,
    allowed_local_media_path: Path | None = None,
    limit_mm_per_prompt_image: int | None = None,
    max_model_len: int | None = None,
    served_model_name: str | None = None,
) -> list[str]:
    command = [
        "vllm",
        "serve",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--quantization",
        quantization,
        "--video-pruning-rate",
        str(video_pruning_rate),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    if allowed_local_media_path is not None:
        command.extend(["--allowed-local-media-path", str(allowed_local_media_path)])
    if limit_mm_per_prompt_image is not None:
        command.extend(["--limit-mm-per-prompt.image", str(limit_mm_per_prompt_image)])
    if max_model_len is not None:
        command.extend(["--max-model-len", str(max_model_len)])
    if served_model_name:
        command.extend(["--served-model-name", served_model_name])
    return command


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _fetch_json(url: str, timeout_sec: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got: {type(data)!r}")
    return data


def get_server_model_ids(base_url: str, timeout_sec: float = 5.0) -> list[str]:
    models_url = base_url.rstrip("/") + "/models"
    payload = _fetch_json(models_url, timeout_sec)
    models = payload.get("data", [])
    if not isinstance(models, list):
        return []
    model_ids: list[str] = []
    for item in models:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            model_ids.append(item["id"])
    return model_ids


def probe_vllm_server(
    base_url: str = DEFAULT_VLLM_BASE_URL,
    *,
    expected_model: str | None = None,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    try:
        model_ids = get_server_model_ids(base_url, timeout_sec=timeout_sec)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "base_url": base_url,
            "error": str(exc),
            "model_ids": [],
        }
    return {
        "ready": expected_model in model_ids if expected_model else True,
        "base_url": base_url,
        "expected_model": expected_model,
        "model_ids": model_ids,
    }


def wait_for_vllm_server(
    base_url: str = DEFAULT_VLLM_BASE_URL,
    *,
    expected_model: str | None = None,
    timeout_sec: float = 600.0,
    poll_interval_sec: float = 5.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_probe: dict[str, Any] | None = None
    while time.time() < deadline:
        last_probe = probe_vllm_server(base_url, expected_model=expected_model)
        if last_probe["ready"]:
            return last_probe
        time.sleep(poll_interval_sec)
    if last_probe is None:
        last_probe = probe_vllm_server(base_url, expected_model=expected_model)
    raise TimeoutError(
        f"Timed out waiting for vLLM server at {base_url}. Last probe: {json.dumps(last_probe, ensure_ascii=False)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and probe vLLM serve commands for Nemotron Nano VL.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    common.add_argument("--host", default="127.0.0.1")
    common.add_argument("--port", type=int, default=8000)
    common.add_argument("--quantization", default="modelopt")
    common.add_argument("--video-pruning-rate", type=float, default=0.0)
    common.add_argument("--allowed-local-media-path", type=Path, default=None)
    common.add_argument("--limit-mm-per-prompt-image", type=int, default=128)
    common.add_argument("--max-model-len", type=int, default=None)
    common.add_argument("--served-model-name", type=str, default=None)

    subparsers.add_parser("print-command", parents=[common], help="Print a ready-to-run vLLM serve command.")

    serve_parser = subparsers.add_parser("serve", parents=[common], help="Exec into the generated vLLM serve command.")
    serve_parser.add_argument("--dry-run", action="store_true", help="Print the command instead of executing it.")

    probe_parser = subparsers.add_parser("probe", help="Check whether a vLLM server is reachable.")
    probe_parser.add_argument("--base-url", default=DEFAULT_VLLM_BASE_URL)
    probe_parser.add_argument("--expected-model", default=DEFAULT_VLLM_MODEL)
    probe_parser.add_argument("--timeout-sec", type=float, default=5.0)

    wait_parser = subparsers.add_parser("wait", help="Poll until a vLLM server is ready.")
    wait_parser.add_argument("--base-url", default=DEFAULT_VLLM_BASE_URL)
    wait_parser.add_argument("--expected-model", default=DEFAULT_VLLM_MODEL)
    wait_parser.add_argument("--timeout-sec", type=float, default=600.0)
    wait_parser.add_argument("--poll-interval-sec", type=float, default=5.0)

    return parser.parse_args()


def _command_from_args(args: argparse.Namespace) -> list[str]:
    return build_vllm_serve_command(
        model_name=args.model,
        host=args.host,
        port=args.port,
        quantization=args.quantization,
        video_pruning_rate=args.video_pruning_rate,
        allowed_local_media_path=args.allowed_local_media_path,
        limit_mm_per_prompt_image=args.limit_mm_per_prompt_image,
        max_model_len=args.max_model_len,
        served_model_name=args.served_model_name,
    )


def main() -> None:
    args = parse_args()
    if args.command == "print-command":
        print(format_command(_command_from_args(args)))
        return
    if args.command == "serve":
        command = _command_from_args(args)
        if args.dry_run:
            print(format_command(command))
            return
        os.execvp(command[0], command)
    if args.command == "probe":
        result = probe_vllm_server(
            args.base_url,
            expected_model=args.expected_model,
            timeout_sec=args.timeout_sec,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["ready"]:
            sys.exit(1)
        return
    if args.command == "wait":
        result = wait_for_vllm_server(
            args.base_url,
            expected_model=args.expected_model,
            timeout_sec=args.timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
