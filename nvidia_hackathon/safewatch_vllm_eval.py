from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .prompt_template import build_interval_sft_prompt
from .safewatch_eval_metrics import (
    ShotInterval,
    compute_temporal_metrics,
    compute_video_level_metrics,
    parse_shot_intervals,
    video_categories_from_shots,
)
from .utils import save_json
from .vllm_inference import DEFAULT_VLLM_BASE_URL, DEFAULT_VLLM_MODEL, VLLMFrameClient, ensure_sample_frames
from .vllm_server import probe_vllm_server, wait_for_vllm_server

DEFAULT_EVAL_JSONL = Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/sft_jsonl/guardrail_safewatch_sft_shot_under60s.jsonl")
DEFAULT_FRAMES_ROOT = Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/eval_frames_1fps")
DEFAULT_OUTPUT_ROOT = Path("artifacts/safewatch_vllm_eval")


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


def _assistant_text_from_record(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if isinstance(conversations, list) and len(conversations) >= 2:
        assistant_turn = conversations[1]
        if isinstance(assistant_turn, dict):
            value = assistant_turn.get("value")
            if isinstance(value, str):
                return value

    messages = record.get("messages")
    if isinstance(messages, list) and len(messages) >= 2:
        assistant_turn = messages[1]
        if isinstance(assistant_turn, dict):
            content = assistant_turn.get("content")
            if isinstance(content, str):
                return content
    raise ValueError(f"Record {record.get('id')} does not contain an assistant response.")


def _user_prompt_from_record(record: dict[str, Any]) -> str | None:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        user_turn = messages[0]
        if isinstance(user_turn, dict):
            content = user_turn.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                        return item["text"]

    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        user_turn = conversations[0]
        if isinstance(user_turn, dict):
            value = user_turn.get("value")
            if isinstance(value, str):
                if value.startswith("<video>\n"):
                    return value[len("<video>\n") :]
                return value.replace("<video>", "", 1).lstrip()
    return None


def _shot_list_to_dicts(shots: list[ShotInterval]) -> list[dict[str, Any]]:
    return [shot.to_dict() for shot in shots]


def _prediction_record_to_json(
    record: dict[str, Any],
    *,
    frames_dir: Path,
    raw_output: str,
    parse_error: str | None,
    gt_shots: list[ShotInterval],
    pred_shots: list[ShotInterval],
) -> dict[str, Any]:
    return {
        "id": record["id"],
        "video": record["video"],
        "video_relative_path": record.get("metadata", {}).get("video_relative_path"),
        "frames_dir": str(frames_dir),
        "num_frames": len(list(frames_dir.glob("frame_*.jpg"))),
        "raw_output": raw_output,
        "parse_error": parse_error,
        "gt_categories": video_categories_from_shots(gt_shots),
        "pred_categories": video_categories_from_shots(pred_shots),
        "gt_shots": _shot_list_to_dicts(gt_shots),
        "pred_shots": _shot_list_to_dicts(pred_shots),
    }


def evaluate_eval_jsonl(
    *,
    eval_jsonl_path: Path,
    frames_root: Path,
    output_root: Path,
    base_url: str,
    model_name: str,
    api_key: str,
    extraction_fps: float,
    max_frames: int,
    max_completion_tokens: int,
    temperature: float,
    media_url_mode: str,
    force_reextract_frames: bool,
    max_samples: int | None,
    wait_for_server_ready: bool,
    server_timeout_sec: float,
    request_timeout_sec: float,
) -> dict[str, Any]:
    if wait_for_server_ready:
        wait_for_vllm_server(base_url, expected_model=model_name, timeout_sec=server_timeout_sec)
    else:
        probe = probe_vllm_server(base_url, expected_model=model_name, timeout_sec=min(server_timeout_sec, 5.0))
        if not probe["ready"]:
            raise RuntimeError(
                "vLLM server is not ready. "
                f"Probe result: {json.dumps(probe, ensure_ascii=False)}. "
                "Use `python -m nvidia_hackathon.vllm_server print-command` to see the serve command, "
                "or rerun with `--wait-for-server`."
            )

    records = load_jsonl(eval_jsonl_path)
    if max_samples is not None:
        records = records[:max_samples]

    fallback_prompt_text = build_interval_sft_prompt()
    client = VLLMFrameClient(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        timeout_sec=request_timeout_sec,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        media_url_mode=media_url_mode,
    )

    prediction_records: list[dict[str, Any]] = []
    metrics_input: list[dict[str, Any]] = []
    num_parse_failures = 0

    for record in records:
        gt_text = _assistant_text_from_record(record)
        gt_shots = parse_shot_intervals(gt_text)
        prompt_text = _user_prompt_from_record(record) or fallback_prompt_text
        frames_dir, frame_paths = ensure_sample_frames(
            sample_id=str(record["id"]),
            video_path=Path(record["video"]),
            frames_root=frames_root,
            fps=extraction_fps,
            max_frames=max_frames,
            overwrite=force_reextract_frames,
        )

        raw_output = client.infer(frame_paths, prompt_text)
        parse_error: str | None = None
        try:
            pred_shots = parse_shot_intervals(raw_output)
        except ValueError as exc:
            pred_shots = []
            parse_error = str(exc)
            num_parse_failures += 1

        prediction_records.append(
            _prediction_record_to_json(
                record,
                frames_dir=frames_dir,
                raw_output=raw_output,
                parse_error=parse_error,
                gt_shots=gt_shots,
                pred_shots=pred_shots,
            )
        )
        metrics_input.append(
            {
                "id": record["id"],
                "gt_shots": gt_shots,
                "pred_shots": pred_shots,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.jsonl"
    write_jsonl(prediction_records, predictions_path)

    video_metrics = compute_video_level_metrics(metrics_input)
    temporal_metrics = compute_temporal_metrics(metrics_input)
    save_json(output_root / "video_metrics.json", video_metrics)
    save_json(output_root / "temporal_metrics.json", temporal_metrics)

    run_config = {
        "eval_jsonl_path": str(eval_jsonl_path),
        "frames_root": str(frames_root),
        "output_root": str(output_root),
        "base_url": base_url,
        "model_name": model_name,
        "extraction_fps": extraction_fps,
        "max_frames": max_frames,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "media_url_mode": media_url_mode,
        "force_reextract_frames": force_reextract_frames,
        "max_samples": max_samples,
        "num_samples": len(records),
        "num_parse_failures": num_parse_failures,
        "predictions_path": str(predictions_path),
    }
    save_json(output_root / "run_config.json", run_config)
    return {
        "run_config": run_config,
        "video_metrics": video_metrics,
        "temporal_metrics": temporal_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shot-level SafeWatch evaluation through a vLLM server.")
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL_JSONL)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--extraction-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--media-url-mode", choices=("file_url", "data_url"), default="file_url")
    parser.add_argument("--force-reextract-frames", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--wait-for-server", action="store_true")
    parser.add_argument("--server-timeout-sec", type=float, default=600.0)
    parser.add_argument("--request-timeout-sec", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_eval_jsonl(
        eval_jsonl_path=args.eval_jsonl,
        frames_root=args.frames_root,
        output_root=args.output_root,
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.api_key,
        extraction_fps=args.extraction_fps,
        max_frames=args.max_frames,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        media_url_mode=args.media_url_mode,
        force_reextract_frames=args.force_reextract_frames,
        max_samples=args.max_samples,
        wait_for_server_ready=args.wait_for_server,
        server_timeout_sec=args.server_timeout_sec,
        request_timeout_sec=args.request_timeout_sec,
    )
    print(json.dumps(result["run_config"], indent=2, ensure_ascii=False))
    print(json.dumps(result["video_metrics"], indent=2, ensure_ascii=False))
    print(json.dumps(result["temporal_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
