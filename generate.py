from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from nvidia_hackathon.prompt_template import build_interval_sft_prompt
from nvidia_hackathon.safewatch_eval_metrics import ShotInterval, parse_shot_intervals
from nvidia_hackathon.vllm_inference import DEFAULT_VLLM_BASE_URL, VLLMFrameClient, ensure_sample_frames
from nvidia_hackathon.vllm_server import DEFAULT_VLLM_MODEL, get_server_model_ids, probe_vllm_server, wait_for_vllm_server

DEFAULT_SOURCE_JSONL = Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos_eval.jsonl")
DEFAULT_FRAMES_ROOT = Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/eval_frames_1fps")
DEFAULT_RAW_OUTPUT_JSONL = Path(
    "/gpfs/public/artifacts/SafeWatch-Bench-200K/results/inserted_videos/model_outputs/baseline_raw_outputs.jsonl"
)
DEFAULT_RECORD_OUTPUT_JSONL = Path(
    "/gpfs/public/artifacts/SafeWatch-Bench-200K/results/inserted_videos/model_outputs/baseline_record.jsonl"
)


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        del format, args


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


class VLLMVideoClient:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        timeout_sec: float = 300.0,
        max_completion_tokens: int = 1024,
        temperature: float = 0.0,
        include_no_think_system_prompt: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.include_no_think_system_prompt = include_no_think_system_prompt

    @property
    def completions_url(self) -> str:
        return self.base_url + "/chat/completions"

    def build_request_payload(self, video_url: str, prompt_text: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.include_no_think_system_prompt:
            messages.append({"role": "system", "content": "/no_think"})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                        "uuid": video_url,
                    },
                ],
            }
        )
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }

    def infer(self, video_url: str, prompt_text: str) -> str:
        payload = self.build_request_payload(video_url, prompt_text)
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


def infer_with_parse_retry(
    *,
    infer_once,
    retry_infer_once,
    enable_retry: bool,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], str | None, bool]:
    raw_output = infer_once()
    interval_payload, interval_dicts, parse_error = parse_intervals_payload(raw_output)
    if parse_error is None or not enable_retry or retry_infer_once is None:
        return raw_output, interval_payload, interval_dicts, parse_error, False

    retried_raw_output = retry_infer_once()
    retry_interval_payload, retry_interval_dicts, retry_parse_error = parse_intervals_payload(retried_raw_output)
    return retried_raw_output, retry_interval_payload, retry_interval_dicts, retry_parse_error, True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object per line in {path}, got: {type(payload)!r}")
                records.append(payload)
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _message_content_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return []
    user_turn = messages[0]
    if not isinstance(user_turn, dict) or user_turn.get("role") != "user":
        return []
    content = user_turn.get("content")
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict)]


def extract_user_prompt(record: dict[str, Any]) -> str:
    for item in _message_content_items(record):
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            return item["text"]
    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        first_turn = conversations[0]
        if isinstance(first_turn, dict) and isinstance(first_turn.get("value"), str):
            value = first_turn["value"]
            if value.startswith("<video>\n"):
                return value[len("<video>\n") :]
            return value.replace("<video>", "", 1).lstrip()
    return build_interval_sft_prompt()


def extract_video_path(record: dict[str, Any]) -> Path:
    raw_video = record.get("video")
    if isinstance(raw_video, str) and raw_video.strip():
        return Path(raw_video)
    for item in _message_content_items(record):
        if item.get("type") == "video" and isinstance(item.get("video"), str):
            return Path(item["video"])
    raise ValueError(f"Record {record.get('id')} does not contain a usable video path.")


def extract_ground_truth_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            if turn.get("from") != "gpt":
                continue
            value = turn.get("value")
            if isinstance(value, str):
                return value

    messages = record.get("messages")
    if isinstance(messages, list):
        for turn in messages:
            if not isinstance(turn, dict):
                continue
            if turn.get("role") != "assistant":
                continue
            content = turn.get("content")
            if isinstance(content, str):
                return content

    raise ValueError(f"Record {record.get('id')} does not contain an assistant response for ground truth.")


def shots_to_chunk_payload(shots: list[ShotInterval]) -> dict[str, Any]:
    chunks: dict[str, dict[str, Any]] = {}
    chunk_order: list[str] = []
    for index, shot in enumerate(shots, start=1):
        chunk_key = f"chunk_{index}"
        chunk_order.append(chunk_key)
        chunks[chunk_key] = {
            "start": shot.start_sec,
            "end": shot.end_sec,
            "description": shot.description,
            "guardrail": shot.guardrail,
            "explanation": shot.explanation,
        }
    return {
        "parse_status": "ok",
        "parse_ok": True,
        "chunks": chunks,
        "chunk_order": chunk_order,
    }


def parse_intervals_payload(text: str) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    try:
        shots = parse_shot_intervals(text)
    except ValueError as exc:
        parse_error = str(exc)
        return (
            {
                "parse_status": "error",
                "parse_ok": False,
                "chunks": {},
                "chunk_order": [],
                "parse_note": parse_error,
            },
            [],
            parse_error,
        )

    interval_dicts = [
        {
            "START_SEC": shot.start_sec,
            "END_SEC": shot.end_sec,
            "DESCRIPTION": shot.description,
            "GUARDRAIL": shot.guardrail,
            "EXPLANATION": shot.explanation,
        }
        for shot in shots
    ]
    return shots_to_chunk_payload(shots), interval_dicts, None


def build_raw_output_record(
    *,
    index: int,
    source_jsonl: Path,
    source_record: dict[str, Any],
    frames_dir: Path | None,
    num_frames: int | None,
    prompt_text: str,
    raw_output: str,
    prediction_interval_payload: dict[str, Any],
    prediction_intervals: list[dict[str, Any]],
    parse_error: str | None,
    request_video_url: str | None,
    used_retry: bool,
) -> dict[str, Any]:
    return {
        "jsonl_record_index": index,
        "source_jsonl": str(source_jsonl),
        "id": source_record.get("id"),
        "video": source_record.get("video"),
        "frames_dir": str(frames_dir) if frames_dir is not None else None,
        "num_frames": num_frames,
        "request_video_url": request_video_url,
        "prompt": prompt_text,
        "raw_output": raw_output,
        "prediction_intervals": prediction_intervals,
        "prediction_interval_payload": prediction_interval_payload,
        "parse_error": parse_error,
        "used_retry": used_retry,
    }


def build_baseline_record(
    *,
    index: int,
    source_jsonl: Path,
    source_record: dict[str, Any],
    raw_output: str,
    prediction_interval_payload: dict[str, Any],
    gt_text: str,
    gt_interval_payload: dict[str, Any],
    used_retry: bool,
) -> dict[str, Any]:
    return {
        "iteration": 0,
        "consumed_train_samples": 0,
        "jsonl_record_index": index,
        "gt_jsonl": str(source_jsonl),
        "gt_record": source_record,
        "prediction": {
            "supervised_span_text": raw_output,
            "note": (
                "Generated by vLLM chat completion using only the user turn. "
                "Assistant messages were excluded."
                + (" Retried once with a larger max_completion_tokens budget." if used_retry else "")
            ),
            "intervals": prediction_interval_payload,
        },
        "ground_truth": {
            "supervised_span_text": gt_text,
            "intervals": gt_interval_payload,
        },
    }


def resolve_model_name(base_url: str, requested_model: str | None) -> str:
    if requested_model:
        return requested_model
    model_ids = get_server_model_ids(base_url, timeout_sec=5.0)
    if model_ids:
        return model_ids[0]
    return DEFAULT_VLLM_MODEL


@contextlib.contextmanager
def local_media_http_server(root_dir: Path, host: str, port: int) -> Any:
    root_dir = root_dir.resolve()

    def handler(*args: Any, **kwargs: Any) -> QuietHTTPRequestHandler:
        return QuietHTTPRequestHandler(*args, directory=str(root_dir), **kwargs)

    httpd = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


def build_video_http_url(video_path: Path, *, serve_root: Path, host: str, port: int) -> str:
    video_path = video_path.resolve()
    serve_root = serve_root.resolve()
    relative_path = video_path.relative_to(serve_root)
    quoted_path = "/".join(urllib.parse.quote(part) for part in relative_path.parts)
    return f"http://{host}:{port}/{quoted_path}"


def infer_media_http_root(records: list[dict[str, Any]]) -> Path:
    video_paths: list[Path] = []
    for record in records:
        try:
            video_paths.append(extract_video_path(record).resolve())
        except Exception:
            continue

    if not video_paths:
        raise ValueError("Could not infer media HTTP root because no usable video paths were found in source-jsonl.")

    common_path = Path(os.path.commonpath([str(path) for path in video_paths]))
    return common_path


def process_source_record(
    *,
    index: int,
    total_records: int,
    source_jsonl: Path,
    source_record: dict[str, Any],
    input_mode: str,
    frames_root: Path,
    extraction_fps: float,
    max_frames: int,
    force_reextract_frames: bool,
    media_http_root: Path,
    media_http_host: str,
    media_http_port: int | None,
    enable_retry: bool,
    frame_client: VLLMFrameClient | None,
    retry_frame_client: VLLMFrameClient | None,
    video_client: VLLMVideoClient | None,
    retry_video_client: VLLMVideoClient | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = str(source_record.get("id", f"record_{index}"))
    video_path = extract_video_path(source_record)
    prompt_text = extract_user_prompt(source_record)
    gt_text = extract_ground_truth_text(source_record)
    gt_interval_payload, _, _ = parse_intervals_payload(gt_text)

    print(f"[submit {index + 1}/{total_records}] {sample_id}")
    frames_dir: Path | None = None
    num_frames: int | None = None
    request_video_url: str | None = None

    if input_mode == "frames":
        assert frame_client is not None
        frames_dir, frame_paths = ensure_sample_frames(
            sample_id=sample_id,
            video_path=video_path,
            frames_root=frames_root,
            fps=extraction_fps,
            max_frames=max_frames,
            overwrite=force_reextract_frames,
        )
        num_frames = len(frame_paths)
        raw_output, prediction_interval_payload, prediction_intervals, parse_error, used_retry = infer_with_parse_retry(
            infer_once=lambda: frame_client.infer(frame_paths, prompt_text),
            retry_infer_once=(
                (lambda: retry_frame_client.infer(frame_paths, prompt_text))
                if retry_frame_client is not None
                else None
            ),
            enable_retry=enable_retry,
        )
    else:
        assert video_client is not None
        if media_http_port is None:
            raise RuntimeError("media_http_port is required for video_url input mode.")
        request_video_url = build_video_http_url(
            video_path,
            serve_root=media_http_root,
            host=media_http_host,
            port=media_http_port,
        )
        raw_output, prediction_interval_payload, prediction_intervals, parse_error, used_retry = infer_with_parse_retry(
            infer_once=lambda: video_client.infer(request_video_url, prompt_text),
            retry_infer_once=(
                (lambda: retry_video_client.infer(request_video_url, prompt_text))
                if retry_video_client is not None
                else None
            ),
            enable_retry=enable_retry,
        )

    raw_output_record = build_raw_output_record(
        index=index,
        source_jsonl=source_jsonl,
        source_record=source_record,
        frames_dir=frames_dir,
        num_frames=num_frames,
        prompt_text=prompt_text,
        raw_output=raw_output,
        prediction_interval_payload=prediction_interval_payload,
        prediction_intervals=prediction_intervals,
        parse_error=parse_error,
        request_video_url=request_video_url,
        used_retry=used_retry,
    )
    baseline_record = build_baseline_record(
        index=index,
        source_jsonl=source_jsonl,
        source_record=source_record,
        raw_output=raw_output,
        prediction_interval_payload=prediction_interval_payload,
        gt_text=gt_text,
        gt_interval_payload=gt_interval_payload,
        used_retry=used_retry,
    )
    return raw_output_record, baseline_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batch inference against inserted_videos_eval.jsonl with a vLLM server, "
            "excluding assistant turns from the request, and emit baseline_record.jsonl-compatible output."
        )
    )
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--raw-output-jsonl", type=Path, default=DEFAULT_RAW_OUTPUT_JSONL)
    parser.add_argument("--record-output-jsonl", type=Path, default=DEFAULT_RECORD_OUTPUT_JSONL)
    parser.add_argument("--base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--input-mode", choices=("video_url", "frames"), default="video_url")
    parser.add_argument("--extraction-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument(
        "--retry-max-completion-tokens",
        type=int,
        default=4096,
        help="If parsing fails, retry once with this larger completion budget.",
    )
    parser.add_argument("--disable-parse-retry", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--media-url-mode",
        choices=("file_url", "data_url"),
        default="data_url",
        help="Use data URLs by default so the server does not need local-media-path access.",
    )
    parser.add_argument(
        "--media-http-root",
        type=Path,
        default=None,
        help="Root directory to expose over HTTP when --input-mode=video_url. If omitted, it is inferred from video paths.",
    )
    parser.add_argument("--media-http-host", default="127.0.0.1")
    parser.add_argument("--media-http-port", type=int, default=8765)
    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        default=16,
        help="Maximum number of records to process concurrently.",
    )
    parser.add_argument("--force-reextract-frames", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--wait-for-server", action="store_true")
    parser.add_argument("--server-timeout-sec", type=float, default=600.0)
    parser.add_argument("--request-timeout-sec", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.wait_for_server:
        wait_for_vllm_server(args.base_url, expected_model=args.model, timeout_sec=args.server_timeout_sec)
    else:
        probe = probe_vllm_server(
            args.base_url,
            expected_model=args.model,
            timeout_sec=min(args.server_timeout_sec, 5.0),
        )
        if not probe["ready"]:
            raise RuntimeError(
                "vLLM server is not ready. "
                f"Probe result: {json.dumps(probe, ensure_ascii=False)}. "
                "Rerun with `--wait-for-server` if the server is still starting."
            )

    model_name = resolve_model_name(args.base_url, args.model)
    source_records = load_jsonl(args.source_jsonl)
    if args.max_samples is not None:
        source_records = source_records[: args.max_samples]
    effective_media_http_root = (
        infer_media_http_root(source_records) if args.input_mode == "video_url" and args.media_http_root is None else args.media_http_root
    )

    if args.max_inflight_requests <= 0:
        raise ValueError(f"--max-inflight-requests must be positive, got {args.max_inflight_requests}")

    enable_retry = (
        not args.disable_parse_retry and args.retry_max_completion_tokens > args.max_completion_tokens
    )

    if args.input_mode == "frames":
        frame_client = VLLMFrameClient(
            base_url=args.base_url,
            model_name=model_name,
            api_key=args.api_key,
            timeout_sec=args.request_timeout_sec,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            media_url_mode=args.media_url_mode,
        )
        retry_frame_client = (
            VLLMFrameClient(
                base_url=args.base_url,
                model_name=model_name,
                api_key=args.api_key,
                timeout_sec=args.request_timeout_sec,
                max_completion_tokens=args.retry_max_completion_tokens,
                temperature=args.temperature,
                media_url_mode=args.media_url_mode,
            )
            if enable_retry
            else None
        )
        media_server_context = contextlib.nullcontext(None)
    else:
        frame_client = None
        retry_frame_client = None
        media_server_context = local_media_http_server(
            effective_media_http_root,
            args.media_http_host,
            args.media_http_port,
        )
        video_client = VLLMVideoClient(
            base_url=args.base_url,
            model_name=model_name,
            api_key=args.api_key,
            timeout_sec=args.request_timeout_sec,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
        )
        retry_video_client = (
            VLLMVideoClient(
                base_url=args.base_url,
                model_name=model_name,
                api_key=args.api_key,
                timeout_sec=args.request_timeout_sec,
                max_completion_tokens=args.retry_max_completion_tokens,
                temperature=args.temperature,
            )
            if enable_retry
            else None
        )

    with media_server_context as media_http_port:
        raw_output_records_by_index: list[dict[str, Any] | None] = [None] * len(source_records)
        baseline_records_by_index: list[dict[str, Any] | None] = [None] * len(source_records)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_inflight_requests) as executor:
            future_to_index = {
                executor.submit(
                    process_source_record,
                    index=index,
                    total_records=len(source_records),
                    source_jsonl=args.source_jsonl,
                    source_record=source_record,
                    input_mode=args.input_mode,
                    frames_root=args.frames_root,
                    extraction_fps=args.extraction_fps,
                    max_frames=args.max_frames,
                    force_reextract_frames=args.force_reextract_frames,
                    media_http_root=effective_media_http_root,
                    media_http_host=args.media_http_host,
                    media_http_port=media_http_port,
                    enable_retry=enable_retry,
                    frame_client=frame_client,
                    retry_frame_client=retry_frame_client,
                    video_client=video_client if args.input_mode == "video_url" else None,
                    retry_video_client=retry_video_client if args.input_mode == "video_url" else None,
                ): index
                for index, source_record in enumerate(source_records)
            }

            for completed_count, future in enumerate(concurrent.futures.as_completed(future_to_index), start=1):
                index = future_to_index[future]
                raw_output_record, baseline_record = future.result()
                raw_output_records_by_index[index] = raw_output_record
                baseline_records_by_index[index] = baseline_record
                print(f"[done {completed_count}/{len(source_records)}] {raw_output_record['id']}")

        raw_output_records = [record for record in raw_output_records_by_index if record is not None]
        baseline_records = [record for record in baseline_records_by_index if record is not None]

    write_jsonl(raw_output_records, args.raw_output_jsonl)
    write_jsonl(baseline_records, args.record_output_jsonl)

    summary = {
        "num_records": len(source_records),
        "source_jsonl": str(args.source_jsonl),
        "raw_output_jsonl": str(args.raw_output_jsonl),
        "record_output_jsonl": str(args.record_output_jsonl),
        "frames_root": str(args.frames_root),
        "base_url": args.base_url,
        "model": model_name,
        "input_mode": args.input_mode,
        "media_http_root": str(effective_media_http_root) if effective_media_http_root is not None else None,
        "media_url_mode": args.media_url_mode,
        "max_inflight_requests": args.max_inflight_requests,
        "retry_enabled": enable_retry,
        "retry_max_completion_tokens": args.retry_max_completion_tokens,
        "num_retried_records": sum(bool(record["used_retry"]) for record in raw_output_records),
        "num_parse_failures": sum(record["parse_error"] is not None for record in raw_output_records),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
