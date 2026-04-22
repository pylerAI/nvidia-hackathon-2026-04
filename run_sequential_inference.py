from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from nvidia_hackathon.vllm_server import wait_for_vllm_server

BASELINE_MODEL = Path("/gpfs/public/artifacts/models/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
FINETUNE_EXPORT_ROOT = Path(
    "/gpfs/public/artifacts/iji/nemotron-hackathon/"
    "nemotron_nano_v2_vl_safewatch_datav2_lr1e-5_nomax_1000/hf_exports"
)
DEFAULT_OUTPUT_ROOT = Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/results/sequential_vllm_inference")
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
GENERATE_SCRIPT = Path(__file__).resolve().parent / "generate.py"


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    source_jsonl: Path


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_path: Path


DATASETS = (
    DatasetSpec("test_set", Path("/gpfs/public/datasets/SafeWatch-Bench/guardrail_safewatch_sft_shot_under60s.jsonl")),
    DatasetSpec("eval_set", Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/sft_jsonl/sft_eval_v2.jsonl")),
    DatasetSpec("synthetic_eval_set", Path("/gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos_eval.jsonl")),
)


def extract_iter_num(path: Path) -> int:
    suffix = path.name.rsplit("iter", 1)[-1]
    return int(suffix)


def is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and (
        (path / "model.safetensors.index.json").exists() or any(path.glob("model-*.safetensors"))
    )


def discover_default_model_specs() -> list[ModelSpec]:
    model_specs = [ModelSpec(slug="baseline", model_path=BASELINE_MODEL)]
    iter_paths = sorted(
        FINETUNE_EXPORT_ROOT.glob("datav2_lr1e5_nomax_1000_iter*"),
        key=extract_iter_num,
        reverse=True,
    )
    model_specs.extend(ModelSpec(slug=path.name, model_path=path) for path in iter_paths)
    return model_specs


def discover_model_specs_from_inputs(model_inputs: list[Path]) -> list[ModelSpec]:
    discovered_paths: list[Path] = []
    for model_input in model_inputs:
        resolved = model_input.resolve()
        if is_model_dir(resolved):
            discovered_paths.append(resolved)
            continue
        if resolved.is_dir():
            child_model_dirs = sorted(path for path in resolved.iterdir() if is_model_dir(path))
            if child_model_dirs:
                discovered_paths.extend(child_model_dirs)
                continue
        raise ValueError(
            f"Model input {model_input} is neither a model directory nor a directory containing model exports."
        )

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in discovered_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return [ModelSpec(slug=path.name, model_path=path) for path in unique_paths]


def tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


@contextmanager
def started_vllm_server(
    *,
    model_spec: ModelSpec,
    base_url: str,
    host: str,
    port: int,
    data_parallel_size: int,
    server_timeout_sec: float,
    log_path: Path,
) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "vllm",
        "serve",
        str(model_spec.model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--video-pruning-rate",
        "0",
        "-dp",
        str(data_parallel_size),
        "--served-model-name",
        model_spec.slug,
    ]

    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_vllm_server(base_url, expected_model=model_spec.slug, timeout_sec=server_timeout_sec)
            yield
        except Exception as exc:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server for {model_spec.slug} exited early with code {process.returncode}.\n"
                    f"Log tail:\n{tail_text(log_path)}"
                ) from exc
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def run_generate(
    *,
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
    output_root: Path,
    base_url: str,
    max_inflight_requests: int,
    request_timeout_sec: float,
) -> dict[str, str]:
    run_dir = output_root / model_spec.slug / dataset_spec.slug
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = run_dir / "raw_outputs.jsonl"
    record_output_path = run_dir / "record.jsonl"
    command = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--source-jsonl",
        str(dataset_spec.source_jsonl),
        "--base-url",
        base_url,
        "--model",
        model_spec.slug,
        "--max-inflight-requests",
        str(max_inflight_requests),
        "--request-timeout-sec",
        str(request_timeout_sec),
        "--raw-output-jsonl",
        str(raw_output_path),
        "--record-output-jsonl",
        str(record_output_path),
    ]
    subprocess.run(command, check=True)
    return {
        "dataset": dataset_spec.slug,
        "source_jsonl": str(dataset_spec.source_jsonl),
        "raw_output_jsonl": str(raw_output_path),
        "record_output_jsonl": str(record_output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially serve the baseline model and each exported fine-tuned model, "
            "then run generate.py over the three SafeWatch datasets."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-parallel-size", type=int, default=8)
    parser.add_argument("--max-inflight-requests", type=int, default=16)
    parser.add_argument("--server-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--request-timeout-sec", type=float, default=300.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--model-path",
        type=Path,
        action="append",
        default=[],
        help=(
            "Specific model directory or export root to run. "
            "May be passed multiple times. If omitted, the default baseline + iter models are used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{args.host}:{args.port}/v1"

    manifest: dict[str, object] = {
        "started_at_unix": time.time(),
        "base_url": base_url,
        "data_parallel_size": args.data_parallel_size,
        "max_inflight_requests": args.max_inflight_requests,
        "models": [],
    }

    model_specs = (
        discover_model_specs_from_inputs(args.model_path)
        if args.model_path
        else discover_default_model_specs()
    )

    for model_spec in model_specs:
        print(f"\n=== Serving model: {model_spec.slug} ===", flush=True)
        model_output_root = output_root / model_spec.slug
        server_log_path = model_output_root / "server.log"

        with started_vllm_server(
            model_spec=model_spec,
            base_url=base_url,
            host=args.host,
            port=args.port,
            data_parallel_size=args.data_parallel_size,
            server_timeout_sec=args.server_timeout_sec,
            log_path=server_log_path,
        ):
            model_runs: list[dict[str, str]] = []
            for dataset_spec in DATASETS:
                run_dir = model_output_root / dataset_spec.slug
                record_output_path = run_dir / "record.jsonl"
                raw_output_path = run_dir / "raw_outputs.jsonl"
                if args.skip_existing and record_output_path.exists() and raw_output_path.exists():
                    print(f"[skip] {model_spec.slug} :: {dataset_spec.slug}", flush=True)
                    model_runs.append(
                        {
                            "dataset": dataset_spec.slug,
                            "source_jsonl": str(dataset_spec.source_jsonl),
                            "raw_output_jsonl": str(raw_output_path),
                            "record_output_jsonl": str(record_output_path),
                        }
                    )
                    continue

                print(f"[run] {model_spec.slug} :: {dataset_spec.slug}", flush=True)
                model_runs.append(
                    run_generate(
                        model_spec=model_spec,
                        dataset_spec=dataset_spec,
                        output_root=output_root,
                        base_url=base_url,
                        max_inflight_requests=args.max_inflight_requests,
                        request_timeout_sec=args.request_timeout_sec,
                    )
                )

        manifest["models"].append(
            {
                "model_slug": model_spec.slug,
                "model_path": str(model_spec.model_path),
                "server_log": str(server_log_path),
                "runs": model_runs,
            }
        )
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    manifest["finished_at_unix"] = time.time()
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
