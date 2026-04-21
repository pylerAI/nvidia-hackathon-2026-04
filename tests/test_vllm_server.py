from __future__ import annotations

from pathlib import Path

from nvidia_hackathon.vllm_server import build_vllm_serve_command, format_command


def test_build_vllm_serve_command_includes_expected_flags() -> None:
    command = build_vllm_serve_command(
        model_name="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",
        host="0.0.0.0",
        port=5566,
        allowed_local_media_path=Path("/tmp/frames"),
        limit_mm_per_prompt_image=64,
        max_model_len=32768,
        served_model_name="nemotron-fp8",
    )

    rendered = format_command(command)

    assert "vllm serve nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8" in rendered
    assert "--quantization modelopt" in rendered
    assert "--trust-remote-code" in rendered
    assert "--allowed-local-media-path /tmp/frames" in rendered
    assert "--limit-mm-per-prompt.image 64" in rendered
    assert "--max-model-len 32768" in rendered
    assert "--served-model-name nemotron-fp8" in rendered
