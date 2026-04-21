from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import torch

def patch_invalid_entry_points() -> None:
    modules = []
    try:
        import importlib.metadata as stdlib_metadata

        modules.append(stdlib_metadata)
    except ImportError:
        pass

    try:
        import importlib_metadata

        modules.append(importlib_metadata)
    except ImportError:
        pass

    for metadata_module in modules:
        distribution_class = metadata_module.Distribution
        if getattr(distribution_class, "_cursor_safe_entry_points_patch", False):
            continue

        def safe_entry_points(self, _metadata_module=metadata_module):
            text = self.read_text("entry_points.txt")
            if not text:
                return _metadata_module.EntryPoints(())
            try:
                return _metadata_module.EntryPoints._from_text_for(text, self)
            except Exception:
                return _metadata_module.EntryPoints(())

        distribution_class.entry_points = property(safe_entry_points)
        distribution_class._cursor_safe_entry_points_patch = True


def extract_json_dict(text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates: list[str] = []
    if fenced_match:
        candidates.append(fenced_match.group(1))

    brace_spans = list(re.finditer(r"\{", text))
    for start_match in reversed(brace_spans):
        start = start_match.start()
        candidate = text[start:].strip()
        if not candidate.endswith("}"):
            end = text.rfind("}")
            if end > start:
                candidate = text[start : end + 1].strip()
        if candidate:
            candidates.append(candidate)

    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    candidates.append(text.strip())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            try:
                payload = ast.literal_eval(candidate)
                if isinstance(payload, dict):
                    return payload
            except (ValueError, SyntaxError):
                continue
    raise ValueError(f"Could not parse model output as JSON:\n{text}")


def extract_assistant_response_text(text: str) -> str:
    assistant_markers = ["<SPECIAL_11>Assistant", "Assistant\n"]
    extracted = text
    for marker in assistant_markers:
        marker_index = extracted.rfind(marker)
        if marker_index != -1:
            extracted = extracted[marker_index + len(marker) :]
            break
    return extracted.replace("<think></think>", "").strip()


def extract_assistant_json_text(text: str) -> str:
    assistant_text = extract_assistant_response_text(text)
    payload = extract_json_dict(assistant_text)
    return json.dumps(payload, ensure_ascii=True, indent=2)


class NemotronVideoModerator:
    def __init__(
        self,
        model_path: Path,
        device: str,
        video_pruning_rate: float,
        max_new_tokens: int,
    ) -> None:
        patch_invalid_entry_points()
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch_dtype,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), fix_mistral_regex=True)
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.video_pruning_rate = video_pruning_rate
        self.max_new_tokens = max_new_tokens

    def _clean_generated_text(self, rendered_prompt: str, generated_ids: Any, input_length: int) -> str:
        continuation_ids = generated_ids[:, input_length:]
        continuation_text = self.processor.batch_decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if continuation_text:
            return continuation_text

        full_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        if full_text.startswith(rendered_prompt):
            full_text = full_text[len(rendered_prompt) :]

        for token in (
            getattr(self.tokenizer, "bos_token", None),
            getattr(self.tokenizer, "eos_token", None),
            getattr(self.tokenizer, "pad_token", None),
        ):
            if token:
                full_text = full_text.replace(token, "")
        return full_text.strip()

    def infer(self, frames: list[Any], clip_duration_sec: float, prompt_text: str) -> str:
        from transformers.video_utils import VideoMetadata

        effective_fps: float | None = None
        if len(frames) > 1 and clip_duration_sec > 0:
            effective_fps = (len(frames) - 1) / clip_duration_sec

        metadata = VideoMetadata(
            total_num_frames=len(frames),
            fps=effective_fps,
            duration=clip_duration_sec if len(frames) > 1 else None,
            video_backend=None,
        )

        messages = [
            {"role": "system", "content": "/no_think"},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": ""},
                    {"type": "text", "text": "\n" + prompt_text},
                ],
            },
        ]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.processor(
            text=[rendered_prompt],
            videos=frames,
            videos_kwargs={"video_metadata": metadata},
            return_tensors="pt",
        ).to(self.device)

        self.model.video_pruning_rate = self.video_pruning_rate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                pixel_values_videos=model_inputs.pixel_values_videos,
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self._clean_generated_text(rendered_prompt, generated_ids, model_inputs.input_ids.shape[1])
