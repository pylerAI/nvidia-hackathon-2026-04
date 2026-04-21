## SafeWatch Pipeline

This repo focuses on SafeWatch video moderation with two prompt styles:

- full-video prompt: one `DESCRIPTION`, one `GUARDRAIL`, one `EXPLANATION`
- shot-level prompt: split a video into shots and return one JSON object per shot

The shot-level prompt lives in `nvidia_hackathon/prompt_template.py` as `build_interval_sft_prompt()`.

## Data Curation

### 1. Build shot-level SFT JSONL

This reads SafeWatch annotations and builds shot-level SFT train/eval JSONL files.

```bash
python -m data_curation.build_safewatch_sft_jsonl \
  --dataset-root /gpfs/public/datasets/SafeWatch-Bench-200K
```

Default outputs:

- `artifacts/safewatch_interval_sft/safewatch_interval_sft_60s.jsonl`
- `artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_train.jsonl`
- `artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_eval.jsonl`

### 2. Build 1 FPS eval frames

This converts the shot-level eval JSONL into an image-based eval set by extracting 1 FPS frames for each video.

```bash
python -m data_curation.build_safewatch_eval_frames_jsonl \
  --eval-jsonl artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_eval.jsonl \
  --frames-root /gpfs/public/artifacts/SafeWatch-Bench-200K/eval_frames
```

Default outputs:

- `artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_eval_images_1fps.jsonl`
- `artifacts/safewatch_interval_sft/safewatch_interval_sft_60s_eval_images_1fps.summary.json`

## vLLM Evaluation

### 1. Print or run the Nemotron FP8 serve command

```bash
python -m nvidia_hackathon.vllm_server print-command \
  --allowed-local-media-path /gpfs/public/artifacts/SafeWatch-Bench-200K/eval_frames_1fps \
  --limit-mm-per-prompt-image 128
```

If you want the helper to `exec` the server directly:

```bash
python -m nvidia_hackathon.vllm_server serve \
  --allowed-local-media-path /gpfs/public/artifacts/SafeWatch-Bench-200K/eval_frames_1fps \
  --limit-mm-per-prompt-image 128
```

The default model is `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8`.

### 2. Run shot-level evaluation through vLLM

This runner:

- reads the eval JSONL
- reuses pre-extracted frames if available
- extracts frames with `ffmpeg` if missing
- sends the shot-level prompt to a running vLLM server
- stores raw predictions and normalized shot predictions
- computes video-level category accuracy and temporal tIoU metrics

```bash
python -m nvidia_hackathon.safewatch_vllm_eval \
  --eval-jsonl /gpfs/public/artifacts/SafeWatch-Bench-200K/sft_jsonl/guardrail_safewatch_sft_shot_under60s.jsonl \
  --frames-root /gpfs/public/artifacts/SafeWatch-Bench-200K/eval_frames_1fps \
  --base-url http://127.0.0.1:8000/v1 \
  --model nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8 \
  --wait-for-server
```

Default outputs under `artifacts/safewatch_vllm_eval/`:

- `predictions.jsonl`
- `video_metrics.json`
- `temporal_metrics.json`
- `run_config.json`

## Notes

- `nvidia_hackathon/safewatch_eval_metrics.py` computes video-level labels by OR-ing shot-level category flags.
- Temporal evaluation is category-wise and uses greedy tIoU matching with thresholds `0.5` and `0.7`.
- For large runs, prefer `file_url` media mode and make sure the vLLM server is started with `--allowed-local-media-path` pointing at the same frame cache root.
