## SafeWatch Minimal README

This directory mainly uses the following four entry points.

1. `data_curation/build_inserted_eval_dataset.py`
2. `run_sequential_inference.py`
3. `evaluate.py`
4. `visualize/app.py`

The overall flow is `data generation -> per-model inference -> quantitative evaluation -> visualization`.

## Environment setup

Targets Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install opencv-python vllm
```

`ffmpeg` is also required on the system.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## 1. Data curation

`data_curation/build_inserted_eval_dataset.py` inserts harmful clips into safe videos to build new eval videos and a JSONL file.  
The output JSONL stores the video path, the prompt, and the ground-truth interval/guardrail information together.

Example:

```bash
python data_curation/build_inserted_eval_dataset.py \
  --safe-root /gpfs/public/datasets/Video-MME/processed_data/short/video \
  --harmful-jsonl /gpfs/public/datasets/SafeWatch-Bench/guardrail_safewatch_sft_shot_under60s.jsonl \
  --output-root /gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos \
  --output-jsonl /gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos_eval.jsonl \
  --summary-json /gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos_eval.summary.json
```

Frequently used options:

- `--max-output-duration-sec`: maximum length of the final video
- `--max-harmful-duration-sec`: maximum length of the harmful clip to insert
- `--max-samples-per-subcategory`: limit on the number of samples per subcategory
- `--dry-run`: only check the sampling, without generating actual files

## 2. Sequential inference

`run_sequential_inference.py` brings up models one at a time on a vLLM server and, for each model, runs `generate.py` sequentially over multiple datasets.  
By default it auto-discovers the baseline model and the exported finetuned models, and results are saved in per-model folders.

Example:

```bash
python run_sequential_inference.py \
  --output-root /gpfs/public/artifacts/SafeWatch-Bench-200K/results/sequential_vllm_inference \
  --host 127.0.0.1 \
  --port 8000 \
  --data-parallel-size 8 \
  --skip-existing
```

To run only a specific model:

```bash
python run_sequential_inference.py \
  --model-path /path/to/model_or_export_root \
  --output-root /path/to/results
```

Main outputs:

- `<output-root>/<model>/<dataset>/record.jsonl`
- `<output-root>/<model>/<dataset>/raw_outputs.jsonl`
- `<output-root>/manifest.json`

## 3. Evaluation

`evaluate.py` reads the inference-result JSONL files and aggregates localization (tIoU) and guardrail F1.  
If you pass the output root of `run_sequential_inference.py` as-is, it recursively reads the JSONL files inside and compares results per model.

Example:

```bash
python evaluate.py \
  --dirs /gpfs/public/artifacts/SafeWatch-Bench-200K/results/sequential_vllm_inference \
  --tiou-thresholds 0.3 0.5 0.7 \
  --unmatched-policy strict \
  --csv artifacts/eval_summary.csv \
  --json-out artifacts/eval_summary.json \
  --plot-dir artifacts/eval_plots
```

Commonly inspected metrics:

- `mean_micro_f1`: guardrail performance over matched intervals
- `mean_tiou_matched`: average tIoU of the matched intervals
- `mean_gt_coverage`: how much of the GT intervals is covered
- `frac_records_with_tiou_match`: fraction of samples with at least one match

## 4. Visualization

`visualize/app.py` is a Streamlit app that compares the notrain results and the trained results on the same video.  
It lets you check the GT interval, both models' harmful intervals, label mismatches, and IoU differences on a single screen.

Run:

```bash
streamlit run visualize/app.py
```

Notes:

- The app currently uses constant paths hard-coded in the source.
- Before running, set `DEFAULT_NOTRAIN_PATH`, `DEFAULT_TRAINED_PATH`, and `DEFAULT_GT_PATH` in `visualize/app.py` to the actual file paths.

## Quick start sequence

```bash
# 1) Build the eval data
python data_curation/build_inserted_eval_dataset.py ...

# 2) Run inference per model
python run_sequential_inference.py ...

# 3) Aggregate the results
python evaluate.py --dirs /path/to/sequential_vllm_inference ...

# 4) Visualize
streamlit run visualize/app.py
```
