## SafeWatch Minimal README

이 디렉토리에서는 아래 4개만 주로 사용합니다.

1. `data_curation/build_inserted_eval_dataset.py`
2. `run_sequential_inference.py`
3. `evaluate.py`
4. `visualize/app.py`

전체 흐름은 `데이터 생성 -> 모델별 추론 -> 정량 평가 -> 시각화`입니다.

## 환경 세팅

Python 3.12 기준입니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install opencv-python vllm
```

추가로 시스템에 `ffmpeg`가 필요합니다.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## 1. Data curation

`data_curation/build_inserted_eval_dataset.py`는 안전한 비디오에 harmful clip을 삽입해 새로운 eval 비디오와 JSONL을 만듭니다.  
출력 JSONL에는 비디오 경로, 프롬프트, 정답 interval/guardrail 정보가 함께 저장됩니다.

예시:

```bash
python data_curation/build_inserted_eval_dataset.py \
  --safe-root /gpfs/public/datasets/Video-MME/processed_data/short/video \
  --harmful-jsonl /gpfs/public/datasets/SafeWatch-Bench/guardrail_safewatch_sft_shot_under60s.jsonl \
  --output-root /gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos \
  --output-jsonl /gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos_eval.jsonl \
  --summary-json /gpfs/public/artifacts/SafeWatch-Bench-200K/inserted_videos_eval.summary.json
```

자주 쓰는 옵션:

- `--max-output-duration-sec`: 최종 비디오 최대 길이
- `--max-harmful-duration-sec`: 삽입할 harmful clip 최대 길이
- `--max-samples-per-subcategory`: 서브카테고리별 샘플 수 제한
- `--dry-run`: 실제 파일 생성 없이 샘플링만 확인

## 2. Sequential inference

`run_sequential_inference.py`는 모델을 하나씩 vLLM 서버로 띄운 뒤, 각 모델에 대해 여러 데이터셋에 `generate.py`를 순차 실행합니다.  
기본적으로 baseline 모델과 export된 finetuned 모델들을 자동 탐색하며, 결과는 모델별 폴더에 저장됩니다.

예시:

```bash
python run_sequential_inference.py \
  --output-root /gpfs/public/artifacts/SafeWatch-Bench-200K/results/sequential_vllm_inference \
  --host 127.0.0.1 \
  --port 8000 \
  --data-parallel-size 8 \
  --skip-existing
```

특정 모델만 돌리고 싶으면:

```bash
python run_sequential_inference.py \
  --model-path /path/to/model_or_export_root \
  --output-root /path/to/results
```

주요 출력:

- `<output-root>/<model>/<dataset>/record.jsonl`
- `<output-root>/<model>/<dataset>/raw_outputs.jsonl`
- `<output-root>/manifest.json`

## 3. Evaluation

`evaluate.py`는 추론 결과 JSONL을 읽어서 localization(tIoU)와 guardrail F1을 집계합니다.  
`run_sequential_inference.py`의 출력 루트를 그대로 넣으면 내부의 JSONL을 재귀적으로 읽어 모델별 결과를 비교합니다.

예시:

```bash
python evaluate.py \
  --dirs /gpfs/public/artifacts/SafeWatch-Bench-200K/results/sequential_vllm_inference \
  --tiou-thresholds 0.3 0.5 0.7 \
  --unmatched-policy strict \
  --csv artifacts/eval_summary.csv \
  --json-out artifacts/eval_summary.json \
  --plot-dir artifacts/eval_plots
```

자주 보는 지표:

- `mean_micro_f1`: 매칭된 interval 기준 guardrail 성능
- `mean_tiou_matched`: 매칭된 구간의 평균 tIoU
- `mean_gt_coverage`: GT 구간이 얼마나 커버됐는지
- `frac_records_with_tiou_match`: 최소 1개 이상 매칭된 샘플 비율

## 4. Visualization

`visualize/app.py`는 notrain 결과와 trained 결과를 같은 비디오에서 비교하는 Streamlit 앱입니다.  
GT interval, 두 모델의 harmful interval, label mismatch, IoU 차이를 한 화면에서 확인할 수 있습니다.

실행:

```bash
streamlit run visualize/app.py
```

주의:

- 앱은 현재 코드 내부 상수 경로를 직접 사용합니다.
- 실행 전 `visualize/app.py`의 `DEFAULT_NOTRAIN_PATH`, `DEFAULT_TRAINED_PATH`, `DEFAULT_GT_PATH`를 실제 파일 경로로 맞춰야 합니다.

## 빠른 실행 순서

```bash
# 1) eval 데이터 만들기
python data_curation/build_inserted_eval_dataset.py ...

# 2) 모델별 추론
python run_sequential_inference.py ...

# 3) 결과 집계
python evaluate.py --dirs /path/to/sequential_vllm_inference ...

# 4) 시각화
streamlit run visualize/app.py
```
