from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import streamlit as st


DEFAULT_NOTRAIN_PATH = Path("/gpfs/public/artifacts/iji/nemotron-hackathon/notrain_records.jsonl")
DEFAULT_TRAINED_PATH = Path("/gpfs/public/artifacts/iji/nemotron-hackathon/datav2_lr1e-5_records.jsonl")
DEFAULT_GT_PATH = Path("/gpfs/public/datasets/SafeWatch-Bench/guardrail_safewatch_sft_shot_under60s.jsonl")
CATEGORY_ORDER = [
    "C1(Sexual Content)",
    "C2(Harassment & Bullying)",
    "C3(Threats, Violence & Harm)",
    "C4(False & Deceptive Information)",
    "C5(Illegal/Regulated Activities)",
    "C6(Hateful Content & Extremism)",
]
GENERIC_PARENT_DIRS = {
    "gpfs",
    "public",
    "datasets",
    "artifacts",
    "safewatch-bench",
    "safewatch-bench-200k",
    "full",
    "videos",
    "target",
    "dataset",
    "real",
}
SEXUAL_MARKERS = {"c1", "sexual", "porn", "nsfw", "explicit"}


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def tokenize_lowered(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if token}


def path_variants(value: str | None) -> list[str]:
    if not value or not isinstance(value, str):
        return []

    normalized = value.strip().replace("\\", "/")
    if not normalized:
        return []

    path = Path(normalized)
    variants = [
        normalized,
        path.name,
        path.stem,
    ]

    parts = path.parts
    for anchor in ("full", "videos", "target", "dataset"):
        if anchor in parts:
            idx = parts.index(anchor)
            variants.append("/".join(parts[idx:]))

    return unique_in_order([variant for variant in variants if variant])


def record_candidate_keys(record: dict[str, Any]) -> list[str]:
    gt_record = record.get("gt_record", {}) if isinstance(record.get("gt_record"), dict) else {}
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}

    candidates: list[str] = []
    for raw_value in (
        gt_record.get("id"),
        record.get("id"),
        gt_record.get("video"),
        metadata.get("video_relative_path"),
        record.get("video"),
    ):
        if isinstance(raw_value, str):
            candidates.extend(path_variants(raw_value))
            candidates.append(raw_value)

    return unique_in_order(candidates)


def primary_record_key(record: dict[str, Any]) -> str:
    gt_record = record.get("gt_record", {}) if isinstance(record.get("gt_record"), dict) else {}
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}

    for preferred in (
        gt_record.get("id"),
        record.get("id"),
        metadata.get("video_relative_path"),
        gt_record.get("video"),
        record.get("video"),
    ):
        if isinstance(preferred, str) and preferred.strip():
            variants = path_variants(preferred)
            return preferred if preferred in {gt_record.get("id"), record.get("id")} else variants[0]

    jsonl_index = record.get("jsonl_record_index", "unknown")
    return f"record_{jsonl_index}"


def pick_video_path(*records: dict[str, Any] | None) -> str:
    candidates: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        gt_record = record.get("gt_record", {}) if isinstance(record.get("gt_record"), dict) else {}
        for value in (
            gt_record.get("video"),
            first_present(record, "video"),
        ):
            if isinstance(value, str) and value.strip():
                candidates.append(value)

    for candidate in unique_in_order(candidates):
        if Path(candidate).exists():
            return candidate

    return candidates[0] if candidates else ""


def derive_parent_group(video_path: str, fallback_key: str) -> str:
    if not video_path:
        return fallback_key

    path = Path(video_path)
    meaningful_parts = [
        part
        for part in path.parts[:-1]
        if part.strip("/") and part.lower() not in GENERIC_PARENT_DIRS
    ]
    if not meaningful_parts:
        return fallback_key

    return " / ".join(meaningful_parts[-2:])


def extract_source_category(*records: dict[str, Any] | None) -> str:
    for record in records:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        value = metadata.get("source_category")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def is_sexual_item(video_path: str, parent_group: str, source_category: str) -> bool:
    searchable_values = [video_path, parent_group, source_category]
    for value in searchable_values:
        if not value:
            continue
        tokens = tokenize_lowered(value)
        if tokens & SEXUAL_MARKERS:
            return True
    return False


def category_labels(guardrail: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for category, value in guardrail.items():
        if is_true(value):
            labels.append(category)
    return labels


def normalize_interval(interval: dict[str, Any]) -> dict[str, Any]:
    guardrail = first_present(interval, "guardrail", "GUARDRAIL") or {}
    if not isinstance(guardrail, dict):
        guardrail = {}

    start = safe_float(first_present(interval, "start", "START_SEC"))
    end = safe_float(first_present(interval, "end", "END_SEC"), default=start)
    categories = category_labels(guardrail)

    return {
        "start": start,
        "end": max(start, end),
        "description": str(first_present(interval, "description", "DESCRIPTION") or "None"),
        "explanation": str(first_present(interval, "explanation", "EXPLANATION") or "None"),
        "guardrail": guardrail,
        "categories": categories,
        "harmful": bool(categories),
    }


def intervals_from_chunk_map(chunks: dict[str, Any], chunk_order: list[str] | None) -> list[dict[str, Any]]:
    if not isinstance(chunks, dict):
        return []

    order = chunk_order if isinstance(chunk_order, list) and chunk_order else list(chunks.keys())
    intervals: list[dict[str, Any]] = []

    for chunk_key in order:
        chunk = chunks.get(chunk_key)
        if isinstance(chunk, dict):
            intervals.append(normalize_interval(chunk))

    intervals.sort(key=lambda interval: (interval["start"], interval["end"]))
    return intervals


def parse_json_interval_list(text: str | None) -> list[dict[str, Any]]:
    if not text or not isinstance(text, str):
        return []

    candidate = text.strip()
    if not candidate:
        return []

    if candidate.startswith("```"):
        candidate = candidate.strip("`")

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    intervals = [normalize_interval(item) for item in parsed if isinstance(item, dict)]
    intervals.sort(key=lambda interval: (interval["start"], interval["end"]))
    return intervals


def extract_prediction_intervals(record: dict[str, Any]) -> list[dict[str, Any]]:
    prediction = record.get("prediction", {}) if isinstance(record.get("prediction"), dict) else {}
    interval_payload = prediction.get("intervals", {}) if isinstance(prediction.get("intervals"), dict) else {}

    chunks = interval_payload.get("chunks")
    chunk_order = interval_payload.get("chunk_order")
    intervals = intervals_from_chunk_map(chunks, chunk_order)
    if intervals:
        return intervals

    return parse_json_interval_list(prediction.get("supervised_span_text"))


def extract_embedded_gt_intervals(record: dict[str, Any]) -> list[dict[str, Any]]:
    ground_truth = record.get("ground_truth", {}) if isinstance(record.get("ground_truth"), dict) else {}
    interval_payload = ground_truth.get("intervals", {}) if isinstance(ground_truth.get("intervals"), dict) else {}

    chunks = interval_payload.get("chunks")
    chunk_order = interval_payload.get("chunk_order")
    intervals = intervals_from_chunk_map(chunks, chunk_order)
    if intervals:
        return intervals

    return parse_json_interval_list(ground_truth.get("supervised_span_text"))


def extract_external_gt_intervals(record: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        return []

    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return []

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") != "gpt":
            continue
        intervals = parse_json_interval_list(turn.get("value"))
        if intervals:
            return intervals

    return []


def harmful_intervals(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [interval for interval in intervals if interval["harmful"]]


def merge_time_ranges(intervals: list[dict[str, Any]]) -> list[tuple[float, float]]:
    ranges = sorted(
        [
            (safe_float(interval.get("start")), safe_float(interval.get("end")))
            for interval in intervals
            if safe_float(interval.get("end")) > safe_float(interval.get("start"))
        ]
    )
    if not ranges:
        return []

    merged: list[tuple[float, float]] = []
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def total_range_duration(ranges: list[tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in ranges)


def intersection_duration(
    left_ranges: list[tuple[float, float]],
    right_ranges: list[tuple[float, float]],
) -> float:
    left_idx = 0
    right_idx = 0
    overlap = 0.0

    while left_idx < len(left_ranges) and right_idx < len(right_ranges):
        left_start, left_end = left_ranges[left_idx]
        right_start, right_end = right_ranges[right_idx]

        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        if overlap_end > overlap_start:
            overlap += overlap_end - overlap_start

        if left_end <= right_end:
            left_idx += 1
        else:
            right_idx += 1

    return overlap


def compute_union_iou(pred_intervals: list[dict[str, Any]], gt_intervals: list[dict[str, Any]]) -> float:
    pred_ranges = merge_time_ranges(pred_intervals)
    gt_ranges = merge_time_ranges(gt_intervals)

    pred_duration = total_range_duration(pred_ranges)
    gt_duration = total_range_duration(gt_ranges)
    if pred_duration == 0.0 and gt_duration == 0.0:
        return 1.0

    overlap = intersection_duration(pred_ranges, gt_ranges)
    union = pred_duration + gt_duration - overlap
    if union <= 0.0:
        return 0.0
    return overlap / union


def video_label_map(intervals: list[dict[str, Any]]) -> dict[str, bool]:
    label_map = {category: False for category in CATEGORY_ORDER}
    for interval in intervals:
        for category in interval["categories"]:
            label_map[category] = True
    return label_map


def compare_video_labels(pred_intervals: list[dict[str, Any]], gt_intervals: list[dict[str, Any]]) -> dict[str, Any]:
    pred_labels = video_label_map(pred_intervals)
    gt_labels = video_label_map(gt_intervals)

    per_category = {
        category: pred_labels[category] == gt_labels[category]
        for category in CATEGORY_ORDER
    }
    false_positives = [
        category for category in CATEGORY_ORDER if pred_labels[category] and not gt_labels[category]
    ]
    false_negatives = [
        category for category in CATEGORY_ORDER if gt_labels[category] and not pred_labels[category]
    ]

    return {
        "pred_labels": pred_labels,
        "gt_labels": gt_labels,
        "per_category_correct": per_category,
        "exact_match": all(per_category.values()),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compute_model_metrics(pred_intervals: list[dict[str, Any]], gt_intervals: list[dict[str, Any]]) -> dict[str, Any]:
    label_comparison = compare_video_labels(pred_intervals, gt_intervals)
    return {
        "union_iou": compute_union_iou(pred_intervals, gt_intervals),
        "label_comparison": label_comparison,
    }


def predicted_labels_differ(left_metrics: dict[str, Any], right_metrics: dict[str, Any]) -> bool:
    left_labels = left_metrics["label_comparison"]["pred_labels"]
    right_labels = right_metrics["label_comparison"]["pred_labels"]
    return any(left_labels[category] != right_labels[category] for category in CATEGORY_ORDER)


def aggregate_model_summary(items: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    if not items:
        return {
            "mean_iou": 0.0,
            "category_accuracy": {category: 0.0 for category in CATEGORY_ORDER},
            "exact_match_accuracy": 0.0,
            "count": 0,
        }

    metrics_list = [item[f"{model_key}_metrics"] for item in items]
    mean_iou = sum(metrics["union_iou"] for metrics in metrics_list) / len(metrics_list)
    category_accuracy = {
        category: (
            sum(
                1
                for metrics in metrics_list
                if metrics["label_comparison"]["per_category_correct"][category]
            )
            / len(metrics_list)
        )
        for category in CATEGORY_ORDER
    }
    exact_match_accuracy = (
        sum(1 for metrics in metrics_list if metrics["label_comparison"]["exact_match"]) / len(metrics_list)
    )

    return {
        "mean_iou": mean_iou,
        "category_accuracy": category_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "count": len(metrics_list),
    }


def percent_text(value: float) -> str:
    return f"{value * 100:.1f}%"


@st.cache_data(show_spinner=False)
def load_jsonl(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@st.cache_data(show_spinner=False)
def load_video_bytes(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


@st.cache_data(show_spinner="Loading comparison data...")
def load_comparison_bundle(
    notrain_path: str,
    trained_path: str,
    gt_path: str,
) -> list[dict[str, Any]]:
    notrain_records = load_jsonl(notrain_path)
    trained_records = load_jsonl(trained_path)
    external_gt_records = load_jsonl(gt_path)

    trained_index = {primary_record_key(record): record for record in trained_records}

    external_gt_index: dict[str, dict[str, Any]] = {}
    for gt_record in external_gt_records:
        for key in record_candidate_keys(gt_record):
            external_gt_index.setdefault(key, gt_record)

    bundle: list[dict[str, Any]] = []
    for notrain_record in notrain_records:
        key = primary_record_key(notrain_record)
        trained_record = trained_index.get(key)
        if not trained_record:
            continue

        candidate_keys = unique_in_order(
            [key] + record_candidate_keys(notrain_record) + record_candidate_keys(trained_record)
        )

        external_gt_record = None
        for candidate in candidate_keys:
            if candidate in external_gt_index:
                external_gt_record = external_gt_index[candidate]
                break

        gt_intervals = extract_external_gt_intervals(external_gt_record) if external_gt_record else []
        if not gt_intervals:
            gt_intervals = extract_embedded_gt_intervals(notrain_record)

        video_path = pick_video_path(notrain_record, trained_record, external_gt_record)
        video_name = Path(video_path).name if video_path else key
        parent_group = derive_parent_group(video_path, key)
        source_category = extract_source_category(notrain_record, trained_record)

        notrain_harmful = harmful_intervals(extract_prediction_intervals(notrain_record))
        trained_harmful = harmful_intervals(extract_prediction_intervals(trained_record))
        gt_harmful = harmful_intervals(gt_intervals)
        notrain_metrics = compute_model_metrics(notrain_harmful, gt_harmful)
        trained_metrics = compute_model_metrics(trained_harmful, gt_harmful)
        iou_gap = abs(notrain_metrics["union_iou"] - trained_metrics["union_iou"])
        label_mismatch = predicted_labels_differ(notrain_metrics, trained_metrics)

        bundle.append(
            {
                "key": key,
                "candidate_keys": candidate_keys,
                "video_path": video_path,
                "video_name": video_name,
                "parent_group": parent_group,
                "source_category": source_category,
                "is_sexual": is_sexual_item(video_path, parent_group, source_category),
                "notrain_intervals": notrain_harmful,
                "trained_intervals": trained_harmful,
                "gt_intervals": gt_harmful,
                "notrain_metrics": notrain_metrics,
                "trained_metrics": trained_metrics,
                "model_iou_gap": iou_gap,
                "model_label_mismatch": label_mismatch,
                "gt_source": "external" if external_gt_record and gt_intervals else "embedded",
            }
        )

    bundle.sort(
        key=lambda item: (
            -max(len(item["notrain_intervals"]), len(item["trained_intervals"]), len(item["gt_intervals"])),
            item["video_name"],
        )
    )
    return bundle


def format_interval_title(interval: dict[str, Any]) -> str:
    categories = ", ".join(interval["categories"]) if interval["categories"] else "Unknown category"
    return f'{interval["start"]:.2f}s - {interval["end"]:.2f}s | {categories}'


def format_interval_option(item: dict[str, Any]) -> str:
    return (
        f'{item["parent_group"]} | {item["video_name"]} | '
        f'notrain {len(item["notrain_intervals"])} | '
        f'trained {len(item["trained_intervals"])} | '
        f'GT {len(item["gt_intervals"])} | '
        f'gap {item["model_iou_gap"]:.3f}'
    )


def render_category_chips(categories: list[str]) -> None:
    if not categories:
        st.caption("No harmful category was predicted.")
        return

    chips = "".join(
        f'<span class="category-chip">{html.escape(category)}</span>' for category in categories
    )
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def render_interval_panel(title: str, intervals: list[dict[str, Any]], panel_key: str, video_key: str) -> None:
    st.subheader(title)
    if not intervals:
        st.info("Predicted harmful interval이 없습니다.")
        return

    for idx, interval in enumerate(intervals):
        block_key = f"{video_key}:{panel_key}:{idx}"
        is_open = st.session_state.get("open_interval_key") == block_key

        with st.container(border=True):
            if st.button(
                format_interval_title(interval),
                key=f"button:{block_key}",
                use_container_width=True,
                type="primary" if is_open else "secondary",
            ):
                st.session_state["seek_time"] = interval["start"]
                st.session_state["open_interval_key"] = None if is_open else block_key
                st.rerun()

            render_category_chips(interval["categories"])

            if is_open:
                st.markdown("**Description**")
                st.write(interval["description"])
                st.markdown("**Explanation**")
                st.write(interval["explanation"])


def render_gt_summary(gt_intervals: list[dict[str, Any]]) -> None:
    with st.expander("GT harmful intervals", expanded=False):
        if not gt_intervals:
            st.write("GT에서 harmful interval이 확인되지 않았습니다.")
            return

        for interval in gt_intervals:
            st.markdown(f'**{format_interval_title(interval)}**')
            render_category_chips(interval["categories"])
            st.write(interval["description"])


def render_label_map(label_map: dict[str, bool]) -> None:
    chips = "".join(
        f'<span class="category-chip">{html.escape(category)}</span>'
        for category in CATEGORY_ORDER
        if label_map.get(category)
    )
    if chips:
        st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
    else:
        st.caption("No harmful video-level label.")


def render_model_eval_panel(title: str, metrics: dict[str, Any]) -> None:
    comparison = metrics["label_comparison"]
    exact_match = comparison["exact_match"]

    with st.container(border=True):
        st.markdown(f"**{title}**")
        metric_cols = st.columns(2)
        metric_cols[0].metric("Union IoU", f'{metrics["union_iou"]:.3f}')
        metric_cols[1].metric("Video label", "Correct" if exact_match else "Wrong")

        st.caption("Predicted video-level label (OR over harmful intervals)")
        render_label_map(comparison["pred_labels"])

        if exact_match:
            st.success("GT video label과 정확히 일치합니다.")
        else:
            if comparison["false_positives"]:
                st.error("False Positive: " + ", ".join(comparison["false_positives"]))
            if comparison["false_negatives"]:
                st.error("False Negative: " + ", ".join(comparison["false_negatives"]))


def render_summary_tables(all_items: list[dict[str, Any]]) -> None:
    notrain_summary = aggregate_model_summary(all_items, "notrain")
    trained_summary = aggregate_model_summary(all_items, "trained")

    st.subheader("Overall Metrics")
    st.caption(
        f'전체 {len(all_items)}개 영상 기준입니다. '
        "Union IoU는 harmful 구간 union 기준이며, 빈 예측/빈 GT 조합은 1.0으로 계산합니다."
    )

    top_cols = st.columns(4)
    top_cols[0].metric("Notrain mean IoU", f'{notrain_summary["mean_iou"]:.3f}')
    top_cols[1].metric("Trained mean IoU", f'{trained_summary["mean_iou"]:.3f}')
    top_cols[2].metric("Notrain exact label", percent_text(notrain_summary["exact_match_accuracy"]))
    top_cols[3].metric("Trained exact label", percent_text(trained_summary["exact_match_accuracy"]))

    accuracy_rows = [
        {
            "Category": category,
            "Notrain Acc": percent_text(notrain_summary["category_accuracy"][category]),
            "Trained Acc": percent_text(trained_summary["category_accuracy"][category]),
        }
        for category in CATEGORY_ORDER
    ]
    st.dataframe(accuracy_rows, use_container_width=True, hide_index=True)


st.set_page_config(page_title="Video Guardrail Compare Demo", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stButton"] > button {
        text-align: left;
        white-space: normal;
        height: auto;
        min-height: 3rem;
        line-height: 1.4;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin: 0.1rem 0 0.5rem 0;
    }
    .category-chip {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        background: rgba(255, 75, 75, 0.12);
        border: 1px solid rgba(255, 75, 75, 0.28);
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Video Guardrail Response Comparison")
st.caption("왼쪽은 비디오, 가운데는 notrain 모델 harmful interval, 오른쪽은 학습된 모델 harmful interval입니다.")

with st.sidebar:
    st.header("Controls")
    harmful_only = st.checkbox("harmful interval이 있는 비디오만 보기", value=True)
    hide_sexual = st.checkbox("sexual/C1 영상 숨기기", value=True)
    only_label_mismatch = st.checkbox("두 모델 video label이 다른 샘플만", value=False)
    min_iou_gap = st.slider("두 모델 IoU 차이 최소값", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    path_expander = st.expander("Data Paths", expanded=False)
    with path_expander:
        st.code(
            "\n".join(
                [
                    f"notrain: {DEFAULT_NOTRAIN_PATH}",
                    f"trained: {DEFAULT_TRAINED_PATH}",
                    f"gt: {DEFAULT_GT_PATH}",
                ]
            )
        )

bundle = load_comparison_bundle(
    str(DEFAULT_NOTRAIN_PATH),
    str(DEFAULT_TRAINED_PATH),
    str(DEFAULT_GT_PATH),
)

render_summary_tables(bundle)

external_gt_count = sum(1 for item in bundle if item["gt_source"] == "external")
if external_gt_count == 0:
    st.warning(
        "지정한 GT JSONL은 현재 prediction records와 직접 매칭되지 않아, "
        "레코드 내부에 저장된 embedded GT를 사용합니다."
    )

visible_items = bundle

if hide_sexual:
    visible_items = [item for item in visible_items if not item["is_sexual"]]

parent_options = sorted({item["parent_group"] for item in visible_items})
selected_parents = st.sidebar.multiselect(
    "Parent directory",
    options=parent_options,
    default=parent_options,
)

if selected_parents:
    visible_items = [item for item in visible_items if item["parent_group"] in selected_parents]
else:
    visible_items = []

if only_label_mismatch:
    visible_items = [item for item in visible_items if item["model_label_mismatch"]]

if min_iou_gap > 0.0:
    visible_items = [item for item in visible_items if item["model_iou_gap"] >= min_iou_gap]

if harmful_only:
    visible_items = [
        item
        for item in visible_items
        if item["notrain_intervals"] or item["trained_intervals"] or item["gt_intervals"]
    ]

if not visible_items:
    st.error("현재 필터 조건에서 비교 가능한 비디오를 찾지 못했습니다.")
    st.stop()

selected_item = st.selectbox(
    "비교할 비디오",
    options=visible_items,
    format_func=format_interval_option,
)

current_video_key = selected_item["key"]
if st.session_state.get("active_video_key") != current_video_key:
    st.session_state["active_video_key"] = current_video_key
    st.session_state["seek_time"] = 0.0
    st.session_state["open_interval_key"] = None

seek_time = safe_float(st.session_state.get("seek_time", 0.0))

stats_cols = st.columns(4)
stats_cols[0].metric("Parent dir", selected_item["parent_group"])
stats_cols[1].metric("Notrain harmful", len(selected_item["notrain_intervals"]))
stats_cols[2].metric("Trained harmful", len(selected_item["trained_intervals"]))
stats_cols[3].metric("GT harmful", len(selected_item["gt_intervals"]))

compare_cols = st.columns(2)
compare_cols[0].metric("Model label mismatch", "Yes" if selected_item["model_label_mismatch"] else "No")
compare_cols[1].metric("Model IoU gap", f'{selected_item["model_iou_gap"]:.3f}')

eval_col1, eval_col2 = st.columns(2)
with eval_col1:
    render_model_eval_panel("Notrain evaluation", selected_item["notrain_metrics"])
with eval_col2:
    render_model_eval_panel("Trained evaluation", selected_item["trained_metrics"])

left_col, middle_col, right_col = st.columns([1.2, 1, 1], gap="large")

with left_col:
    st.subheader("Video")
    if selected_item["video_path"] and Path(selected_item["video_path"]).exists():
        try:
            st.video(
                load_video_bytes(selected_item["video_path"]),
                start_time=seek_time,
                autoplay=True,
                muted=False,
            )
            st.caption(f"현재 이동 시점: {seek_time:.2f}s")
        except Exception as exc:  # pragma: no cover - demo fallback
            st.error(f"비디오를 불러오지 못했습니다: {exc}")
            st.code(selected_item["video_path"])
    else:
        st.error("선택된 비디오 파일을 찾지 못했습니다.")
        if selected_item["video_path"]:
            st.code(selected_item["video_path"])

    render_gt_summary(selected_item["gt_intervals"])

with middle_col:
    render_interval_panel(
        "Not Trained Model",
        selected_item["notrain_intervals"],
        "notrain",
        current_video_key,
    )

with right_col:
    render_interval_panel(
        "Trained Model",
        selected_item["trained_intervals"],
        "trained",
        current_video_key,
    )