from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .prompt_template import STANDARD_CATEGORY_CODES, STANDARD_GUARDRAIL


@dataclass(frozen=True)
class ShotInterval:
    start_sec: float
    end_sec: float
    description: str
    explanation: str
    guardrail: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_assistant_response_text(text: str) -> str:
    assistant_markers = ["<SPECIAL_11>Assistant", "Assistant\n"]
    extracted = text
    for marker in assistant_markers:
        marker_index = extracted.rfind(marker)
        if marker_index != -1:
            extracted = extracted[marker_index + len(marker) :]
            break
    return extracted.replace("<think></think>", "").strip()


def _normalize_guardrail(raw_guardrail: dict[str, Any]) -> dict[str, bool]:
    normalized: dict[str, bool] = {}
    for standard_key in STANDARD_GUARDRAIL:
        code = standard_key.split("(", 1)[0]
        matched_value = None
        for raw_key, raw_value in raw_guardrail.items():
            raw_key_str = str(raw_key)
            if raw_key_str == standard_key or raw_key_str == code or raw_key_str.startswith(f"{code}("):
                matched_value = bool(raw_value)
                break
        normalized[standard_key] = matched_value if matched_value is not None else False
    return normalized


def parse_interval_json_list(text: str) -> list[dict[str, Any]]:
    assistant_text = extract_assistant_response_text(text).strip()
    candidates: list[str] = []
    if assistant_text.startswith("[") and assistant_text.endswith("]"):
        candidates.append(assistant_text)

    starts = [index for index, char in enumerate(assistant_text) if char == "["]
    for start_index in reversed(starts):
        candidate = assistant_text[start_index:].strip()
        if not candidate.endswith("]"):
            end_index = assistant_text.rfind("]")
            if end_index > start_index:
                candidate = assistant_text[start_index : end_index + 1].strip()
        if candidate:
            candidates.append(candidate)

    candidates.append(assistant_text)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return payload
    raise ValueError(f"Could not parse interval JSON list:\n{assistant_text}")


def normalize_shot_list(payload: list[dict[str, Any]]) -> list[ShotInterval]:
    normalized_shots: list[ShotInterval] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            start_sec = float(item.get("START_SEC", 0.0))
            end_sec = float(item.get("END_SEC", 0.0))
        except (TypeError, ValueError):
            continue
        if end_sec < start_sec:
            continue
        raw_guardrail = item.get("GUARDRAIL", {})
        if not isinstance(raw_guardrail, dict):
            raw_guardrail = {}
        normalized_shots.append(
            ShotInterval(
                start_sec=round(start_sec, 6),
                end_sec=round(end_sec, 6),
                description=str(item.get("DESCRIPTION", "")),
                explanation=str(item.get("EXPLANATION", "")),
                guardrail=_normalize_guardrail(raw_guardrail),
            )
        )
    return normalized_shots


def parse_shot_intervals(text: str) -> list[ShotInterval]:
    return normalize_shot_list(parse_interval_json_list(text))


def guardrail_true_codes(guardrail: dict[str, bool]) -> list[str]:
    true_codes: list[str] = []
    for code in STANDARD_CATEGORY_CODES:
        matched_key = next((key for key in guardrail if key.startswith(f"{code}(")), None)
        if matched_key and bool(guardrail[matched_key]):
            true_codes.append(code)
    return true_codes


def video_categories_from_shots(shots: list[ShotInterval]) -> list[str]:
    categories = {
        code
        for shot in shots
        for code in guardrail_true_codes(shot.guardrail)
    }
    return sorted(categories)


def compute_per_category_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_samples": len(results),
        "per_category": {},
    }
    for category in STANDARD_CATEGORY_CODES:
        tp = tn = fp = fn = 0
        for result in results:
            gt_has = category in result["gt_categories"]
            pred_has = category in result["pred_categories"]
            if gt_has and pred_has:
                tp += 1
            elif gt_has and not pred_has:
                fn += 1
            elif not gt_has and pred_has:
                fp += 1
            else:
                tn += 1

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 0.0
        summary["per_category"][category] = {
            "accuracy": accuracy,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    summary["exact_match_accuracy"] = (
        sum(set(result["gt_categories"]) == set(result["pred_categories"]) for result in results) / len(results)
        if results
        else 0.0
    )
    return summary


def interval_tiou(first: tuple[float, float], second: tuple[float, float]) -> float:
    start = max(first[0], second[0])
    end = min(first[1], second[1])
    intersection = max(0.0, end - start)
    union = max(first[1], second[1]) - min(first[0], second[0])
    if union <= 0:
        return 0.0
    return intersection / union


def _category_intervals(shots: list[ShotInterval], category: str) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for shot in shots:
        if category in guardrail_true_codes(shot.guardrail):
            intervals.append((shot.start_sec, shot.end_sec))
    return intervals


def greedy_match_by_tiou(
    predicted_intervals: list[tuple[float, float]],
    gt_intervals: list[tuple[float, float]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    effective_threshold = threshold if threshold > 0 else 1e-9
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred_interval in enumerate(predicted_intervals):
        for gt_index, gt_interval in enumerate(gt_intervals):
            tiou = interval_tiou(pred_interval, gt_interval)
            if tiou >= effective_threshold:
                candidates.append((tiou, pred_index, gt_index))
    candidates.sort(key=lambda item: item[0], reverse=True)

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    for tiou, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append(
            {
                "pred_index": pred_index,
                "gt_index": gt_index,
                "tiou": round(tiou, 6),
                "pred_interval": predicted_intervals[pred_index],
                "gt_interval": gt_intervals[gt_index],
            }
        )
    return matches


def _summarize_category_temporal_metrics(
    *,
    category: str,
    gt_intervals: list[tuple[float, float]],
    pred_intervals: list[tuple[float, float]],
) -> dict[str, Any]:
    matches_any = greedy_match_by_tiou(pred_intervals, gt_intervals, threshold=0.0)
    matches_05 = greedy_match_by_tiou(pred_intervals, gt_intervals, threshold=0.5)
    matches_07 = greedy_match_by_tiou(pred_intervals, gt_intervals, threshold=0.7)

    def safe_ratio(numerator: int, denominator: int) -> float | None:
        if denominator <= 0:
            return None
        return numerator / denominator

    return {
        "category": category,
        "num_gt_intervals": len(gt_intervals),
        "num_pred_intervals": len(pred_intervals),
        "matched_at_0_5": len(matches_05),
        "matched_at_0_7": len(matches_07),
        "accuracy_at_0_5": safe_ratio(len(matches_05), len(gt_intervals)),
        "accuracy_at_0_7": safe_ratio(len(matches_07), len(gt_intervals)),
        "precision_at_0_5": safe_ratio(len(matches_05), len(pred_intervals)),
        "precision_at_0_7": safe_ratio(len(matches_07), len(pred_intervals)),
        "mean_tiou_matched": (
            sum(match["tiou"] for match in matches_any) / len(matches_any) if matches_any else None
        ),
        "mean_tiou_at_0_5": (
            sum(match["tiou"] for match in matches_05) / len(matches_05) if matches_05 else None
        ),
        "mean_tiou_at_0_7": (
            sum(match["tiou"] for match in matches_07) / len(matches_07) if matches_07 else None
        ),
        "matches": {
            "any": matches_any,
            "0.5": matches_05,
            "0.7": matches_07,
        },
    }


def compute_temporal_metrics(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    per_category: dict[str, dict[str, Any]] = {}
    for category in STANDARD_CATEGORY_CODES:
        gt_intervals: list[tuple[float, float]] = []
        pred_intervals: list[tuple[float, float]] = []
        for record in prediction_records:
            gt_intervals.extend(_category_intervals(record["gt_shots"], category))
            pred_intervals.extend(_category_intervals(record["pred_shots"], category))
        per_category[category] = _summarize_category_temporal_metrics(
            category=category,
            gt_intervals=gt_intervals,
            pred_intervals=pred_intervals,
        )

    def macro_average(metric_name: str) -> float | None:
        values = [
            category_metrics[metric_name]
            for category_metrics in per_category.values()
            if category_metrics[metric_name] is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "num_samples": len(prediction_records),
        "per_category": per_category,
        "macro_accuracy_at_0_5": macro_average("accuracy_at_0_5"),
        "macro_accuracy_at_0_7": macro_average("accuracy_at_0_7"),
        "macro_mean_tiou_matched": macro_average("mean_tiou_matched"),
        "macro_mean_tiou_at_0_5": macro_average("mean_tiou_at_0_5"),
        "macro_mean_tiou_at_0_7": macro_average("mean_tiou_at_0_7"),
    }


def compute_video_level_metrics(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    results = [
        {
            "id": record["id"],
            "gt_categories": video_categories_from_shots(record["gt_shots"]),
            "pred_categories": video_categories_from_shots(record["pred_shots"]),
        }
        for record in prediction_records
    ]
    return compute_per_category_accuracy(results)
