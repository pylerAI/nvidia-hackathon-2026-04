#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregate tIoU + GUARDRAIL metrics from eval artifacts.

Each record is a JSON object with ``ground_truth.intervals`` and ``prediction.intervals`` (``chunks`` /
``chunk_order``) as produced by ``eval_dump``. Preferred layout: ``iter_<step>/records.jsonl`` (one object
per line). Legacy per-file ``iter_<step>/pred_*.json`` and flat ``iter_<step>_pred_*.json`` are still read.
Alternatively, if a root contains regular ``*.jsonl`` files (excluding ``records.jsonl``), each JSONL file is
treated as one model result file and metrics are aggregated per model filename.

Matching: greedy maximum tIoU pairing with tIoU >= a cutoff (``--tiou-threshold`` or each value in
``--tiou-thresholds`` for a sweep). One GT segment maps to at most one prediction segment and vice versa.

Guardrail: multi-label over the C1–C6 keys; micro / macro F1, subset accuracy, Hamming loss.
When every counted label decision is a true negative (no TP/FP/FN mass), micro/macro F1 are defined as
1.0 so ``[]`` / all-false guardrails that match are not scored as zero F1.
**Micro F1 vs τ:** if raising τ only drops matches on segments whose GT and pred guardrails are all
false, strict-mode label counts do not change, so micro F1 can stay flat while coverage drops.
**Mean tIoU:** aggregated only over records with at least one matched pair (see
``frac_records_with_tiou_match``); zeros from no-match records are excluded so the mean stays ≥ τ
when every retained match has tIoU ≥ τ.
Per τ, ``mean_micro_f1`` is averaged **only over records with ≥1 GT–pred pair at tIoU ≥ τ**
(``n_records_for_f1_avg`` equals that count; same as ``n_records_with_tiou_match``). If none qualify,
it falls back to averaging over all non-skipped records. ``mean_micro_f1_all_records`` / ``n_records_all``
keep the mean over every evaluated row for comparison.
``--unmatched-policy`` controls whether localization misses count toward guardrail (strict) or not
(matched_only).

Threshold-free localization (printed once): **mean max tIoU per GT** (each GT vs best overlapping
prediction) and **mean max tIoU per pred** (each prediction vs best overlapping GT). These do not
use ``--tiou-threshold`` or greedy matching.

**Video vs chunk vs category:** Headline ``mean_micro_f1`` is the mean of each **video** record's
multilabel micro-F1 (bits pooled inside that video). **Chunk-pool** metrics sum TP/FP/FN/TN over every
binary guardrail decision. Micro P/R/F1 use only TP/FP/FN (**TN is excluded from F1 by definition**).
**Chunk-pool accuracy** is ``(TP+TN)/(TP+TN+FP+FN)``. ``n_guardrail_chunk_rows_total`` is the number of
segment-level guardrail rows (matched chunk pairs plus strict-unmatched segments); each row compares
all category bits once. ``n_binary_label_decisions_total`` counts every bit compared (equals the sum of
tp+fp+fn+tn at the pool). **Per-category** reports binary F1 and binary accuracy per guardrail key.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Iterable, Iterator, Optional


_LEGACY_FLAT = re.compile(r"^iter_(\d+)_pred_(\d+)\.json$", re.IGNORECASE)
_NESTED_PRED = re.compile(r"^pred_(\d+)\.json$", re.IGNORECASE)
_ITER_DIR = re.compile(r"^iter_(\d+)$", re.IGNORECASE)
RECORDS_JSONL = "records.jsonl"


def temporal_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    """Temporal IoU on the real line; returns 0.0 if intervals are degenerate or disjoint."""
    inter = min(a1, b1) - max(a0, b0)
    if inter <= 0.0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    if union <= 0.0:
        return 0.0
    return inter / union


def threshold_free_localization_scores(
    gt_segs: list[dict[str, Any]], pred_segs: list[dict[str, Any]]
) -> tuple[float, float]:
    """Mean max tIoU: (per-GT best overlap with any pred), (per-pred best overlap with any GT).

    Independent of any matching threshold; summarizes raw temporal overlap quality.
    """
    if not gt_segs:
        mean_gt = 0.0
    else:
        s_gt = 0.0
        for g in gt_segs:
            g0, g1 = g["start"], g["end"]
            if pred_segs:
                s_gt += max(temporal_iou(g0, g1, p["start"], p["end"]) for p in pred_segs)
        mean_gt = s_gt / len(gt_segs)

    if not pred_segs:
        mean_pr = 0.0
    else:
        s_pr = 0.0
        for p in pred_segs:
            p0, p1 = p["start"], p["end"]
            if gt_segs:
                s_pr += max(temporal_iou(g0, g1, p0, p1) for g0, g1 in ((g["start"], g["end"]) for g in gt_segs))
        mean_pr = s_pr / len(pred_segs)

    return mean_gt, mean_pr


def _chunks_dict(intervals: dict[str, Any]) -> dict[str, Any]:
    chunks = intervals.get("chunks")
    return chunks if isinstance(chunks, dict) else {}


def _ordered_chunk_keys(chunks: dict[str, Any], chunk_order: Any) -> list[str]:
    if isinstance(chunk_order, list) and chunk_order:
        keys = [k for k in chunk_order if isinstance(k, str) and k in chunks]
        rest = sorted(k for k in chunks if k not in set(keys))
        return keys + rest
    return sorted(chunks.keys())


def intervals_to_segments(intervals: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return list of segments with float start/end and guardrail dict."""
    if not intervals or not isinstance(intervals, dict):
        return []
    chunks = _chunks_dict(intervals)
    if not chunks:
        return []
    order = _ordered_chunk_keys(chunks, intervals.get("chunk_order"))
    out: list[dict[str, Any]] = []
    for k in order:
        c = chunks.get(k)
        if not isinstance(c, dict):
            continue
        try:
            s = float(c["start"])
            e = float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        gr = c.get("guardrail")
        if not isinstance(gr, dict):
            gr = {}
        out.append({"start": s, "end": e, "guardrail": gr})
    return out


def guardrail_keys_union(segments: Iterable[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for seg in segments:
        gr = seg.get("guardrail")
        if isinstance(gr, dict):
            keys.update(gr.keys())
    return sorted(keys)


def vectorize_guardrail(gr: dict[str, Any], keys: list[str]) -> list[int]:
    vec: list[int] = []
    for k in keys:
        v = gr.get(k, False)
        if isinstance(v, bool):
            vec.append(int(v))
        elif isinstance(v, (int, float)):
            vec.append(1 if v else 0)
        else:
            vec.append(0)
    return vec


@dataclass
class MatchResult:
    pairs: list[tuple[int, int, float]]  # gt_idx, pred_idx, tiou
    unmatched_gt: list[int]
    unmatched_pred: list[int]


def greedy_match_by_tiou(
    gt: list[dict[str, Any]], pred: list[dict[str, Any]], tiou_threshold: float
) -> MatchResult:
    """Greedy many-to-many matching: sort all admissible pairs by tIoU descending, take disjoint."""
    pairs_scored: list[tuple[float, int, int]] = []
    for i, g in enumerate(gt):
        g0, g1 = g["start"], g["end"]
        for j, p in enumerate(pred):
            t = temporal_iou(g0, g1, p["start"], p["end"])
            if t >= tiou_threshold:
                pairs_scored.append((t, i, j))
    pairs_scored.sort(key=lambda x: -x[0])

    used_g: set[int] = set()
    used_p: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for t, i, j in pairs_scored:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched.append((i, j, t))

    unmatched_gt = [i for i in range(len(gt)) if i not in used_g]
    unmatched_pred = [j for j in range(len(pred)) if j not in used_p]
    return MatchResult(pairs=matched, unmatched_gt=unmatched_gt, unmatched_pred=unmatched_pred)


@dataclass
class LabelStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add_pair(self, gt_bit: int, pred_bit: int) -> None:
        if gt_bit == 1 and pred_bit == 1:
            self.tp += 1
        elif gt_bit == 0 and pred_bit == 1:
            self.fp += 1
        elif gt_bit == 1 and pred_bit == 0:
            self.fn += 1
        else:
            # gt_bit == 0 and pred_bit == 0
            self.tn += 1


@dataclass
class FileMetrics:
    json_path: str
    iteration: int
    record_index: int
    num_gt: int
    num_pred: int
    num_matched: int
    mean_tiou_matched: float
    gt_coverage: float
    pred_match_rate: float
    subset_acc: float
    hamming_mean: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    per_label_f1: dict[str, float] = field(default_factory=dict)
    # Pooled TP/FP/FN/TN per guardrail key for this video (for dataset-level chunk/category aggregation).
    label_tp: dict[str, int] = field(default_factory=dict)
    label_fp: dict[str, int] = field(default_factory=dict)
    label_fn: dict[str, int] = field(default_factory=dict)
    label_tn: dict[str, int] = field(default_factory=dict)
    # subset_total: one row per matched chunk pair or strict unmatched segment (guardrail vector eval).
    n_guardrail_chunk_rows: int = 0
    # Total binary label decisions in this video (= hamming_denom; chunk_rows × n_keys typically).
    n_binary_label_decisions: int = 0
    mean_max_tiou_per_gt: float = 0.0
    mean_max_tiou_per_pred: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0.0 else 0.0


def compute_file_metrics(
    gt_segs: list[dict[str, Any]],
    pred_segs: list[dict[str, Any]],
    tiou_threshold: float,
    unmatched_policy: str,
) -> tuple[Optional[FileMetrics], str]:
    if not gt_segs and not pred_segs:
        return None, "empty_gt_and_pred"

    mean_max_tiou_per_gt, mean_max_tiou_per_pred = threshold_free_localization_scores(gt_segs, pred_segs)

    keys = sorted(set(guardrail_keys_union(gt_segs)) | set(guardrail_keys_union(pred_segs)))
    if not keys:
        return None, "no_guardrail_keys"

    m = greedy_match_by_tiou(gt_segs, pred_segs, tiou_threshold)
    per_label = {k: LabelStats() for k in keys}

    matched_tious: list[float] = []
    subset_hits = 0
    subset_total = 0
    hamming_denom = 0
    hamming_wrong = 0

    def add_vec_pair(gt_vec: list[int], pred_vec: list[int]) -> None:
        nonlocal subset_hits, subset_total, hamming_denom, hamming_wrong
        ok = True
        for ki, k in enumerate(keys):
            g_b, p_b = gt_vec[ki], pred_vec[ki]
            per_label[k].add_pair(g_b, p_b)
            hamming_denom += 1
            if g_b != p_b:
                hamming_wrong += 1
                ok = False
        subset_total += 1
        if ok:
            subset_hits += 1

    for gi, pj, t in m.pairs:
        matched_tious.append(t)
        gt_vec = vectorize_guardrail(gt_segs[gi]["guardrail"], keys)
        pred_vec = vectorize_guardrail(pred_segs[pj]["guardrail"], keys)
        add_vec_pair(gt_vec, pred_vec)

    zero = [0] * len(keys)
    if unmatched_policy == "strict":
        for gi in m.unmatched_gt:
            gt_vec = vectorize_guardrail(gt_segs[gi]["guardrail"], keys)
            add_vec_pair(gt_vec, zero)
        for pj in m.unmatched_pred:
            pred_vec = vectorize_guardrail(pred_segs[pj]["guardrail"], keys)
            add_vec_pair(zero, pred_vec)

    # Micro over all label decisions accumulated in per_label.
    # Standard TP/FP/FN ignore true negatives; if every bit is TN (gt=0,pred=0), sums are zero and
    # precision/recall/F1 are undefined — treat as perfect agreement (1.0).
    sum_tp = sum(s.tp for s in per_label.values())
    sum_fp = sum(s.fp for s in per_label.values())
    sum_fn = sum(s.fn for s in per_label.values())
    if sum_tp + sum_fp + sum_fn == 0:
        micro_p = 1.0
        micro_r = 1.0
        micro_f1 = 1.0
    else:
        micro_p = _safe_div(sum_tp, sum_tp + sum_fp)
        micro_r = _safe_div(sum_tp, sum_tp + sum_fn)
        micro_f1 = _safe_div(2.0 * micro_p * micro_r, micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    per_f1: dict[str, float] = {}
    macro_terms: list[float] = []
    for k, s in per_label.items():
        denom = 2.0 * s.tp + s.fp + s.fn
        if denom <= 0.0:
            per_f1[k] = 1.0
        else:
            per_f1[k] = _safe_div(2.0 * s.tp, denom)
        if s.tp + s.fp + s.fn > 0:
            macro_terms.append(per_f1[k])
    if macro_terms:
        macro_f1 = sum(macro_terms) / len(macro_terms)
    else:
        macro_f1 = 1.0 if hamming_wrong == 0 else 0.0

    mean_tiou = sum(matched_tious) / len(matched_tious) if matched_tious else 0.0
    n_gt = len(gt_segs)
    n_pr = len(pred_segs)
    gt_cov = _safe_div(len(m.pairs), n_gt) if n_gt else 1.0
    pr_match = _safe_div(len(m.pairs), n_pr) if n_pr else 1.0

    label_tp = {k: s.tp for k, s in per_label.items()}
    label_fp = {k: s.fp for k, s in per_label.items()}
    label_fn = {k: s.fn for k, s in per_label.items()}
    label_tn = {k: s.tn for k, s in per_label.items()}

    return (
        FileMetrics(
            json_path="",
            iteration=-1,
            record_index=-1,
            num_gt=n_gt,
            num_pred=n_pr,
            num_matched=len(m.pairs),
            mean_tiou_matched=mean_tiou,
            gt_coverage=gt_cov,
            pred_match_rate=pr_match,
            subset_acc=_safe_div(subset_hits, subset_total) if subset_total else 0.0,
            hamming_mean=_safe_div(hamming_wrong, hamming_denom) if hamming_denom else 0.0,
            micro_precision=micro_p,
            micro_recall=micro_r,
            micro_f1=micro_f1,
            macro_f1=macro_f1,
            per_label_f1=per_f1,
            label_tp=label_tp,
            label_fp=label_fp,
            label_fn=label_fn,
            label_tn=label_tn,
            n_guardrail_chunk_rows=subset_total,
            n_binary_label_decisions=hamming_denom,
            mean_max_tiou_per_gt=mean_max_tiou_per_gt,
            mean_max_tiou_per_pred=mean_max_tiou_per_pred,
        ),
        "",
    )


def parse_pred_filename(path: Path) -> Optional[tuple[int, int]]:
    """Return (training_iteration, jsonl_record_index) for eval_dump per-record JSON paths."""
    m = _NESTED_PRED.match(path.name)
    if m:
        pm = _ITER_DIR.match(path.parent.name)
        if pm:
            return int(pm.group(1)), int(m.group(1))
        return None
    m = _LEGACY_FLAT.match(path.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _iter_dirs_with_records_jsonl(root: Path) -> set[int]:
    iters: set[int] = set()
    for jsonl_path in root.glob(f"iter_*/{RECORDS_JSONL}"):
        if not jsonl_path.is_file():
            continue
        pm = _ITER_DIR.match(jsonl_path.parent.name)
        if pm:
            iters.add(int(pm.group(1)))
    return iters


def iter_eval_pred_json_paths(root: Path) -> Iterator[Path]:
    """Per-record JSON files (legacy flat + iter_/pred_*.json), excluding iters that use records.jsonl."""
    root_resolved = root.resolve()
    skip_iters = _iter_dirs_with_records_jsonl(root)
    paths: set[Path] = set()
    for p in root.glob("iter_*/pred_*.json"):
        if not p.is_file():
            continue
        pm = _ITER_DIR.match(p.parent.name)
        if pm and int(pm.group(1)) in skip_iters:
            continue
        paths.add(p.resolve())
    for p in root.glob("iter_*_pred_*.json"):
        if not p.is_file() or p.parent.resolve() != root_resolved:
            continue
        m = _LEGACY_FLAT.match(p.name)
        if m and int(m.group(1)) in skip_iters:
            continue
        paths.add(p.resolve())
    ordered = sorted(paths, key=lambda p: (parse_pred_filename(p) or (-1, -1), str(p)))
    for p in ordered:
        if parse_pred_filename(p) is not None:
            yield p


def iter_eval_dump_records(root: Path) -> Iterator[tuple[Path, int, int, Optional[dict[str, Any]]]]:
    """Yield ``(path, iteration, record_index, payload)`` for metrics (JSONL lines + legacy JSON files)."""
    for jsonl_path in sorted(root.glob(f"iter_*/{RECORDS_JSONL}"), key=lambda p: p.as_posix()):
        if not jsonl_path.is_file():
            continue
        pm = _ITER_DIR.match(jsonl_path.parent.name)
        if not pm:
            continue
        iteration = int(pm.group(1))
        try:
            with jsonl_path.open("r", encoding="utf-8") as fp:
                for line_idx, raw in enumerate(fp):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        yield jsonl_path, iteration, line_idx, None
                        continue
                    if not isinstance(obj, dict):
                        yield jsonl_path, iteration, line_idx, None
                        continue
                    rec_idx = int(obj.get("jsonl_record_index", line_idx))
                    yield jsonl_path, iteration, rec_idx, obj
        except OSError:
            continue

    for path in iter_eval_pred_json_paths(root):
        parsed = parse_pred_filename(path)
        if parsed is None:
            continue
        iteration, rec_idx = parsed
        yield path, iteration, rec_idx, load_one_json(path)


def load_one_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def iter_jsonl_records(path: Path) -> Iterator[tuple[int, Optional[dict[str, Any]]]]:
    """Yield ``(record_index, payload)`` from a JSONL file; malformed rows become ``payload=None``."""
    try:
        with path.open("r", encoding="utf-8") as fp:
            for line_idx, raw in enumerate(fp):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    yield line_idx, None
                    continue
                if not isinstance(obj, dict):
                    yield line_idx, None
                    continue
                try:
                    rec_idx = int(obj.get("jsonl_record_index", line_idx))
                except (TypeError, ValueError):
                    rec_idx = line_idx
                yield rec_idx, obj
    except OSError:
        return


def iter_model_result_jsonl_paths(root: Path) -> Iterator[Path]:
    """Yield model result JSONL files under ``root``, excluding eval_dump ``records.jsonl`` files."""
    seen: set[Path] = set()
    for path in sorted(root.rglob("*.jsonl"), key=lambda p: p.as_posix()):
        if not path.is_file() or path.name == RECORDS_JSONL:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def model_name_from_jsonl_path(root: Path, path: Path) -> str:
    """Stable model display name derived from file path relative to ``root``."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path(path.name)
    return rel.with_suffix("").as_posix()


def count_jsonl_records(path: Path) -> tuple[int, int]:
    """Return ``(total_rows, malformed_rows)`` for a JSONL file."""
    total = 0
    malformed = 0
    for _rec_idx, obj in iter_jsonl_records(path):
        total += 1
        if obj is None:
            malformed += 1
    return total, malformed


def build_record_metrics(
    path: Path,
    iteration: int,
    rec_idx: int,
    obj: Optional[dict[str, Any]],
    tiou_threshold: float,
    unmatched_policy: str,
) -> FileMetrics:
    """Compute one record's metrics or return a skipped row with the appropriate reason."""
    if obj is None:
        return FileMetrics(
            json_path=str(path),
            iteration=iteration,
            record_index=rec_idx,
            num_gt=0,
            num_pred=0,
            num_matched=0,
            mean_tiou_matched=0.0,
            gt_coverage=0.0,
            pred_match_rate=0.0,
            subset_acc=0.0,
            hamming_mean=0.0,
            micro_precision=0.0,
            micro_recall=0.0,
            micro_f1=0.0,
            macro_f1=0.0,
            skipped=True,
            skip_reason="json_load_failed",
        )

    gt = intervals_to_segments(obj.get("ground_truth", {}).get("intervals"))
    pred = intervals_to_segments(obj.get("prediction", {}).get("intervals"))
    metrics, skip = compute_file_metrics(gt, pred, tiou_threshold, unmatched_policy)
    if metrics is None:
        return FileMetrics(
            json_path=str(path),
            iteration=iteration,
            record_index=rec_idx,
            num_gt=len(gt),
            num_pred=len(pred),
            num_matched=0,
            mean_tiou_matched=0.0,
            gt_coverage=0.0,
            pred_match_rate=0.0,
            subset_acc=0.0,
            hamming_mean=0.0,
            micro_precision=0.0,
            micro_recall=0.0,
            micro_f1=0.0,
            macro_f1=0.0,
            skipped=True,
            skip_reason=skip,
        )

    metrics.json_path = str(path)
    metrics.iteration = iteration
    metrics.record_index = rec_idx
    return metrics


def count_eval_records_per_iteration(root: Path) -> tuple[dict[int, int], int, int]:
    """Count eval_dump rows used for metrics: ``(per_train_iter, total_rows, n_parse_fail)``."""
    per_iter: dict[int, int] = defaultdict(int)
    n_parse_fail = 0
    for _path, iteration, _rec_idx, obj in iter_eval_dump_records(root):
        per_iter[iteration] += 1
        if obj is None:
            n_parse_fail += 1
    total = sum(per_iter.values())
    return dict(sorted(per_iter.items())), total, n_parse_fail


def process_directory(
    root: Path,
    tiou_threshold: float,
    unmatched_policy: str,
) -> dict[int, list[FileMetrics]]:
    """Group metrics by training iteration (from path / payload)."""
    by_iter: dict[int, list[FileMetrics]] = defaultdict(list)
    for path, iteration, rec_idx, obj in iter_eval_dump_records(root):
        by_iter[iteration].append(
            build_record_metrics(path, iteration, rec_idx, obj, tiou_threshold, unmatched_policy)
        )
    return dict(by_iter)


def process_model_jsonl_file(
    path: Path,
    tiou_threshold: float,
    unmatched_policy: str,
) -> list[FileMetrics]:
    """Aggregate all record rows in one model JSONL file."""
    rows: list[FileMetrics] = []
    for rec_idx, obj in iter_jsonl_records(path):
        rows.append(build_record_metrics(path, -1, rec_idx, obj, tiou_threshold, unmatched_policy))
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pool_chunk_and_category_metrics(
    usable: list[FileMetrics],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Sum label TP/FP/FN/TN across videos; chunk-pool micro P/R/F1, per-key F1, per-key accuracy."""
    pooled: dict[str, LabelStats] = defaultdict(LabelStats)
    for r in usable:
        keys = set(r.label_tp.keys()) | set(r.label_fp.keys()) | set(r.label_fn.keys()) | set(r.label_tn.keys())
        for k in keys:
            pooled[k].tp += r.label_tp.get(k, 0)
            pooled[k].fp += r.label_fp.get(k, 0)
            pooled[k].fn += r.label_fn.get(k, 0)
            pooled[k].tn += r.label_tn.get(k, 0)

    g_tp = sum(s.tp for s in pooled.values())
    g_fp = sum(s.fp for s in pooled.values())
    g_fn = sum(s.fn for s in pooled.values())
    g_tn = sum(s.tn for s in pooled.values())
    total_bits = g_tp + g_fp + g_fn + g_tn
    if g_tp + g_fp + g_fn == 0:
        chunk_p, chunk_r, chunk_f1 = 1.0, 1.0, 1.0
    else:
        chunk_p = _safe_div(float(g_tp), float(g_tp + g_fp))
        chunk_r = _safe_div(float(g_tp), float(g_tp + g_fn))
        chunk_f1 = _safe_div(2.0 * chunk_p * chunk_r, chunk_p + chunk_r) if (chunk_p + chunk_r) > 0 else 0.0

    chunk_acc = _safe_div(float(g_tp + g_tn), float(total_bits)) if total_bits else 1.0

    per_cat_f1: dict[str, float] = {}
    per_cat_acc: dict[str, float] = {}
    for k in sorted(pooled.keys()):
        s = pooled[k]
        denom_f1 = 2.0 * s.tp + s.fp + s.fn
        if denom_f1 <= 0.0:
            per_cat_f1[k] = 1.0
        else:
            per_cat_f1[k] = _safe_div(2.0 * s.tp, denom_f1)
        n_k = s.tp + s.fp + s.fn + s.tn
        per_cat_acc[k] = _safe_div(float(s.tp + s.tn), float(n_k)) if n_k else 1.0

    n_chunk_rows = float(sum(r.n_guardrail_chunk_rows for r in usable))
    n_binary_sum = float(sum(r.n_binary_label_decisions for r in usable))

    scalars: dict[str, float] = {
        "chunk_pool_micro_precision": chunk_p,
        "chunk_pool_micro_recall": chunk_r,
        "chunk_pool_micro_f1": chunk_f1,
        "chunk_pool_accuracy": chunk_acc,
        "chunk_pool_sum_tp": float(g_tp),
        "chunk_pool_sum_fp": float(g_fp),
        "chunk_pool_sum_fn": float(g_fn),
        "chunk_pool_sum_tn": float(g_tn),
        "n_guardrail_chunk_rows_total": n_chunk_rows,
        "n_binary_label_decisions_total": n_binary_sum,
    }
    return scalars, per_cat_f1, per_cat_acc


def aggregate_iter(rows: list[FileMetrics]) -> dict[str, Any]:
    usable = [r for r in rows if not r.skipped]
    skipped = len(rows) - len(usable)
    # Per-file mean_tiou is 0 when num_matched==0; averaging those in would dilute below the τ cutoff
    # and contradict "matched pairs are all ≥ τ". Report mean only over records with ≥1 match.
    tiou_nonempty = [r.mean_tiou_matched for r in usable if r.num_matched > 0]
    f1_rows = [r for r in usable if r.num_matched > 0]
    mean_f1_all = mean([r.micro_f1 for r in usable]) if usable else 0.0
    if f1_rows:
        mean_f1_headline = mean([r.micro_f1 for r in f1_rows])
        n_f1_headline = float(len(f1_rows))
    else:
        mean_f1_headline = mean_f1_all
        n_f1_headline = float(len(usable))
    out: dict[str, Any] = {
        "n_files": float(len(rows)),
        "n_skipped": float(skipped),
        "mean_tiou_matched": mean(tiou_nonempty) if tiou_nonempty else 0.0,
        "frac_records_with_tiou_match": _safe_div(float(len(tiou_nonempty)), float(len(usable))) if usable else 0.0,
        "n_records_for_f1_avg": n_f1_headline,
        "n_records_with_tiou_match": float(len(tiou_nonempty)),
        "n_records_all": float(len(usable)),
        "mean_gt_coverage": mean([r.gt_coverage for r in usable]),
        "mean_pred_match_rate": mean([r.pred_match_rate for r in usable]),
        "mean_subset_acc": mean([r.subset_acc for r in usable]),
        "mean_hamming": mean([r.hamming_mean for r in usable]),
        "mean_micro_p": mean([r.micro_precision for r in usable]),
        "mean_micro_r": mean([r.micro_recall for r in usable]),
        "mean_micro_f1": mean_f1_headline,
        "mean_micro_f1_all_records": mean_f1_all,
        "mean_macro_f1": mean([r.macro_f1 for r in usable]),
        "mean_num_gt": mean([float(r.num_gt) for r in usable]),
        "mean_num_pred": mean([float(r.num_pred) for r in usable]),
        "mean_num_matched": mean([float(r.num_matched) for r in usable]),
        "mean_max_tiou_per_gt": mean([r.mean_max_tiou_per_gt for r in usable]),
        "mean_max_tiou_per_pred": mean([r.mean_max_tiou_per_pred for r in usable]),
    }
    chunk_scalars, per_category_f1, per_category_accuracy = pool_chunk_and_category_metrics(usable)
    out.update(chunk_scalars)
    out["per_category_f1"] = per_category_f1
    out["per_category_accuracy"] = per_category_accuracy
    return out


def print_model_mode_counts(names: list[str], dirs: list[Path], model_paths_by_run: dict[str, list[Path]]) -> None:
    print("=== eval sample counts (rows aggregated per model jsonl) ===")
    for name, root in zip(names, dirs):
        paths = model_paths_by_run.get(name, [])
        if not paths:
            print(f"  {name} ({root}): 0 model jsonl files", flush=True)
            continue
        print(f"  {name} ({root}): {len(paths)} model jsonl file(s)", flush=True)
        for path in paths:
            total, n_bad = count_jsonl_records(path)
            model_name = model_name_from_jsonl_path(root, path)
            bad_s = f", skipped/bad JSON: {n_bad}" if n_bad else ""
            print(f"    {model_name}: {total} record(s){bad_s}", flush=True)


def print_model_mode_tables(
    names: list[str],
    thresholds: list[float],
    unmatched_policy: str,
    all_runs: dict[float, dict[str, dict[str, dict[str, Any]]]],
) -> None:
    all_models = sorted(
        {
            model
            for tau_data in all_runs.values()
            for run_rows in tau_data.values()
            for model in run_rows
        }
    )
    if not all_models:
        return

    for tau in thresholds:
        runs_at_tau = all_runs[tau]
        print(
            f"\n=== model comparison  tIoU>={tau:g}  unmatched={unmatched_policy} ===\n"
            "model | " + " | ".join(names) + "\n"
            "--- f1, f1_all, tiou, coverage, hit, n_f1 ---"
        )
        for model in all_models:
            parts = [model]
            for name in names:
                row = runs_at_tau.get(name, {}).get(model)
                if not row:
                    parts.append("—")
                    continue
                nf = int(row["n_records_for_f1_avg"])
                na = int(row["n_records_all"])
                parts.append(
                    f"f1={row['mean_micro_f1']:.3f} "
                    f"f1_all={row['mean_micro_f1_all_records']:.3f} "
                    f"tiou={row['mean_tiou_matched']:.3f} "
                    f"coverage={row['mean_gt_coverage']:.3f} "
                    f"hit={row['frac_records_with_tiou_match']:.3f} "
                    f"n_f1={nf}/{na}"
                )
            print(" | ".join(parts))


def write_model_csv(
    csv_path: Path,
    names: list[str],
    dirs: list[Path],
    thresholds: list[float],
    unmatched_policy: str,
    all_runs: dict[float, dict[str, dict[str, dict[str, Any]]]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "model",
        "source_jsonl",
        "tiou_threshold",
        "unmatched_policy",
        "f1",
        "f1_all",
        "tiou",
        "coverage",
        "hit",
        "n_f1",
        "n_records_all",
    ]
    root_by_name = dict(zip(names, dirs))
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for tau in thresholds:
            for name in names:
                root = root_by_name[name]
                for model in sorted(all_runs[tau].get(name, {})):
                    row = all_runs[tau][name][model]
                    source_jsonl = str(root / f"{model}.jsonl")
                    if "source_jsonl" in row:
                        source_jsonl = str(row["source_jsonl"])
                    writer.writerow(
                        {
                            "run": name,
                            "model": model,
                            "source_jsonl": source_jsonl,
                            "tiou_threshold": tau,
                            "unmatched_policy": unmatched_policy,
                            "f1": row["mean_micro_f1"],
                            "f1_all": row["mean_micro_f1_all_records"],
                            "tiou": row["mean_tiou_matched"],
                            "coverage": row["mean_gt_coverage"],
                            "hit": row["frac_records_with_tiou_match"],
                            "n_f1": int(row["n_records_for_f1_avg"]),
                            "n_records_all": int(row["n_records_all"]),
                        }
                    )
    print(f"Wrote {csv_path}")


PLOT_METRICS: list[tuple[str, str]] = [
    ("f1", "F1"),
    ("f1_all", "F1 All"),
    ("tiou", "Mean tIoU"),
    ("coverage", "Coverage"),
    ("hit", "Hit"),
    ("n_f1", "n_f1"),
]


def _safe_float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _slugify_plot_name(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "plot"


def _annotate_bars(ax: Any, bars: Any, metric: str) -> None:
    for bar in bars:
        height = float(bar.get_height())
        label = f"{int(round(height))}" if metric == "n_f1" else f"{height:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def _plot_model_metric_group(rows: list[dict[str, Any]], run: str, tau: float, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: (-_safe_float_or_default(row["f1"]), str(row["model"])))
    models = [str(row["model"]) for row in rows]
    xs = list(range(len(rows)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Model comparison | run={run} | tIoU>={tau:g}", fontsize=16)
    for ax, (metric, title) in zip(axes.flat, PLOT_METRICS):
        values = [_safe_float_or_default(row[metric]) for row in rows]
        bars = ax.bar(xs, values, color="#4C78A8")
        _annotate_bars(ax, bars, metric)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(models, rotation=35, ha="right")
        upper = max(values) if values else 0.0
        if metric == "n_f1":
            ax.set_ylim(0.0, max(1.0, upper * 1.15 + 0.5))
        else:
            ax.set_ylim(0.0, max(1.0, upper * 1.10))
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    summary_path = out_dir / f"{_slugify_plot_name(run)}__tau_{tau:g}__summary.png"
    fig.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    for metric, title in PLOT_METRICS:
        values = [_safe_float_or_default(row[metric]) for row in rows]
        fig_single, ax_single = plt.subplots(figsize=(max(8, len(rows) * 1.3), 6))
        bars = ax_single.bar(xs, values, color="#F58518")
        _annotate_bars(ax_single, bars, metric)
        ax_single.set_title(f"{title} | run={run} | tIoU>={tau:g}")
        ax_single.set_xticks(xs)
        ax_single.set_xticklabels(models, rotation=35, ha="right")
        upper = max(values) if values else 0.0
        if metric == "n_f1":
            ax_single.set_ylim(0.0, max(1.0, upper * 1.15 + 0.5))
        else:
            ax_single.set_ylim(0.0, max(1.0, upper * 1.10))
        ax_single.grid(axis="y", linestyle="--", alpha=0.25)
        fig_single.tight_layout()
        metric_path = out_dir / f"{_slugify_plot_name(run)}__tau_{tau:g}__{_slugify_plot_name(metric)}.png"
        fig_single.savefig(metric_path, dpi=220, bbox_inches="tight")
        plt.close(fig_single)


def write_model_plots(
    plot_dir: Path,
    names: list[str],
    thresholds: list[float],
    all_runs: dict[float, dict[str, dict[str, dict[str, Any]]]],
) -> None:
    try:
        import matplotlib.pyplot as _plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it first, for example: pip install matplotlib"
        ) from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    for tau in thresholds:
        for name in names:
            model_rows = all_runs[tau].get(name, {})
            if not model_rows:
                continue
            rows = [
                {
                    "model": model,
                    "f1": row["mean_micro_f1"],
                    "f1_all": row["mean_micro_f1_all_records"],
                    "tiou": row["mean_tiou_matched"],
                    "coverage": row["mean_gt_coverage"],
                    "hit": row["frac_records_with_tiou_match"],
                    "n_f1": row["n_records_for_f1_avg"],
                }
                for model, row in sorted(model_rows.items())
            ]
            _plot_model_metric_group(rows, name, tau, plot_dir)
    print(f"Wrote plots to {plot_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dirs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more eval roots. Supports either iter_<step>/records.jsonl layout or directories containing "
        "model result *.jsonl files.",
    )
    p.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional display names for --dirs (same length as --dirs). Defaults to directory names.",
    )
    p.add_argument(
        "--tiou-threshold",
        type=float,
        default=0.5,
        help="Single tIoU cutoff when --tiou-thresholds is not set (default 0.5).",
    )
    p.add_argument(
        "--tiou-thresholds",
        nargs="+",
        type=float,
        default=None,
        metavar="TAU",
        help="If set, run metrics at each cutoff (e.g. 0.3 0.5 0.7). Overrides --tiou-threshold.",
    )
    p.add_argument(
        "--unmatched-policy",
        choices=("matched_only", "strict"),
        default="strict",
        help="strict: unmatched GT vs all-zero pred and unmatched pred vs all-zero GT also count for "
        "guardrail. matched_only: only matched pairs contribute.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="If set, write aggregate metrics to CSV. Iter-mode writes per (run, iteration); model-jsonl mode "
        "writes per (run, model).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="If set, write full aggregate outputs as JSON.",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="If set in model-jsonl mode, write model comparison bar charts to this directory.",
    )
    args = p.parse_args()

    names = args.names if args.names else [d.name for d in args.dirs]
    if len(names) != len(args.dirs):
        print("error: --names must have the same length as --dirs", file=sys.stderr)
        return 2

    if args.tiou_thresholds is not None:
        thresholds = sorted(set(float(t) for t in args.tiou_thresholds))
    else:
        thresholds = [float(args.tiou_threshold)]

    for d in args.dirs:
        if not d.is_dir():
            print(f"error: not a directory: {d}", file=sys.stderr)
            return 2

    model_paths_by_run = {name: list(iter_model_result_jsonl_paths(d)) for name, d in zip(names, args.dirs)}
    if any(model_paths_by_run.values()):
        print_model_mode_counts(names, args.dirs, model_paths_by_run)

        all_runs_model: dict[float, dict[str, dict[str, dict[str, Any]]]] = {}
        for tau in thresholds:
            by_tau: dict[str, dict[str, dict[str, Any]]] = {}
            for name, root in zip(names, args.dirs):
                agg_by_model: dict[str, dict[str, Any]] = {}
                for path in model_paths_by_run[name]:
                    rows = process_model_jsonl_file(path, tau, args.unmatched_policy)
                    if not rows:
                        continue
                    model_name = model_name_from_jsonl_path(root, path)
                    agg_row = aggregate_iter(rows)
                    agg_row["source_jsonl"] = str(path)
                    agg_by_model[model_name] = agg_row
                by_tau[name] = agg_by_model
            all_runs_model[tau] = by_tau

        if not any(run_rows for tau_rows in all_runs_model.values() for run_rows in tau_rows.values()):
            print("No model JSONL data found.", file=sys.stderr)
            return 1

        print_model_mode_tables(names, thresholds, args.unmatched_policy, all_runs_model)

        if args.csv:
            write_model_csv(args.csv, names, args.dirs, thresholds, args.unmatched_policy, all_runs_model)
        if args.plot_dir:
            write_model_plots(args.plot_dir, names, thresholds, all_runs_model)

        if args.json_out:
            payload = {
                "tiou_thresholds": thresholds,
                "unmatched_policy": args.unmatched_policy,
                "mode": "model_jsonl",
                "by_threshold": {
                    str(tau): {
                        "runs": {
                            name: {
                                model: {
                                    key: value
                                    for key, value in row.items()
                                }
                                for model, row in sorted(all_runs_model[tau][name].items())
                            }
                            for name in names
                        }
                    }
                    for tau in thresholds
                },
            }
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote {args.json_out}")
        return 0

    print("=== eval sample counts (rows aggregated per training iter) ===")
    for name, d in zip(names, args.dirs):
        per_iter, total, n_bad = count_eval_records_per_iteration(d)
        if not per_iter:
            print(f"  {name} ({d}): 0 records", flush=True)
            continue
        parts = [f"iter_{it:07d}={per_iter[it]}" for it in per_iter]
        bad_s = f", skipped/bad JSON: {n_bad}" if n_bad else ""
        print(f"  {name} ({d}): {total} record(s) total{bad_s}", flush=True)
        print("    " + " | ".join(parts), flush=True)

    # all_runs[tau][name][iter] -> aggregate row
    all_runs: dict[float, dict[str, dict[int, dict[str, float]]]] = {}

    for tau in thresholds:
        by_tau: dict[str, dict[int, dict[str, float]]] = {}
        for name, d in zip(names, args.dirs):
            grouped = process_directory(d, tau, args.unmatched_policy)
            agg: dict[int, dict[str, float]] = {}
            for it in sorted(grouped):
                agg[it] = aggregate_iter(grouped[it])
            by_tau[name] = agg
        all_runs[tau] = by_tau

    # Console: one block per threshold
    all_iters = sorted({it for tau_data in all_runs.values() for run in tau_data.values() for it in run})
    if not all_iters:
        print(
            "No eval_dump data found (iter_*/records.jsonl and/or legacy iter_*/pred_*.json / iter_*_pred_*.json).",
            file=sys.stderr,
        )
        return 1

    tau0 = thresholds[0]
    print(
        "\n=== localization (threshold-free; same for all τ) ===\n"
        "iteration | " + " | ".join(names) + "\n"
        "--- mean max tIoU per GT / mean max tIoU per pred ---"
    )
    for it in all_iters:
        parts = [str(it)]
        for name in names:
            row = all_runs[tau0][name].get(it)
            if not row:
                parts.append("—")
            else:
                parts.append(
                    f"maxGT={row['mean_max_tiou_per_gt']:.3f} maxPR={row['mean_max_tiou_per_pred']:.3f}"
                )
        print(" | ".join(parts))

    for tau in thresholds:
        runs_at_tau = all_runs[tau]
        print(
            f"\n=== tIoU>={tau:g}  unmatched={args.unmatched_policy} ===\n"
            "iteration | " + " | ".join(names) + "\n"
            "--- f1 = mean over n_f1 rows (≥1 loc match @τ); f1_all = all rows; mean_tiou subset same ---"
        )
        for it in all_iters:
            parts = [str(it)]
            for name in names:
                row = runs_at_tau[name].get(it)
                if not row:
                    parts.append("—")
                else:
                    nf = int(row["n_records_for_f1_avg"])
                    na = int(row["n_records_all"])
                    parts.append(
                        f"f1={row['mean_micro_f1']:.3f} f1_all={row['mean_micro_f1_all_records']:.3f} "
                        f"tiou={row['mean_tiou_matched']:.3f} cov={row['mean_gt_coverage']:.3f} "
                        f"hit={row['frac_records_with_tiou_match']:.2f} n_f1={nf}/{na}"
                    )
            print(" | ".join(parts))

        print(
            "\n--- chunk_pool: counts, TN, accuracy, F1 (F1 ignores TN by definition; acc uses TP+TN) ---\n"
            "iteration | " + " | ".join(names)
        )
        for it in all_iters:
            parts = [str(it)]
            for name in names:
                row = runs_at_tau[name].get(it)
                if not row:
                    parts.append("—")
                else:
                    pcf = row.get("per_category_f1") or {}
                    pca = row.get("per_category_accuracy") or {}
                    cats_f1 = " ".join(f"{k}={v:.3f}" for k, v in sorted(pcf.items()))
                    cats_a = " ".join(f"{k}={v:.3f}" for k, v in sorted(pca.items()))
                    parts.append(
                        f"chunks={int(row['n_guardrail_chunk_rows_total'])} "
                        f"bits={int(row['n_binary_label_decisions_total'])} "
                        f"acc={row['chunk_pool_accuracy']:.3f} | "
                        f"pool_f1={row['chunk_pool_micro_f1']:.3f} "
                        f"P={row['chunk_pool_micro_precision']:.3f} R={row['chunk_pool_micro_recall']:.3f} | "
                        f"tp={int(row['chunk_pool_sum_tp'])} fp={int(row['chunk_pool_sum_fp'])} "
                        f"fn={int(row['chunk_pool_sum_fn'])} tn={int(row['chunk_pool_sum_tn'])} | "
                        f"f1: {cats_f1} | acc: {cats_a}"
                    )
            print(" | ".join(parts))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run",
            "iteration",
            "tiou_threshold",
            "unmatched_policy",
            "n_files",
            "n_skipped",
            "mean_tiou_matched",
            "frac_records_with_tiou_match",
            "n_records_for_f1_avg",
            "n_records_with_tiou_match",
            "n_records_all",
            "mean_micro_f1_all_records",
            "mean_gt_coverage",
            "mean_pred_match_rate",
            "mean_subset_acc",
            "mean_hamming",
            "mean_micro_p",
            "mean_micro_r",
            "mean_micro_f1",
            "mean_macro_f1",
            "mean_num_gt",
            "mean_num_pred",
            "mean_num_matched",
            "mean_max_tiou_per_gt",
            "mean_max_tiou_per_pred",
            "chunk_pool_micro_f1",
            "chunk_pool_micro_precision",
            "chunk_pool_micro_recall",
            "chunk_pool_sum_tp",
            "chunk_pool_sum_fp",
            "chunk_pool_sum_fn",
            "chunk_pool_sum_tn",
            "chunk_pool_accuracy",
            "n_guardrail_chunk_rows_total",
            "n_binary_label_decisions_total",
            "per_category_f1_json",
            "per_category_accuracy_json",
        ]
        with args.csv.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            for tau in thresholds:
                for name in names:
                    for it in sorted(all_runs[tau][name]):
                        row = dict(all_runs[tau][name][it])
                        pcf = row.pop("per_category_f1", {})
                        pca = row.pop("per_category_accuracy", {})
                        flat = {
                            "run": name,
                            "iteration": it,
                            "tiou_threshold": tau,
                            "unmatched_policy": args.unmatched_policy,
                            **{k: row[k] for k in fieldnames if k in row and k not in ("per_category_f1_json", "per_category_accuracy_json")},
                            "per_category_f1_json": json.dumps(pcf, sort_keys=True, ensure_ascii=False),
                            "per_category_accuracy_json": json.dumps(pca, sort_keys=True, ensure_ascii=False),
                        }
                        w.writerow(flat)
        print(f"Wrote {args.csv}")

    if args.json_out:
        payload = {
            "tiou_thresholds": thresholds,
            "unmatched_policy": args.unmatched_policy,
            "by_threshold": {
                str(tau): {
                    "runs": {
                        name: {str(k): v for k, v in sorted(all_runs[tau][name].items())} for name in names
                    }
                }
                for tau in thresholds
            },
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
