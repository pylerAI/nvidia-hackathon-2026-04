from __future__ import annotations

from nvidia_hackathon.safewatch_eval_metrics import (
    ShotInterval,
    compute_temporal_metrics,
    compute_video_level_metrics,
    greedy_match_by_tiou,
    parse_shot_intervals,
    video_categories_from_shots,
)


def _shot(start_sec: float, end_sec: float, *, c3: bool = False, c4: bool = False) -> ShotInterval:
    return ShotInterval(
        start_sec=start_sec,
        end_sec=end_sec,
        description="desc",
        explanation="exp",
        guardrail={
            "C1(Sexual Content)": False,
            "C2(Harassment & Bullying)": False,
            "C3(Threats, Violence & Harm)": c3,
            "C4(False & Deceptive Information)": c4,
            "C5(Illegal/Regulated Activities)": False,
            "C6(Hateful Content & Extremism)": False,
        },
    )


def test_parse_shot_intervals_extracts_json_list_from_assistant_text() -> None:
    raw_text = """
<SPECIAL_11>Assistant
[
  {
    "START_SEC": 0.0,
    "END_SEC": 3.0,
    "DESCRIPTION": "safe shot",
    "GUARDRAIL": {
      "C1(Sexual Content)": false,
      "C2(Harassment & Bullying)": false,
      "C3(Threats, Violence & Harm)": false,
      "C4(False & Deceptive Information)": false,
      "C5(Illegal/Regulated Activities)": false,
      "C6(Hateful Content & Extremism)": false
    },
    "EXPLANATION": "safe"
  },
  {
    "START_SEC": 3.0,
    "END_SEC": 6.0,
    "DESCRIPTION": "harmful shot",
    "GUARDRAIL": {
      "C1(Sexual Content)": false,
      "C2(Harassment & Bullying)": false,
      "C3(Threats, Violence & Harm)": true,
      "C4(False & Deceptive Information)": false,
      "C5(Illegal/Regulated Activities)": false,
      "C6(Hateful Content & Extremism)": false
    },
    "EXPLANATION": "harmful"
  }
]
""".strip()

    shots = parse_shot_intervals(raw_text)

    assert len(shots) == 2
    assert shots[1].start_sec == 3.0
    assert shots[1].guardrail["C3(Threats, Violence & Harm)"] is True


def test_video_categories_from_shots_ors_flags_across_intervals() -> None:
    categories = video_categories_from_shots([_shot(0.0, 2.0, c3=True), _shot(2.0, 4.0, c4=True)])
    assert categories == ["C3", "C4"]


def test_compute_video_level_metrics_uses_or_aggregated_categories() -> None:
    metrics = compute_video_level_metrics(
        [
            {
                "id": "sample_1",
                "gt_shots": [_shot(0.0, 2.0, c3=True), _shot(2.0, 4.0)],
                "pred_shots": [_shot(0.0, 4.0, c3=True)],
            },
            {
                "id": "sample_2",
                "gt_shots": [_shot(0.0, 2.0)],
                "pred_shots": [_shot(0.0, 2.0, c4=True)],
            },
        ]
    )

    assert metrics["num_samples"] == 2
    assert metrics["per_category"]["C3"]["tp"] == 1
    assert metrics["per_category"]["C4"]["fp"] == 1


def test_greedy_match_by_tiou_prefers_best_non_overlapping_pairs() -> None:
    matches = greedy_match_by_tiou(
        predicted_intervals=[(0.0, 5.0), (5.0, 10.0)],
        gt_intervals=[(0.0, 4.0), (6.0, 10.0)],
        threshold=0.5,
    )

    assert len(matches) == 2
    assert matches[0]["tiou"] >= matches[1]["tiou"]


def test_compute_temporal_metrics_reports_thresholded_matches() -> None:
    metrics = compute_temporal_metrics(
        [
            {
                "id": "sample_1",
                "gt_shots": [_shot(0.0, 4.0, c3=True), _shot(4.0, 8.0, c4=True)],
                "pred_shots": [_shot(0.0, 4.0, c3=True), _shot(4.0, 7.0, c4=True)],
            }
        ]
    )

    c3_metrics = metrics["per_category"]["C3"]
    c4_metrics = metrics["per_category"]["C4"]

    assert c3_metrics["matched_at_0_5"] == 1
    assert c3_metrics["matched_at_0_7"] == 1
    assert c3_metrics["accuracy_at_0_5"] == 1.0
    assert c4_metrics["matched_at_0_5"] == 1
    assert c4_metrics["matched_at_0_7"] == 1
    assert c4_metrics["mean_tiou_matched"] is not None
