import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import nvidia_hackathon.prompt_template as prompt_template
from data_curation.build_safewatch_sft_jsonl import (
    STANDARD_GUARDRAIL,
    VideoSample,
    annotation_path_to_full_video,
    build_single_shot_from_full_response,
    has_acceptable_shot_coverage,
    build_assistant_response,
    choose_full_response,
    clip_time_range,
    normalize_clip_response,
    parse_safewatch_response,
    split_train_eval_by_source_category,
    summarize_samples,
    ShotRecord,
)


def test_parse_safewatch_response_extracts_standard_fields():
    raw = """DESCRIPTION: A person is punching another person.
GUARDRAIL: {
  "C1(Sexual Content)": false,
  "C2(Harassment & Bullying)": true,
  "C3(Threats, Violence & Harm)": true,
  "C4(False & Deceptive Information)": false,
  "C5(Illegal/Regulated Activities)": false,
  "C6(Hateful Content & Extremism)": false
}
EXPLANATION: The clip shows harassment and physical harm."""

    parsed = parse_safewatch_response(raw)

    assert parsed is not None
    assert parsed.description == "A person is punching another person."
    assert parsed.guardrail["C2(Harassment & Bullying)"] is True
    assert parsed.guardrail["C3(Threats, Violence & Harm)"] is True
    assert parsed.explanation == "The clip shows harassment and physical harm."
    assert parsed.is_standard is True
    assert parsed.has_harm is True
    assert parsed.has_required_details is True


def test_normalize_clip_response_keeps_harmful_missing_details_excluded():
    parsed = parse_safewatch_response(
        """DESCRIPTION:
GUARDRAIL: {
  "C1(Sexual Content)": false,
  "C2(Harassment & Bullying)": true,
  "C3(Threats, Violence & Harm)": false,
  "C4(False & Deceptive Information)": false,
  "C5(Illegal/Regulated Activities)": false,
  "C6(Hateful Content & Extremism)": false
}
EXPLANATION:"""
    )

    assert parsed is not None
    assert normalize_clip_response(parsed) is None


def test_normalize_clip_response_fills_benign_missing_details_with_none():
    parsed = parse_safewatch_response(
        """DESCRIPTION:
GUARDRAIL: {
  "C1(Sexual Content)": false,
  "C2(Harassment & Bullying)": false,
  "C3(Threats, Violence & Harm)": false,
  "C4(False & Deceptive Information)": false,
  "C5(Illegal/Regulated Activities)": false,
  "C6(Hateful Content & Extremism)": false
}
EXPLANATION:"""
    )

    assert parsed is not None
    normalized = normalize_clip_response(parsed)
    assert normalized is not None
    assert normalized.description == "None"
    assert normalized.explanation == "None"
    assert normalized.has_harm is False


def test_choose_full_response_prefers_standard_normalized_response():
    nonstandard = parse_safewatch_response(
        """DESCRIPTION: partial response
GUARDRAIL: {
  "C1(Sexual Content)": true
}
EXPLANATION: partial"""
    )
    benign_missing = parse_safewatch_response(
        """DESCRIPTION:
GUARDRAIL: {
  "C1(Sexual Content)": false,
  "C2(Harassment & Bullying)": false,
  "C3(Threats, Violence & Harm)": false,
  "C4(False & Deceptive Information)": false,
  "C5(Illegal/Regulated Activities)": false,
  "C6(Hateful Content & Extremism)": false
}
EXPLANATION:"""
    )

    assert nonstandard is not None
    assert benign_missing is not None

    selected = choose_full_response([nonstandard, benign_missing])
    assert selected is not None
    assert selected.description == "None"
    assert selected.explanation == "None"


def test_load_policy_excerpts_backfills_missing_categories_from_defaults(monkeypatch):
    monkeypatch.setattr(
        prompt_template,
        "video_moderation_prompt_vg_fine_grained_final",
        """<box>
C1: Sexual Content:
Core Value: custom text
</box>""",
    )

    excerpts = prompt_template._load_safewatch_policy_excerpts(Path("."), prompt_template.STANDARD_CATEGORY_CODES)

    assert tuple(excerpts) == prompt_template.STANDARD_CATEGORY_CODES
    assert excerpts["C1"] == "C1: Sexual Content:\nCore Value: custom text"
    assert "C4: False & Deceptive Information:" in excerpts["C4"]
    assert "C5: Illegal/Regulated Activities:" in excerpts["C5"]
    assert "C6: Hateful Content & Extremism:" in excerpts["C6"]


def test_build_single_shot_from_full_response_spans_entire_video():
    parsed = parse_safewatch_response(
        """DESCRIPTION: A calm landscape scene.
GUARDRAIL: {
  "C1(Sexual Content)": false,
  "C2(Harassment & Bullying)": false,
  "C3(Threats, Violence & Harm)": false,
  "C4(False & Deceptive Information)": false,
  "C5(Illegal/Regulated Activities)": false,
  "C6(Hateful Content & Extremism)": false
}
EXPLANATION: The full video is benign."""
    )

    assert parsed is not None
    shot = build_single_shot_from_full_response(
        "dataset/full/benign_sharegpt4video/target/example.mp4",
        12.3456789,
        parsed,
    )

    assert shot.start_sec == 0.0
    assert shot.end_sec == 12.345679
    assert shot.description == "A calm landscape scene."
    assert shot.explanation == "The full video is benign."
    assert shot.clip_relative_path.endswith("#full_fallback")


def test_annotation_path_to_full_video_maps_clip_path():
    clip_path = "dataset/clip/misinformation_1/7041921581278416158/000003_000006.mp4"

    full_path = annotation_path_to_full_video(clip_path)

    assert full_path == "dataset/full/misinformation_1/target/7041921581278416158.mp4"


def test_clip_time_range_parses_seconds():
    assert clip_time_range("dataset/clip/foo/bar/000003_000006.mp4") == (3.0, 6.0)
    assert clip_time_range("dataset/clip/foo/bar/000004_000004.mp4") is None


def test_build_assistant_response_serializes_interval_payload():
    interval = ShotRecord(
        start_sec=3.0,
        end_sec=6.0,
        description="A misleading claim appears on screen.",
        guardrail={**STANDARD_GUARDRAIL, "C4(False & Deceptive Information)": True},
        explanation="The segment includes deceptive AI-generated scientific claims.",
        clip_relative_path="dataset/clip/misinformation_1/example/000003_000006.mp4",
    )

    payload = json.loads(build_assistant_response("A space-themed video.", [interval]))

    assert payload[0]["START_SEC"] == 3.0
    assert payload[0]["END_SEC"] == 6.0
    assert payload[0]["DESCRIPTION"] == "A misleading claim appears on screen."
    assert payload[0]["GUARDRAIL"]["C4(False & Deceptive Information)"] is True
    assert payload[0]["EXPLANATION"] == "The segment includes deceptive AI-generated scientific claims."


def test_build_assistant_response_returns_empty_list_for_benign_video():
    payload = json.loads(build_assistant_response("A benign video.", []))

    assert payload == []


def _make_sample(sample_id: str, source_category: str, guardrails: list[dict[str, bool]]) -> VideoSample:
    shots = [
        ShotRecord(
            start_sec=float(index),
            end_sec=float(index + 1),
            description=f"segment {index}",
            guardrail=guardrail,
            explanation=f"why {index}",
            clip_relative_path=f"dataset/clip/{source_category}/{sample_id}/{index:06d}_{index+1:06d}.mp4",
        )
        for index, guardrail in enumerate(guardrails)
    ]
    return VideoSample(
        sample_id=sample_id,
        video_path=f"/tmp/{sample_id}.mp4",
        video_relative_path=f"dataset/full/{source_category}/target/{sample_id}.mp4",
        duration_sec=10.0,
        prompt="prompt",
        assistant_response="[]",
        video_description="desc",
        shots=shots,
        source_category=source_category,
    )


def test_summarize_samples_counts_subcategories_and_main_categories():
    samples = [
        _make_sample("a", "abuse_1", [{**STANDARD_GUARDRAIL, "C2(Harassment & Bullying)": True}]),
        _make_sample("b", "abuse_1", []),
        _make_sample("c", "misinformation_1", [{**STANDARD_GUARDRAIL, "C4(False & Deceptive Information)": True}]),
    ]

    summary = summarize_samples(samples)

    assert summary["num_samples"] == 3
    assert summary["num_harmful_samples"] == 2
    assert summary["num_benign_samples"] == 1
    assert summary["num_total_shots"] == 2
    assert summary["num_harmful_shots"] == 2
    assert summary["num_benign_shots"] == 0
    assert summary["source_category_counts"]["abuse_1"] == 2
    assert summary["source_category_counts"]["misinformation_1"] == 1
    assert summary["main_category_sample_counts"]["C2"] == 1
    assert summary["main_category_sample_counts"]["C4"] == 1
    assert summary["main_category_shot_counts"]["C2"] == 1
    assert summary["main_category_shot_counts"]["C4"] == 1


def test_has_acceptable_shot_coverage_allows_small_gaps_but_not_large_gaps():
    assert has_acceptable_shot_coverage(
        [
            ShotRecord(3.0, 4.0, "a", STANDARD_GUARDRAIL, "safe", "clip_a"),
            ShotRecord(4.5, 6.0, "b", STANDARD_GUARDRAIL, "safe", "clip_b"),
        ],
        duration_sec=10.0,
    )
    assert not has_acceptable_shot_coverage(
        [
            ShotRecord(6.0, 7.0, "a", STANDARD_GUARDRAIL, "safe", "clip_a"),
            ShotRecord(7.0, 9.0, "b", STANDARD_GUARDRAIL, "safe", "clip_b"),
        ],
        duration_sec=10.0,
    )
    assert not has_acceptable_shot_coverage(
        [
            ShotRecord(0.0, 2.0, "a", STANDARD_GUARDRAIL, "safe", "clip_a"),
            ShotRecord(8.5, 9.0, "b", STANDARD_GUARDRAIL, "safe", "clip_b"),
        ],
        duration_sec=10.0,
    )
    assert not has_acceptable_shot_coverage(
        [
            ShotRecord(0.0, 4.0, "a", STANDARD_GUARDRAIL, "safe", "clip_a"),
            ShotRecord(3.5, 7.0, "b", STANDARD_GUARDRAIL, "safe", "clip_b"),
        ],
        duration_sec=10.0,
    )


def test_split_train_eval_by_source_category_uses_ten_percent_ratio():
    samples = [
        _make_sample("a1", "abuse_1", []),
        _make_sample("a2", "abuse_1", []),
        _make_sample("a3", "abuse_1", []),
        _make_sample("m1", "misinformation_1", []),
        _make_sample("m2", "misinformation_1", []),
        _make_sample("m3", "misinformation_1", []),
        _make_sample("m4", "misinformation_1", []),
        _make_sample("m5", "misinformation_1", []),
        _make_sample("m6", "misinformation_1", []),
        _make_sample("m7", "misinformation_1", []),
        _make_sample("m8", "misinformation_1", []),
        _make_sample("m9", "misinformation_1", []),
        _make_sample("m10", "misinformation_1", []),
    ]

    train_samples, eval_samples, split_summary = split_train_eval_by_source_category(
        samples,
        eval_ratio=0.1,
        seed=13,
    )

    assert len(train_samples) == 11
    assert len(eval_samples) == 2
    assert split_summary["eval_counts_by_source_category"]["abuse_1"] == 1
    assert split_summary["eval_counts_by_source_category"]["misinformation_1"] == 1
    assert split_summary["train_summary"]["source_category_counts"]["abuse_1"] == 2
    assert split_summary["train_summary"]["source_category_counts"]["misinformation_1"] == 9
