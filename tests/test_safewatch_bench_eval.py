from __future__ import annotations

from nvidia_hackathon.safewatch_bench_eval import (
    compute_per_category_accuracy,
    guardrail_true_codes,
    parse_full_method_categories,
    parse_interval_method_categories,
)


def test_guardrail_true_codes_extracts_expected_categories() -> None:
    guardrail = {
        "C1(Sexual Content)": False,
        "C2(Harassment & Bullying)": True,
        "C3(Threats, Violence & Harm)": False,
        "C4(False & Deceptive Information)": True,
        "C5(Illegal/Regulated Activities)": False,
        "C6(Hateful Content & Extremism)": False,
    }
    assert guardrail_true_codes(guardrail) == ["C2", "C4"]


def test_parse_full_method_categories_parses_safewatch_format() -> None:
    text = """
<SPECIAL_11>Assistant
DESCRIPTION: A person is making false medical claims.
GUARDRAIL: {
  "C1(Sexual Content)": false,
  "C2(Harassment & Bullying)": false,
  "C3(Threats, Violence & Harm)": false,
  "C4(False & Deceptive Information)": true,
  "C5(Illegal/Regulated Activities)": false,
  "C6(Hateful Content & Extremism)": false
}
EXPLANATION: The video contains misinformation.
""".strip()

    categories, normalized_output = parse_full_method_categories(text)
    assert categories == ["C4"]
    assert '"C4(False & Deceptive Information)": true' in normalized_output


def test_parse_interval_method_categories_unions_shot_guardrails() -> None:
    text = """
<SPECIAL_11>Assistant
[
  {
    "START_SEC": 0.0,
    "END_SEC": 4.0,
    "DESCRIPTION": "benign shot",
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
    "START_SEC": 4.0,
    "END_SEC": 8.0,
    "DESCRIPTION": "violent shot",
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

    categories, normalized_output = parse_interval_method_categories(text)
    assert categories == ["C3"]
    assert '"START_SEC": 4.0' in normalized_output


def test_compute_per_category_accuracy_uses_binary_accuracy() -> None:
    results = [
        {"gt_categories": ["C1"], "pred_categories": ["C1"]},
        {"gt_categories": [], "pred_categories": []},
        {"gt_categories": ["C2"], "pred_categories": []},
        {"gt_categories": [], "pred_categories": ["C3"]},
    ]
    metrics = compute_per_category_accuracy(results)
    assert metrics["num_samples"] == 4
    assert metrics["per_category"]["C1"]["accuracy"] == 1.0
    assert metrics["per_category"]["C2"]["fn"] == 1
    assert metrics["per_category"]["C3"]["fp"] == 1
