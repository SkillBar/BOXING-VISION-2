from __future__ import annotations

from copy import deepcopy

import pytest

from boxing_vision.quality import apply_result_gate, result_eligibility
from boxing_vision.ui_presenters import build_presentation_payload


def qualified_summary() -> dict:
    return {
        "quality": {"identity_verified_coverage": .97, "identity_swap_suspected": False,
                    "required_review_count": 0},
        "winner_id": "fighter_a", "winner_name": "Анна", "confidence": .82,
        "winner": {"fighter_id": "fighter_a", "name": "Анна"},
        "score_total": {"fighter_a": 30, "fighter_b": 27},
        "round_scores": [{"round": 1, "fighter_a": 10, "fighter_b": 9}],
        "fighters": {"fighter_a": {"name": "Анна", "attempts": 20}},
    }


@pytest.mark.parametrize("coverage", [.9, .97, 1.0])
def test_result_gate_accepts_only_verified_coverage_interval(coverage) -> None:
    summary = qualified_summary()
    summary["quality"]["identity_verified_coverage"] = coverage
    assert result_eligibility(summary) == (True, [])
    gated = apply_result_gate(summary)
    assert gated["winner_id"] == "fighter_a"
    assert gated["round_scores"] == summary["round_scores"]


@pytest.mark.parametrize("coverage", [
    None, "not a number", "NaN", float("nan"), float("inf"), float("-inf"),
    -.1, .89999, 1.00001, 10, True, False, {}, [],
])
def test_result_gate_rejects_missing_malformed_or_out_of_range_coverage(coverage) -> None:
    summary = qualified_summary()
    summary["quality"]["identity_verified_coverage"] = coverage
    allowed, reasons = result_eligibility(summary)
    assert not allowed and reasons


@pytest.mark.parametrize("quality", [None, [], "verified", {}, {"required_review_count": 0}])
def test_result_gate_fails_closed_without_identity_evidence(quality) -> None:
    summary = qualified_summary()
    summary["quality"] = quality
    assert result_eligibility(summary)[0] is False


@pytest.mark.parametrize("pending", [1, -1, "0", "1", None, 0.0, float("nan"), True, False])
def test_required_review_count_must_be_an_explicit_zero_integer(pending) -> None:
    summary = qualified_summary()
    summary["quality"]["required_review_count"] = pending
    allowed, reasons = result_eligibility(summary)
    assert not allowed
    assert "Остались обязательные проверки" in reasons


@pytest.mark.parametrize("flag,value", [("identity_swap_suspected", True), ("winner_visible", False)])
def test_explicit_identity_or_result_block_is_not_overridden(flag, value) -> None:
    summary = qualified_summary()
    summary["quality"][flag] = value
    assert result_eligibility(summary)[0] is False


def test_blocked_result_clears_every_score_path_without_mutating_source() -> None:
    summary = qualified_summary()
    summary["quality"]["required_review_count"] = 2
    original = deepcopy(summary)
    gated = apply_result_gate(summary)
    assert summary == original
    assert gated["winner_id"] is None and gated["winner_name"] is None
    assert gated["confidence"] == 0
    assert gated["winner"] == gated["score_total"] == {}
    assert gated["round_scores"] == []
    assert gated["fighters"] == original["fighters"]
    assert gated["quality"]["winner_visible"] is False
    assert apply_result_gate(gated) == gated


def test_presentation_does_not_leak_blocked_winner_or_round_points() -> None:
    summary = qualified_summary()
    summary["quality"]["identity_verified_coverage"] = .7
    payload = build_presentation_payload([], summary, 30)
    assert payload["result"]["winner"] == {}
    assert payload["result"]["score_total"] == {}
    assert payload["rounds"] == []
    assert payload["fighters"]["fighter_a"]["name"] == "Анна"
