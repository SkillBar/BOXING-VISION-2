from __future__ import annotations

from boxing_vision.contracts import PunchEvent
from boxing_vision.scoring import build_fight_summary, score_round, score_rounds


def _event(
    event_id: str,
    attacker: str,
    outcome: str,
    *,
    round_number: int = 1,
    confidence: float = 0.9,
    impact: int = 60,
    replay: bool = False,
) -> PunchEvent:
    defender = "fighter_b" if attacker == "fighter_a" else "fighter_a"
    number = int(
        "".join(character for character in event_id if character.isdigit()) or 1
    )
    return PunchEvent(
        event_id=event_id,
        round=round_number,
        start_ms=number * 500,
        peak_ms=number * 500 + 100,
        end_ms=number * 500 + 250,
        attacker_id=attacker,
        defender_id=defender,
        hand="left",
        technique="jab",
        target="head",
        outcome=outcome,
        confidence=confidence,
        impact_proxy_0_100=impact,
        is_replay=replay,
    )


def test_70_15_10_5_index_prefers_effective_punches() -> None:
    events = [
        _event("a1", "fighter_a", "likely_landed", impact=80),
        _event("a2", "fighter_a", "likely_landed", impact=70),
        _event("a3", "fighter_a", "missed", impact=20),
        _event("b4", "fighter_b", "likely_landed", impact=35),
        _event("b5", "fighter_b", "missed", impact=20),
        _event("b6", "fighter_b", "missed", impact=20),
    ]

    score = score_round(1, events)

    assert score.fighter_a_points == 10
    assert score.fighter_b_points == 9
    assert score.fighter_a_index > score.fighter_b_index
    assert score.fighter_a_index + score.fighter_b_index == 100.0


def test_unclear_and_replay_events_do_not_affect_score() -> None:
    baseline = [_event("a1", "fighter_a", "likely_landed")]
    ignored = [
        _event("b2", "fighter_b", "unclear", confidence=1.0, impact=100),
        _event(
            "b3", "fighter_b", "likely_landed", confidence=1.0, impact=100, replay=True
        ),
    ]

    assert score_round(1, baseline) == score_round(1, baseline + ignored)


def test_confirmed_knockdown_allows_10_8() -> None:
    events = [
        _event("a1", "fighter_a", "likely_landed"),
        _event("a2", "fighter_a", "likely_landed"),
        _event("b3", "fighter_b", "missed"),
    ]

    score = score_round(
        1,
        events,
        confirmed_knockdowns_suffered={"fighter_b": 1},
    )

    assert (score.fighter_a_points, score.fighter_b_points) == (10, 8)
    assert "нокдаун" in score.reason


def test_confirmed_knockdown_overrides_leaders_dominance() -> None:
    events = [
        *[
            _event(f"a{index}", "fighter_a", "likely_landed", impact=90)
            for index in range(1, 11)
        ],
        _event("b20", "fighter_b", "likely_landed", impact=40),
        _event("b21", "fighter_b", "likely_landed", impact=40),
    ]

    score = score_round(
        1,
        events,
        confirmed_knockdowns_suffered={"fighter_a": 1},
    )

    assert (score.fighter_a_points, score.fighter_b_points) == (8, 10)
    assert "нокдаун" in score.reason


def test_summary_is_json_ready_and_uses_neutral_model_language() -> None:
    events = [
        _event("a1", "fighter_a", "likely_landed", round_number=1),
        _event("b2", "fighter_b", "missed", round_number=1),
        _event("b3", "fighter_b", "likely_landed", round_number=1, replay=True),
    ]

    summary = build_fight_summary(
        events,
        fighter_names={"fighter_a": "Красный", "fighter_b": "Синий"},
        scheduled_rounds=1,
    )

    assert summary["winner_id"] == "fighter_a"
    assert summary["winner_name"] == "Красный"
    assert summary["fighters"]["fighter_b"]["attempts"] == 1
    assert summary["disclaimer"] == "Оценка модели с указанием уверенности."
    assert "эксперимент" not in summary["disclaimer"].lower()
    assert summary["weights"] == {
        "effective_landed": 0.70,
        "accuracy": 0.15,
        "activity": 0.10,
        "pressure_proxy": 0.05,
    }
    assert score_rounds(events, scheduled_rounds=1)[0].round == 1


def test_summary_exposes_applied_and_received_head_body_breakdowns() -> None:
    landed_head = _event("a1", "fighter_a", "likely_landed")
    blocked_body = _event("a2", "fighter_a", "blocked")
    blocked_body.target = "body"
    landed_unknown = _event("b3", "fighter_b", "likely_landed")
    landed_unknown.target = "unknown"
    ignored_replay = _event("b4", "fighter_b", "likely_landed", replay=True)
    ignored_replay.target = "body"
    ignored_rejected = _event("a5", "fighter_a", "likely_landed")
    ignored_rejected.target = "body"
    ignored_rejected.review_status = "rejected"

    summary = build_fight_summary(
        [
            landed_head,
            blocked_body,
            landed_unknown,
            ignored_replay,
            ignored_rejected,
        ],
        scheduled_rounds=1,
    )

    fighter_a = summary["fighters"]["fighter_a"]
    fighter_b = summary["fighters"]["fighter_b"]
    assert fighter_a["landed_targets"] == {
        "head": {"landed": 1, "thrown": 1},
        "body": {"landed": 0, "thrown": 1},
        "unknown": {"landed": 0, "thrown": 0},
    }
    assert fighter_a["received_landed_targets"] == {
        "head": {"landed": 0, "thrown": 0},
        "body": {"landed": 0, "thrown": 0},
        "unknown": {"landed": 1, "thrown": 1},
    }
    assert fighter_b["received_landed_targets"] == fighter_a["landed_targets"]
