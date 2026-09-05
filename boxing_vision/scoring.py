"""Transparent experimental round scoring for the investor prototype.

The score is not an official boxing verdict.  It is a deterministic view over
reviewable event candidates using the advertised 70/15/10/5 weighting.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean

from .contracts import PunchEvent, RoundScore

LANDED_OUTCOMES = {"likely_landed", "landed", "likely landed"}
BLOCKED_OUTCOMES = {"blocked", "block"}
MISSED_OUTCOMES = {"missed", "miss"}
UNCLEAR_OUTCOMES = {"unclear", "unknown", "cannot_determine"}


@dataclass(frozen=True, slots=True)
class FighterRoundMetrics:
    fighter_id: str
    effective: float
    accuracy: float
    activity: float
    pressure: float
    attempts: int
    likely_landed: int
    blocked: int
    missed: int
    average_confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "fighter_id": self.fighter_id,
            "effective": round(self.effective, 4),
            "accuracy": round(self.accuracy, 4),
            "activity": round(self.activity, 4),
            "pressure": round(self.pressure, 4),
            "attempts": self.attempts,
            "likely_landed": self.likely_landed,
            "blocked": self.blocked,
            "missed": self.missed,
            "average_confidence": round(self.average_confidence, 4),
        }


def _canonical_outcome(value: str) -> str:
    normalized = value.lower().strip().replace("-", "_")
    if normalized in LANDED_OUTCOMES:
        return "likely_landed"
    if normalized in BLOCKED_OUTCOMES:
        return "blocked"
    if normalized in MISSED_OUTCOMES:
        return "missed"
    return "unclear"


def _eligible_events(events: Iterable[PunchEvent]) -> list[PunchEvent]:
    result: list[PunchEvent] = []
    for event in events:
        if event.is_replay or event.review_status.lower() in {"rejected", "deleted"}:
            continue
        if _canonical_outcome(event.outcome) == "unclear":
            continue
        result.append(event)
    return result


def fighter_round_metrics(
    events: Iterable[PunchEvent],
    fighter_id: str,
) -> FighterRoundMetrics:
    """Calculate unnormalized features for one fighter in one round."""

    fighter_events = [event for event in _eligible_events(events) if event.attacker_id == fighter_id]
    if not fighter_events:
        return FighterRoundMetrics(fighter_id, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0)

    likely_landed = [event for event in fighter_events if _canonical_outcome(event.outcome) == "likely_landed"]
    blocked = [event for event in fighter_events if _canonical_outcome(event.outcome) == "blocked"]
    missed = [event for event in fighter_events if _canonical_outcome(event.outcome) == "missed"]
    # Effectiveness rewards confidence first and the deliberately non-physical
    # impact proxy second.  A low-impact likely hit still counts.
    effective = sum(
        event.confidence * (0.65 + 0.35 * event.impact_proxy_0_100 / 100.0)
        for event in likely_landed
    )
    weighted_attempts = sum(max(0.05, event.confidence) for event in fighter_events)
    weighted_landed = sum(event.confidence for event in likely_landed)
    accuracy = weighted_landed / weighted_attempts if weighted_attempts else 0.0
    activity = weighted_attempts

    # Pressure is a reproducible proxy: proactive (non-counter) attempts plus a
    # small reward for maintaining a combination.  It does not claim to infer
    # ring generalship from camera perspective.
    proactive = sum(event.confidence for event in fighter_events if not event.is_counter)
    combination_bonus = sum(
        0.15 * event.confidence for event in fighter_events if event.combo_id is not None
    )
    pressure = proactive + combination_bonus
    return FighterRoundMetrics(
        fighter_id=fighter_id,
        effective=effective,
        accuracy=accuracy,
        activity=activity,
        pressure=pressure,
        attempts=len(fighter_events),
        likely_landed=len(likely_landed),
        blocked=len(blocked),
        missed=len(missed),
        average_confidence=mean(event.confidence for event in fighter_events),
    )


def _share(first: float, second: float) -> tuple[float, float]:
    total = max(0.0, first) + max(0.0, second)
    if total <= 1e-9:
        return 0.5, 0.5
    return max(0.0, first) / total, max(0.0, second) / total


def _indices(
    fighter_a: FighterRoundMetrics,
    fighter_b: FighterRoundMetrics,
) -> tuple[float, float]:
    effective_a, effective_b = _share(fighter_a.effective, fighter_b.effective)
    accuracy_a, accuracy_b = _share(fighter_a.accuracy, fighter_b.accuracy)
    activity_a, activity_b = _share(fighter_a.activity, fighter_b.activity)
    pressure_a, pressure_b = _share(fighter_a.pressure, fighter_b.pressure)
    index_a = 100.0 * (
        0.70 * effective_a
        + 0.15 * accuracy_a
        + 0.10 * activity_a
        + 0.05 * pressure_a
    )
    index_b = 100.0 * (
        0.70 * effective_b
        + 0.15 * accuracy_b
        + 0.10 * activity_b
        + 0.05 * pressure_b
    )
    return index_a, index_b


def score_round(
    round_number: int,
    events: Iterable[PunchEvent],
    fighter_a_id: str = "fighter_a",
    fighter_b_id: str = "fighter_b",
    *,
    confirmed_knockdowns_suffered: Mapping[str, int] | None = None,
    allow_high_confidence_dominance: bool = True,
) -> RoundScore:
    """Return a transparent experimental 10-point-must score.

    ``confirmed_knockdowns_suffered`` is deliberately user-confirmed data: a
    value of one for fighter B means B was knocked down once in this round.
    """

    if round_number < 1:
        raise ValueError("Номер раунда должен быть положительным")
    round_events = [event for event in events if event.round == round_number]
    eligible = _eligible_events(round_events)
    metrics_a = fighter_round_metrics(eligible, fighter_a_id)
    metrics_b = fighter_round_metrics(eligible, fighter_b_id)
    index_a, index_b = _indices(metrics_a, metrics_b)
    relevant_confidences = [event.confidence for event in eligible]
    data_volume = min(1.0, len(eligible) / 12.0)
    confidence = (mean(relevant_confidences) * (0.55 + 0.45 * data_volume)) if relevant_confidences else 0.0

    difference = index_a - index_b
    if abs(difference) < 3.0:
        points_a, points_b = 10, 10
        reason = "Недостаточная разница индексов модели"
    elif difference > 0:
        points_a, points_b = 10, 9
        reason = "Преимущество по взвешенным эффективным попаданиям"
    else:
        points_a, points_b = 9, 10
        reason = "Преимущество по взвешенным эффективным попаданиям"

    knockdowns = dict(confirmed_knockdowns_suffered or {})
    a_suffered = max(0, int(knockdowns.get(fighter_a_id, 0)))
    b_suffered = max(0, int(knockdowns.get(fighter_b_id, 0)))
    if b_suffered > a_suffered:
        points_a = 10
        points_b = max(7, 9 - (b_suffered - a_suffered))
        reason = "Преимущество и подтверждённый пользователем нокдаун"
    elif a_suffered > b_suffered:
        points_b = 10
        points_a = max(7, 9 - (a_suffered - b_suffered))
        reason = "Преимущество и подтверждённый пользователем нокдаун"
    elif allow_high_confidence_dominance and confidence >= 0.78:
        if (
            index_a >= 78.0
            and metrics_a.likely_landed >= 8
            and metrics_a.likely_landed - metrics_b.likely_landed >= 6
        ):
            points_a, points_b = 10, 8
            reason = "Явное доминирование при высокой уверенности данных"
        elif (
            index_b >= 78.0
            and metrics_b.likely_landed >= 8
            and metrics_b.likely_landed - metrics_a.likely_landed >= 6
        ):
            points_a, points_b = 8, 10
            reason = "Явное доминирование при высокой уверенности данных"

    return RoundScore(
        round=round_number,
        fighter_a_points=points_a,
        fighter_b_points=points_b,
        fighter_a_index=round(index_a, 3),
        fighter_b_index=round(index_b, 3),
        confidence=round(confidence, 4),
        reason=reason,
    )


def score_rounds(
    events: Iterable[PunchEvent],
    fighter_a_id: str = "fighter_a",
    fighter_b_id: str = "fighter_b",
    *,
    scheduled_rounds: int | None = None,
    confirmed_knockdowns_suffered: Mapping[int, Mapping[str, int]] | None = None,
) -> list[RoundScore]:
    event_list = list(events)
    if scheduled_rounds is not None:
        if scheduled_rounds < 1:
            raise ValueError("Количество раундов должно быть положительным")
        round_numbers = list(range(1, scheduled_rounds + 1))
    else:
        round_numbers = sorted({event.round for event in event_list})
    confirmed = confirmed_knockdowns_suffered or {}
    return [
        score_round(
            round_number,
            event_list,
            fighter_a_id,
            fighter_b_id,
            confirmed_knockdowns_suffered=confirmed.get(round_number),
        )
        for round_number in round_numbers
    ]


def _event_counts(events: Sequence[PunchEvent], fighter_id: str) -> dict[str, object]:
    fighter_events = [
        event
        for event in events
        if event.attacker_id == fighter_id
        and not event.is_replay
        and event.review_status.lower() not in {"rejected", "deleted"}
    ]
    outcomes: dict[str, int] = defaultdict(int)
    techniques: dict[str, int] = defaultdict(int)
    hands: dict[str, int] = defaultdict(int)
    for event in fighter_events:
        outcomes[_canonical_outcome(event.outcome)] += 1
        techniques[event.technique] += 1
        hands[event.hand] += 1
    scored_attempts = sum(outcomes[outcome] for outcome in ("likely_landed", "blocked", "missed"))
    accuracy = outcomes["likely_landed"] / scored_attempts if scored_attempts else 0.0
    average_impact = mean(event.impact_proxy_0_100 for event in fighter_events) if fighter_events else 0.0
    return {
        "attempts": len(fighter_events),
        "likely_landed": outcomes["likely_landed"],
        "blocked": outcomes["blocked"],
        "missed": outcomes["missed"],
        "unclear": outcomes["unclear"],
        "accuracy": round(accuracy, 4),
        "average_impact_proxy": round(average_impact, 1),
        "techniques": dict(sorted(techniques.items())),
        "hands": dict(sorted(hands.items())),
        "landed_targets": _target_breakdown(events, fighter_id),
        "received_landed_targets": _target_breakdown(
            events,
            fighter_id,
            received=True,
        ),
    }


def _canonical_target(value: str) -> str:
    normalized = value.lower().strip().replace("-", "_")
    return normalized if normalized in {"head", "body"} else "unknown"


def _target_breakdown(
    events: Sequence[PunchEvent],
    fighter_id: str,
    *,
    received: bool = False,
) -> dict[str, dict[str, int]]:
    """Return stable head/body/unknown landed-over-thrown counters.

    ``received=False`` describes punches thrown by the fighter. ``received=True``
    describes incoming punches whose ``defender_id`` is the fighter. Replays and
    rejected/deleted events are excluded exactly as they are from headline totals.
    """

    identifier_field = "defender_id" if received else "attacker_id"
    selected = [
        event
        for event in events
        if getattr(event, identifier_field) == fighter_id
        and not event.is_replay
        and event.review_status.lower() not in {"rejected", "deleted"}
    ]
    breakdown = {
        target: {"landed": 0, "thrown": 0}
        for target in ("head", "body", "unknown")
    }
    for event in selected:
        target = _canonical_target(event.target)
        breakdown[target]["thrown"] += 1
        if _canonical_outcome(event.outcome) == "likely_landed":
            breakdown[target]["landed"] += 1
    return breakdown


def build_fight_summary(
    events: Iterable[PunchEvent],
    fighter_a_id: str = "fighter_a",
    fighter_b_id: str = "fighter_b",
    *,
    fighter_names: Mapping[str, str] | None = None,
    scheduled_rounds: int | None = None,
    confirmed_knockdowns_suffered: Mapping[int, Mapping[str, int]] | None = None,
) -> dict[str, object]:
    """Build JSON-ready totals, round cards and a confidence-labelled prediction."""

    event_list = sorted(events, key=lambda event: event.peak_ms)
    cards = score_rounds(
        event_list,
        fighter_a_id,
        fighter_b_id,
        scheduled_rounds=scheduled_rounds,
        confirmed_knockdowns_suffered=confirmed_knockdowns_suffered,
    )
    total_a = sum(card.fighter_a_points for card in cards)
    total_b = sum(card.fighter_b_points for card in cards)
    names = dict(fighter_names or {})
    if total_a > total_b:
        winner_id: str | None = fighter_a_id
    elif total_b > total_a:
        winner_id = fighter_b_id
    else:
        winner_id = None
    score_confidences = [card.confidence for card in cards if card.confidence > 0]
    return {
        "disclaimer": "Оценка модели с указанием уверенности.",
        "fighter_a_id": fighter_a_id,
        "fighter_b_id": fighter_b_id,
        "fighters": {
            fighter_a_id: {
                "name": names.get(fighter_a_id, fighter_a_id),
                **_event_counts(event_list, fighter_a_id),
            },
            fighter_b_id: {
                "name": names.get(fighter_b_id, fighter_b_id),
                **_event_counts(event_list, fighter_b_id),
            },
        },
        "round_scores": [card.to_dict() for card in cards],
        "score_total": {fighter_a_id: total_a, fighter_b_id: total_b},
        "winner_id": winner_id,
        "winner_name": names.get(winner_id, winner_id) if winner_id else "draw",
        "confidence": round(mean(score_confidences), 4) if score_confidences else 0.0,
        "weights": {
            "effective_landed": 0.70,
            "accuracy": 0.15,
            "activity": 0.10,
            "pressure_proxy": 0.05,
        },
    }


# Compatibility aliases for concise pipeline code.
calculate_round_scores = score_rounds
summarize_fight = build_fight_summary
