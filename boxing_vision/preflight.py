"""Advisory prefix diagnostics; uncertain identities remain inactive, not fatal."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import pairwise
from statistics import median

from .contracts import PoseObservation
from .identity import is_confirmed_identity

REQUIRED_PAIR_COVERAGE = 0.70
FIGHTERS = ("fighter_a", "fighter_b")


def build_preflight_report(
    observations: Sequence[PoseObservation],
    frame_states: Sequence[dict],
    diagnostics: Sequence[dict],
    *,
    window_ms: int,
    minimum_margin: float = 0.12,
) -> dict:
    """Coverage is over *scheduled* active frames, never surviving detections.

    These are internal confirmation rates, not measured identity accuracy.
    Candidate rejection counts include OTHER people; they are not fighter counts.
    """
    expected = {
        int(row["timestamp_ms"]): row
        for row in frame_states
        if 0 <= int(row["timestamp_ms"]) < window_ms
    }
    active = {
        stamp for stamp, row in expected.items()
        if row.get("scheduled_scene_state", row["scene_state"]) == "ACTIVE_FIGHT"
    }
    roles: dict[int, set[str]] = defaultdict(set)
    for observation in observations:
        if (observation.timestamp_ms in active
            and expected[observation.timestamp_ms]["scene_state"] == "ACTIVE_FIGHT"
            and is_confirmed_identity(
            observation, minimum_margin=minimum_margin
        )):
            roles[observation.timestamp_ms].add(observation.fighter_id)
    pairs = sum(set(FIGHTERS) <= ids for ids in roles.values())
    denominator = len(active)
    coverage = pairs / max(1, denominator)
    counts = {
        role: sum(role in ids for ids in roles.values()) for role in FIGHTERS
    }
    relevant = [row for row in diagnostics if row["timestamp_ms"] in active]
    rejected = Counter(
        str(row.get("reason", "unknown")) for row in relevant
        if row.get("selected_fighter_id") is None
    )
    # Only gallery-near candidates suggest an ROI error. Hundreds of audience
    # detections outside the ring must not obscure a fighter's actual problem.
    ring_candidates = sum(
        row.get("reason") == "outside_ring"
        and min(float(row.get("a_distance", 1)), float(row.get("b_distance", 1))) <= .35
        for row in relevant
    )
    passed = bool(denominator and coverage >= REQUIRED_PAIR_COVERAGE)
    reason = None if passed else "insufficient_confirmed_pair"
    if not denominator:
        reason = "no_active_frames"
    elif not any(counts.values()) and ring_candidates:
        reason = "ring_excludes_enrolled_candidates"
    elif not relevant and not any(counts.values()):
        reason = "no_candidate_evidence"
    scene_counts = Counter(str(expected[stamp]["scene_state"]) for stamp in active)
    if not passed and denominator and denominator - scene_counts["REPLAY"] < REQUIRED_PAIR_COVERAGE * denominator:
        reason = "replay_in_prefix"
    times = sorted(expected)
    step = round(median(b - a for a, b in pairwise(times))) if len(times) > 1 else window_ms
    intervals: dict[str, list[dict]] = {}
    for role in FIGHTERS:
        intervals[role] = []
        pending = None
        previous = None
        for stamp in times:
            missing = stamp in active and role not in roles[stamp]
            boundary = previous is not None and (
                stamp - previous > 1.5 * step
                or expected[stamp].get("shot_id") != expected[previous].get("shot_id")
                or expected[stamp].get("is_scene_cut", False)
            )
            if pending is not None and (not missing or boundary):
                intervals[role].append({"start_ms": pending, "end_ms": min(window_ms, previous + step)})
                pending = None
            if missing and pending is None:
                pending = stamp
            previous = stamp
        if pending is not None:
            intervals[role].append({"start_ms": pending, "end_ms": min(window_ms, times[-1] + step)})
    return {
        "version": 3,
        "status": "passed" if passed else "needs_review",
        "blocking": False,
        "policy": "continue_with_inactive_intervals",
        "window_ms": window_ms,
        "reason": reason,
        "required_pair_coverage": REQUIRED_PAIR_COVERAGE,
        "active_frames": denominator,
        "scheduled_active_scene_counts": dict(scene_counts),
        "confirmed_pair_frames": pairs,
        "pair_coverage": coverage,
        "fighter_coverage": {role: count / max(1, denominator) for role, count in counts.items()},
        "confirmed_fighter_frames": counts,
        "unconfirmed_intervals": intervals,
        "candidate_rejection_counts": dict(rejected),
        "diagnostics_file": "preflight_diagnostics.json",
    }


def preflight_advisory_message(report: dict) -> str:
    rates = report["fighter_coverage"]
    message = (
        f"Проверка участка 0–{report['window_ms'] / 1000:g} с: "
        f"A подтверждён в {rates['fighter_a']:.0%} кадров, B — в {rates['fighter_b']:.0%}. "
        f"Вместе — {report['pair_coverage']:.0%}. Анализ продолжается. "
    )
    if report["reason"] == "no_active_frames":
        return message + "В этом интервале нет активного раунда; удары не считаются. При необходимости проверьте границы раундов."
    if report["reason"] == "replay_in_prefix":
        return message + "Значительная часть участка отмечена как повтор и исключена из подсчёта ударов."
    if report["reason"] == "ring_excludes_enrolled_candidates":
        return message + "Размеченная зона исключает похожих на бойцов людей. Если нужно её исправить, отметьте пол вокруг стоп, а не полосу канатов."
    if report["reason"] == "no_candidate_evidence":
        return message + "Пока нет пригодных обнаружений людей; система продолжает поиск."
    weakest = min(FIGHTERS, key=rates.get)
    intervals = report["unconfirmed_intervals"][weakest]
    if intervals:
        longest = max(intervals, key=lambda row: row["end_ms"] - row["start_ms"])
        message += (
            f"Самый длинный участок без подтверждения {'A' if weakest == 'fighter_a' else 'B'}: "
            f"{longest['start_ms'] / 1000:.1f}–{longest['end_ms'] / 1000:.1f} с. "
        )
    return message + "Без подтверждения боец неактивен для подсчёта ударов. После восстановления личности подсчёт возобновится."


def preflight_failure_message(report: dict) -> str:
    """Compatibility alias for callers that used the former blocking message."""
    return preflight_advisory_message(report)
