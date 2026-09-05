from dataclasses import replace

from boxing_vision.contracts import BBox, PoseObservation
from boxing_vision.preflight import build_preflight_report, preflight_failure_message


def fixtures():
    frames = [{"timestamp_ms": i * 100, "shot_id": 0, "scene_state": "ACTIVE_FIGHT"} for i in range(100)]
    observations = [PoseObservation(
        i, i * 100, role, BBox(0, 0, 100, 200), {},
        identity_confidence=.9, identity_margin=.2,
    ) for i in range(100) for role in ("fighter_a", "fighter_b")]
    return observations, frames


def test_report_is_advisory_without_inflating_60_percent():
    observations, frames = fixtures()
    observations = [o for o in observations if o.fighter_id == "fighter_a" or o.timestamp_ms >= 4000]
    report = build_preflight_report(observations, frames, [], window_ms=10000)
    assert report["pair_coverage"] == .6
    assert report["fighter_coverage"] == {"fighter_a": 1, "fighter_b": .6}
    assert report["status"] == "needs_review"
    assert report["required_pair_coverage"] == .7
    assert report["unconfirmed_intervals"]["fighter_b"] == [{"start_ms": 0, "end_ms": 4000}]
    message = preflight_failure_message(report)
    assert "B — в 60%" in message and "Анализ продолжается" in message
    assert "для запуска" not in message and "нужно 70%" not in message
    assert report["blocking"] is False
    assert report["policy"] == "continue_with_inactive_intervals"
    assert "0.0–4.0 с" in message


def test_denominator_does_not_hide_uncertain_or_missing_frames():
    observations, frames = fixtures()
    frames = [{**row, "scheduled_scene_state": "ACTIVE_FIGHT",
               "scene_state": "UNCERTAIN" if row['timestamp_ms'] >= 5000 else 'ACTIVE_FIGHT'} for row in frames]
    report = build_preflight_report(observations[:100], frames, [], window_ms=10000)
    assert report["active_frames"] == 100
    assert report["pair_coverage"] == .5


def test_duplicates_future_and_unexpected_observations_cannot_inflate_coverage():
    observations, frames = fixtures()
    unexpected = [replace(o, timestamp_ms=o.timestamp_ms + 1) for o in observations]
    report = build_preflight_report(observations[:100] * 3 + unexpected, frames, [], window_ms=10000)
    assert report["confirmed_pair_frames"] == 50
    assert report["pair_coverage"] == .5


def test_low_confidence_and_nonfight_are_not_confirmed():
    observations, frames = fixtures()
    for field in ({"identity_confidence": .54}, {"scene_state": "BREAK"}, {"identity_margin": .11}):
        report = build_preflight_report([replace(o, **field) for o in observations], frames, [], window_ms=10000)
        assert report["pair_coverage"] == 0


def test_no_active_frames_has_specific_guidance():
    observations, frames = fixtures()
    report = build_preflight_report(observations, [{**row, "scene_state": "BREAK"} for row in frames], [], window_ms=10000)
    assert report["reason"] == "no_active_frames"
    assert "границы раундов" in preflight_failure_message(report)


def test_stale_active_observations_cannot_override_frame_replay():
    observations, frames = fixtures()
    frames = [{**row, "scheduled_scene_state": "ACTIVE_FIGHT", "scene_state": "REPLAY"} for row in frames]
    report = build_preflight_report(observations, frames, [], window_ms=10000)
    assert report["active_frames"] == 100
    assert report["pair_coverage"] == 0
    assert report["reason"] == "replay_in_prefix"
    assert "отмечена как повтор" in preflight_failure_message(report)


def test_ring_guidance_requires_near_gallery_not_just_outside_spectators():
    _, frames = fixtures()
    row = {"timestamp_ms": 0, "reason": "outside_ring", "a_distance": .8, "b_distance": .8}
    report = build_preflight_report([], frames, [row], window_ms=10000)
    assert report["reason"] == "insufficient_confirmed_pair"
    report = build_preflight_report([], frames, [{**row, "a_distance": .1}], window_ms=10000)
    assert report["reason"] == "ring_excludes_enrolled_candidates"
    assert "полосу канатов" in preflight_failure_message(report)


def test_loss_intervals_split_at_cut_and_break():
    _, frames = fixtures()
    frames[30] = {**frames[30], "is_scene_cut": True}
    frames[50] = {**frames[50], "scene_state": "BREAK"}
    report = build_preflight_report([], frames, [], window_ms=10000)
    assert report["unconfirmed_intervals"]["fighter_a"] == [
        {"start_ms": 0, "end_ms": 3000},
        {"start_ms": 3000, "end_ms": 5000},
        {"start_ms": 5100, "end_ms": 10000},
    ]


def test_short_clip_not_mislabelled_as_ten_seconds():
    observations, frames = fixtures()
    report = build_preflight_report(observations[:2], frames, [], window_ms=1000)
    assert "участка 0–1 с" in preflight_failure_message(report)
