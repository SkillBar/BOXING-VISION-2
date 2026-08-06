from __future__ import annotations

from dataclasses import replace

from boxing_vision.contracts import BBox, Keypoint, PoseObservation, PunchEvent
from boxing_vision.events import (
    PunchDetectionConfig,
    annotate_exchanges,
    annotate_possible_knockdowns,
    detect_punch_events,
)
from boxing_vision.pose import RawPose, TwoFighterTracker


def _keypoints(center_x: float, *, punching_wrist: tuple[float, float] | None = None) -> dict[str, Keypoint]:
    right_wrist = punching_wrist or (center_x + 10, 112)
    right_elbow = ((center_x + right_wrist[0]) / 2, (120 + right_wrist[1]) / 2)
    coordinates = {
        "nose": (center_x, 90),
        "left_eye": (center_x - 4, 87),
        "right_eye": (center_x + 4, 87),
        "left_ear": (center_x - 8, 91),
        "right_ear": (center_x + 8, 91),
        "left_shoulder": (center_x - 15, 120),
        "right_shoulder": (center_x + 15, 120),
        "left_elbow": (center_x - 23, 145),
        "right_elbow": right_elbow,
        "left_wrist": (center_x - 8, 112),
        "right_wrist": right_wrist,
        "left_hip": (center_x - 13, 190),
        "right_hip": (center_x + 13, 190),
        "left_knee": (center_x - 12, 245),
        "right_knee": (center_x + 12, 245),
        "left_ankle": (center_x - 12, 295),
        "right_ankle": (center_x + 12, 295),
    }
    return {name: Keypoint(float(x), float(y), 0.96) for name, (x, y) in coordinates.items()}


def _observation(
    fighter_id: str,
    timestamp_ms: int,
    center_x: float,
    *,
    punching_wrist: tuple[float, float] | None = None,
    scene_cut: bool = False,
) -> PoseObservation:
    return PoseObservation(
        frame_index=timestamp_ms // 100,
        timestamp_ms=timestamp_ms,
        fighter_id=fighter_id,
        bbox=BBox(center_x - 50, 60, center_x + 50, 310, 0.96),
        keypoints=_keypoints(center_x, punching_wrist=punching_wrist),
        track_confidence=0.96,
        is_scene_cut=scene_cut,
    )


def _straight_punch_sequence(*, obscure_defender: bool = False) -> list[PoseObservation]:
    times = list(range(0, 900, 100))
    wrist_positions = [
        (160, 112),
        (162, 111),
        (175, 108),
        (210, 102),
        (270, 95),
        (305, 91),
        (260, 98),
        (210, 106),
        (165, 112),
    ]
    observations: list[PoseObservation] = []
    for timestamp, wrist in zip(times, wrist_positions):
        observations.append(_observation("fighter_a", timestamp, 135, punching_wrist=wrist))
        defender = _observation("fighter_b", timestamp, 305)
        if obscure_defender:
            defender = replace(
                defender,
                keypoints={name: replace(point, score=0.05) for name, point in defender.keypoints.items()},
            )
        observations.append(defender)
    return observations


def test_detects_confident_straight_head_candidate() -> None:
    config = PunchDetectionConfig(event_confidence_threshold=0.35)
    events = detect_punch_events(
        _straight_punch_sequence(),
        config,
        stances={"fighter_a": "orthodox", "fighter_b": "southpaw"},
    )

    assert len(events) == 1
    event = events[0]
    assert event.attacker_id == "fighter_a"
    assert event.defender_id == "fighter_b"
    assert event.hand == "right"
    assert event.technique == "cross"
    assert event.target == "head"
    assert event.outcome == "likely_landed"
    assert event.confidence >= 0.65
    assert 1 <= event.impact_proxy_0_100 <= 100
    assert event.event_id == "evt_00001"


def test_missing_defender_geometry_uses_unknown_and_unclear() -> None:
    events = detect_punch_events(
        _straight_punch_sequence(obscure_defender=True),
        PunchDetectionConfig(event_confidence_threshold=0.30),
    )

    assert len(events) == 1
    assert events[0].target == "unknown"
    assert events[0].outcome == "unclear"
    assert events[0].technique == "straight"


def _event(event_id: str, attacker: str, defender: str, peak_ms: int) -> PunchEvent:
    return PunchEvent(
        event_id=event_id,
        round=1,
        start_ms=peak_ms - 100,
        peak_ms=peak_ms,
        end_ms=peak_ms + 150,
        attacker_id=attacker,
        defender_id=defender,
        hand="left",
        technique="jab",
        target="head",
        outcome="likely_landed",
        confidence=0.8,
        impact_proxy_0_100=55,
    )


def test_combo_and_counter_annotations_are_deterministic() -> None:
    first = _event("a1", "fighter_a", "fighter_b", 1_000)
    counter = _event("b1", "fighter_b", "fighter_a", 1_600)
    follow_up = _event("b2", "fighter_b", "fighter_a", 2_050)

    annotated = annotate_exchanges([follow_up, first, counter])

    assert [event.event_id for event in annotated] == ["a1", "b1", "b2"]
    assert counter.is_counter is True
    assert counter.combo_id is not None
    assert follow_up.combo_id == counter.combo_id


def test_two_fighter_tracker_ignores_large_referee_and_keeps_crossing_ids() -> None:
    tracker = TwoFighterTracker(anchors={"fighter_a": (100, 150), "fighter_b": (300, 150)})

    def raw(center: float, area_scale: float = 1.0) -> RawPose:
        half_width = 40 * area_scale
        return RawPose(
            BBox(center - half_width, 60, center + half_width, 300, 0.95),
            _keypoints(center),
            0.95,
        )

    first = tracker.process(0, 0, [raw(100), raw(300), raw(205, 1.3)])
    second = tracker.process(1, 100, [raw(125), raw(275), raw(205, 1.3)])

    assert [observation.fighter_id for observation in first] == ["fighter_a", "fighter_b"]
    assert first[0].bbox.center[0] == 100
    assert first[1].bbox.center[0] == 300
    assert second[0].bbox.center[0] == 125
    assert second[1].bbox.center[0] == 275


def test_two_fighter_tracker_uses_appearance_after_reversed_camera_cut() -> None:
    tracker = TwoFighterTracker(anchors={"fighter_a": (100, 150), "fighter_b": (300, 150)})
    red = (1.0, 0.0, 0.0)
    blue = (0.0, 1.0, 0.0)
    referee = (0.0, 0.0, 1.0)

    def raw(center: float, appearance: tuple[float, ...]) -> RawPose:
        return RawPose(
            BBox(center - 40, 60, center + 40, 300, 0.95),
            _keypoints(center),
            0.95,
            appearance,
        )

    tracker.process(0, 0, [raw(100, red), raw(300, blue), raw(205, referee)])
    tracker.clear_anchors()
    after_cut = tracker.process(
        1,
        100,
        [raw(100, blue), raw(300, red), raw(205, referee)],
        scene_cut=True,
    )

    by_id = {observation.fighter_id: observation for observation in after_cut}
    assert by_id["fighter_a"].bbox.center[0] == 300
    assert by_id["fighter_b"].bbox.center[0] == 100


def test_tracker_keeps_anchors_until_both_fighters_are_visible() -> None:
    tracker = TwoFighterTracker(anchors={"fighter_a": (100, 150), "fighter_b": (300, 150)})

    def raw(center: float, scale: float = 1.0) -> RawPose:
        return RawPose(
            BBox(center - 40 * scale, 60, center + 40 * scale, 300, 0.95),
            _keypoints(center),
            0.95,
        )

    only_blue = tracker.process(0, 0, [raw(295)])
    both = tracker.process(1, 100, [raw(100), raw(300), raw(205, 1.35)])

    assert [observation.fighter_id for observation in only_blue] == ["fighter_b"]
    by_id = {observation.fighter_id: observation for observation in both}
    assert by_id["fighter_a"].bbox.center[0] == 100
    assert by_id["fighter_b"].bbox.center[0] == 300


def test_post_cut_single_visible_fighter_uses_appearance_profile() -> None:
    tracker = TwoFighterTracker(anchors={"fighter_a": (100, 150), "fighter_b": (300, 150)})
    red = (1.0, 0.0, 0.0)
    blue = (0.0, 1.0, 0.0)

    def raw(center: float, appearance: tuple[float, ...]) -> RawPose:
        return RawPose(
            BBox(center - 40, 60, center + 40, 300, 0.95),
            _keypoints(center),
            0.95,
            appearance,
        )

    tracker.process(0, 0, [raw(100, red), raw(300, blue)])
    tracker.clear_anchors()
    after_cut = tracker.process(1, 100, [raw(110, blue)], scene_cut=True)

    assert [observation.fighter_id for observation in after_cut] == ["fighter_b"]


def test_possible_knockdown_requires_upright_to_horizontal_reaction() -> None:
    event = _event("a1", "fighter_a", "fighter_b", 1_000)
    observations = [
        replace(_observation("fighter_b", 900, 300), bbox=BBox(250, 60, 350, 310, 0.9)),
        replace(_observation("fighter_b", 1_400, 300), bbox=BBox(190, 230, 410, 330, 0.9)),
        replace(_observation("fighter_b", 1_600, 300), bbox=BBox(185, 235, 415, 335, 0.9)),
    ]

    annotated = annotate_possible_knockdowns([event], observations)

    assert annotated[0].review_status == "needs_review"
    assert annotated[0].evidence["possible_knockdown"] >= 0.5


def test_round_number_accounts_for_rest_and_suppresses_rest_events() -> None:
    shifted = [replace(observation, timestamp_ms=observation.timestamp_ms + 240_000) for observation in _straight_punch_sequence()]
    config = PunchDetectionConfig(
        event_confidence_threshold=0.35,
        round_length_s=180,
        rest_length_s=60,
    )
    round_two = detect_punch_events(shifted, config)
    assert len(round_two) == 1
    assert round_two[0].round == 2

    in_rest = [replace(observation, timestamp_ms=observation.timestamp_ms + 190_000) for observation in _straight_punch_sequence()]
    assert detect_punch_events(in_rest, config) == []

    after_scheduled_fight = [
        replace(observation, timestamp_ms=observation.timestamp_ms + 240_000)
        for observation in _straight_punch_sequence()
    ]
    one_round = replace(config, scheduled_rounds=1)
    assert detect_punch_events(after_scheduled_fight, one_round) == []
