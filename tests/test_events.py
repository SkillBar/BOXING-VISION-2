from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from boxing_vision.contracts import (
    BBox,
    IdentityState,
    Keypoint,
    PoseObservation,
    PunchEvent,
    ReviewStatus,
    SceneState,
)
from boxing_vision.events import (
    PunchDetectionConfig,
    _temporal_nms,
    annotate_exchanges,
    annotate_possible_knockdowns,
    detect_punch_events,
    retain_motion_proposals,
)
from boxing_vision.pose import RawPose, TwoFighterTracker
from boxing_vision.temporal import (
    TEMPORAL_FEATURE_NAMES,
    HysteresisConfig,
    TemporalFeature,
    TemporalProposal,
    feature_matrix,
)


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
    assert event.start_ms < event.evidence["velocity_peak_ms"] <= event.peak_ms < event.end_ms
    assert event.evidence["candidate_duration_ms"] == event.end_ms - event.start_ms
    assert event.evidence["return_ratio"] >= 0.30
    assert event.evidence["contact_timestamp_ms"] == 500.0
    assert event.evidence["contact_point_confidence"] > 0.75
    assert event.target_point_space == "detector_bbox_v1"
    assert event.target_point_source == "pose_projection_v1"


def test_unresolved_real_motion_is_retained_without_becoming_a_scored_event():
    from boxing_vision.tracking import TrackingFrame

    observations = [replace(item, identity_state=IdentityState.UNKNOWN,
                            source_track_id=1, identity_confidence=0.0)
                    for item in _straight_punch_sequence() if item.fighter_id == "fighter_a"]
    frames = [TrackingFrame(item.frame_index, item.timestamp_ms, 0,
                [RawPose(item.bbox, item.keypoints, confidence=.96, source_track_id=1,
                         detector_confidence=.96, pose_confidence=.96)]) for item in observations]
    assert detect_punch_events(observations) == []
    proposals = retain_motion_proposals(frames, [])
    assert proposals and proposals[0].fighter_id is None
    assert proposals[0].resolved_event_id is None
    assert proposals[0].status == "needs_review"
    assert proposals[0].hand == "right"
    for frame in frames:
        frame.scene_state = "BREAK"
    assert retain_motion_proposals(frames, []) == []


def test_continuous_session_does_not_stop_at_180_seconds():
    observations = [replace(item, timestamp_ms=item.timestamp_ms + 185000)
                    for item in _straight_punch_sequence()]
    assert detect_punch_events(observations, PunchDetectionConfig(scheduled_rounds=1)) == []
    events = detect_punch_events(observations, PunchDetectionConfig(
        timing_mode="continuous", scheduled_rounds=1))
    assert events and events[0].round == 1


def test_display_predictions_are_not_event_evidence():
    from boxing_vision.contracts import DisplayTrack
    displayed = [DisplayTrack(timestamp_ms=item.timestamp_ms, evidence_timestamp_ms=0,
                    bbox=item.bbox, keypoints=item.keypoints, fighter_id=item.fighter_id,
                    identity_state=item.identity_state, identity_confidence=.99,
                    display_state="PREDICTED") for item in _straight_punch_sequence()]
    assert detect_punch_events(displayed) == []


def test_confirmed_identity_without_measured_wrists_does_not_generate_punches():
    sequence = [replace(item, identity_confidence=.99, pose_confidence=.2,
        keypoints={name: point for name, point in item.keypoints.items() if "wrist" not in name})
        for item in _straight_punch_sequence()]
    assert detect_punch_events(sequence) == []


def test_missing_defender_geometry_uses_unknown_and_unclear() -> None:
    events = detect_punch_events(
        _straight_punch_sequence(obscure_defender=True),
        PunchDetectionConfig(event_confidence_threshold=0.30),
    )

    assert len(events) == 1
    assert events[0].target == "unknown"
    assert events[0].outcome == "unclear"
    assert events[0].technique == "straight"
    assert "contact_point_x" not in events[0].evidence


def test_extension_without_return_is_not_a_completed_punch() -> None:
    wrist_positions = [
        (160, 112),
        (165, 111),
        (190, 108),
        (225, 103),
        (260, 98),
        (285, 94),
        (300, 92),
        (301, 92),
        (300, 92),
    ]
    observations: list[PoseObservation] = []
    for timestamp, wrist in zip(range(0, 900, 100), wrist_positions):
        observations.append(_observation("fighter_a", timestamp, 135, punching_wrist=wrist))
        observations.append(_observation("fighter_b", timestamp, 305))

    assert (
        detect_punch_events(
            observations,
            PunchDetectionConfig(event_confidence_threshold=0.30),
        )
        == []
    )


def test_velocity_jitter_within_one_extension_return_emits_one_event() -> None:
    wrist_positions = [
        (160, 112),
        (166, 111),
        (205, 106),
        (255, 99),
        (240, 101),
        (282, 95),
        (305, 91),
        (260, 98),
        (205, 107),
        (165, 112),
    ]
    observations: list[PoseObservation] = []
    for timestamp, wrist in zip(range(0, 1_000, 100), wrist_positions):
        observations.append(_observation("fighter_a", timestamp, 135, punching_wrist=wrist))
        observations.append(_observation("fighter_b", timestamp, 305))

    events = detect_punch_events(
        observations,
        PunchDetectionConfig(event_confidence_threshold=0.30),
    )

    assert len(events) == 1
    assert events[0].hand == "right"
    assert events[0].evidence["temporal_confidence"] >= 0.5


def test_scene_cut_discards_an_incomplete_temporal_proposal() -> None:
    observations = _straight_punch_sequence()
    observations = [
        replace(observation, is_scene_cut=observation.timestamp_ms == 500)
        for observation in observations
    ]

    assert (
        detect_punch_events(
            observations,
            PunchDetectionConfig(event_confidence_threshold=0.30),
        )
        == []
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scene_state", SceneState.REPLAY),
        ("scene_state", SceneState.BREAK),
        ("identity_state", IdentityState.UNKNOWN),
        ("review_status", ReviewStatus.NEEDS_REVIEW),
    ],
)
def test_events_require_active_confirmed_pair(field: str, value: object) -> None:
    observations = _straight_punch_sequence()
    observations = [
        replace(observation, **{field: value})
        if observation.fighter_id == "fighter_a"
        else observation
        for observation in observations
    ]

    assert (
        detect_punch_events(
            observations,
            PunchDetectionConfig(event_confidence_threshold=0.30),
        )
        == []
    )


def test_fast_alternating_hands_survive_temporal_nms() -> None:
    right_positions = [
        (160, 112),
        (175, 110),
        (220, 105),
        (280, 96),
        (305, 91),
        (250, 100),
        (165, 112),
        (160, 112),
        (160, 112),
        (160, 112),
        (160, 112),
        (160, 112),
    ]
    left_positions = [
        (127, 112),
        (127, 112),
        (127, 112),
        (127, 112),
        (127, 112),
        (140, 110),
        (190, 105),
        (250, 97),
        (305, 91),
        (250, 100),
        (127, 112),
        (127, 112),
    ]
    observations: list[PoseObservation] = []
    for index, (right_wrist, left_wrist) in enumerate(
        zip(right_positions, left_positions)
    ):
        timestamp = index * 100
        attacker = _observation(
            "fighter_a",
            timestamp,
            135,
            punching_wrist=right_wrist,
        )
        keypoints = dict(attacker.keypoints)
        keypoints["left_wrist"] = Keypoint(
            float(left_wrist[0]),
            float(left_wrist[1]),
            0.96,
        )
        keypoints["left_elbow"] = Keypoint(
            (120 + left_wrist[0]) / 2,
            (120 + left_wrist[1]) / 2,
            0.96,
        )
        observations.append(replace(attacker, keypoints=keypoints))
        observations.append(_observation("fighter_b", timestamp, 305))

    events = detect_punch_events(
        observations,
        PunchDetectionConfig(event_confidence_threshold=0.30),
    )

    assert [(event.hand, event.peak_ms) for event in events] == [
        ("right", 400),
        ("left", 800),
    ]


def test_temporal_model_seam_has_stable_features_and_can_replace_proposals() -> None:
    features = [
        TemporalFeature(
            timestamp_ms=100,
            extension=0.4,
            speed=1.2,
            outward_speed=0.7,
            target_speed=0.6,
            elbow_angle=144.0,
            visibility=0.9,
        )
    ]
    matrix = feature_matrix(features)
    assert TEMPORAL_FEATURE_NAMES == (
        "extension",
        "wrist_x",
        "wrist_y",
        "speed",
        "outward_speed",
        "target_speed",
        "elbow_angle_normalized",
        "visibility",
    )
    assert matrix.shape == (1, 8)
    assert matrix[0].tolist() == pytest.approx(
        [0.4, 0.0, 0.0, 1.2, 0.7, 0.6, 0.8, 0.9]
    )

    class EmptyTemporalModel:
        calls = 0

        def propose(
            self,
            model_features: Sequence[TemporalFeature],
            config: HysteresisConfig,
        ) -> list[TemporalProposal]:
            assert model_features
            assert config.min_duration_ms > 0
            self.calls += 1
            return []

    model = EmptyTemporalModel()
    assert detect_punch_events(_straight_punch_sequence(), proposal_provider=model) == []
    assert model.calls == 4


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


def test_temporal_nms_preserves_different_hands_100ms_apart_but_removes_duplicates() -> None:
    first = _event("one", "fighter_a", "fighter_b", 1000)
    second = replace(first, event_id="two", hand="right", start_ms=900, peak_ms=1100, end_ms=1300)
    first.hand = "left"
    duplicate = replace(first, event_id="duplicate", confidence=.2)
    selected = _temporal_nms([first, second, duplicate], refractory_ms=260, interval_iou_threshold=.45)
    assert [value.event_id for value in selected] == ["one", "two"]


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


def test_two_fighter_tracker_requires_three_votes_after_reversed_camera_cut() -> None:
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

    for frame_index in range(3):
        tracker.process(
            frame_index,
            frame_index * 100,
            [raw(100, red), raw(300, blue), raw(205, referee)],
        )
    tracker.clear_anchors()
    first_after_cut = tracker.process(
        3,
        300,
        [raw(100, blue), raw(300, red), raw(205, referee)],
        scene_cut=True,
    )
    second_after_cut = tracker.process(
        4,
        400,
        [raw(100, blue), raw(300, red), raw(205, referee)],
    )
    after_cut = tracker.process(
        5,
        500,
        [raw(100, blue), raw(300, red), raw(205, referee)],
    )

    assert first_after_cut == []
    assert second_after_cut == []
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


def test_post_cut_single_visible_fighter_requires_three_profile_votes() -> None:
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

    for frame_index in range(3):
        tracker.process(
            frame_index,
            frame_index * 100,
            [raw(100, red), raw(300, blue)],
        )
    tracker.clear_anchors()
    first_after_cut = tracker.process(3, 300, [raw(110, blue)], scene_cut=True)
    second_after_cut = tracker.process(4, 400, [raw(110, blue)])
    after_cut = tracker.process(5, 500, [raw(110, blue)])

    assert first_after_cut == []
    assert second_after_cut == []
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
