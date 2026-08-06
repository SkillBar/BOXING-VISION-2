from __future__ import annotations

import numpy as np

from boxing_vision.contracts import (
    BBox,
    Keypoint,
    PoseObservation,
    PunchEvent,
    RoundScore,
)
from boxing_vision.render import FrameRenderer, render_frame


def _pose(
    fighter_id: str,
    bbox: BBox,
    *,
    offset_x: float = 0.0,
    scene_cut: bool = False,
) -> PoseObservation:
    x1, y1 = bbox.x1, bbox.y1
    width = bbox.width
    height = bbox.height

    def point(x_ratio: float, y_ratio: float, score: float = 0.95) -> Keypoint:
        return Keypoint(x1 + width * x_ratio + offset_x, y1 + height * y_ratio, score)

    return PoseObservation(
        frame_index=1,
        timestamp_ms=1000,
        fighter_id=fighter_id,
        bbox=bbox,
        keypoints={
            "nose": point(0.50, 0.12),
            "left_eye": point(0.46, 0.10),
            "right_eye": point(0.54, 0.10),
            "left_ear": point(0.41, 0.13),
            "right_ear": point(0.59, 0.13),
            "left_shoulder": point(0.35, 0.28),
            "right_shoulder": point(0.65, 0.28),
            "left_elbow": point(0.25, 0.42),
            "right_elbow": point(0.76, 0.38),
            "left_wrist": point(0.28, 0.54),
            "right_wrist": point(0.90, 0.29),
            "left_hip": point(0.40, 0.60),
            "right_hip": point(0.60, 0.60),
        },
        track_confidence=0.91,
        is_scene_cut=scene_cut,
    )


def _event() -> PunchEvent:
    return PunchEvent(
        event_id="event-001",
        round=1,
        start_ms=850,
        peak_ms=1000,
        end_ms=1150,
        attacker_id="fighter_a",
        defender_id="fighter_b",
        hand="right",
        technique="cross",
        target="head",
        outcome="likely_landed",
        confidence=0.84,
        impact_proxy_0_100=71,
    )


def test_renderer_draws_boxes_arms_zones_event_hud_and_score() -> None:
    frame = np.full((360, 640, 3), 12, dtype=np.uint8)
    original = frame.copy()
    observations = [
        _pose("fighter_a", BBox(45, 55, 235, 335)),
        _pose("fighter_b", BBox(270, 50, 450, 333)),
    ]
    summary = {
        "fighters": {
            "fighter_a": {
                "name": "Красный угол",
                "attempts": 18,
                "likely_landed": 8,
                "blocked": 3,
                "missed": 5,
                "unclear": 2,
                "impact_proxy_avg": 63,
            },
            "fighter_b": {
                "name": "Синий угол",
                "attempts": 14,
                "likely_landed": 5,
                "blocked": 4,
                "missed": 4,
                "unclear": 1,
                "impact_proxy_avg": 51,
            },
        }
    }
    scores = [RoundScore(1, 10, 9, 72.0, 61.0, 0.76, "effective punches")]
    renderer = FrameRenderer(
        {"fighter_a": "Красный угол", "fighter_b": "Синий угол"},
        trail_length=6,
    )

    rendered = renderer.draw(
        frame,
        observations,
        [_event()],
        summary,
        scores,
        timestamp_ms=1000,
    )

    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8
    assert np.array_equal(frame, original), "renderer must not mutate its input"
    assert np.count_nonzero(rendered != original) > 15_000
    assert not np.array_equal(rendered[55:62, 45:235], original[55:62, 45:235])
    assert not np.array_equal(rendered[:, 470:], original[:, 470:]), (
        "side HUD is missing"
    )
    assert not np.array_equal(rendered[6:40, 100:500], original[6:40, 100:500]), (
        "event banner is missing"
    )


def test_renderer_keeps_wrist_trail_and_resets_it_on_scene_cut() -> None:
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    renderer = FrameRenderer({"fighter_a": "A"}, trail_length=5)

    def wrist_pose(
        x: float, timestamp_ms: int, *, scene_cut: bool = False
    ) -> PoseObservation:
        return PoseObservation(
            frame_index=int(x),
            timestamp_ms=timestamp_ms,
            fighter_id="fighter_a",
            bbox=BBox(4, 4, 300, 225),
            keypoints={
                "left_wrist": Keypoint(x, 200, 0.99),
                "right_wrist": Keypoint(x, 210, 0.05),
            },
            is_scene_cut=scene_cut,
        )

    renderer.draw(frame, [wrist_pose(25, 1000)], timestamp_ms=1000)
    with_trail = renderer.draw(frame, [wrist_pose(105, 1033)], timestamp_ms=1033)
    after_cut = renderer.draw(
        frame, [wrist_pose(185, 1066, scene_cut=True)], timestamp_ms=1066
    )

    assert np.count_nonzero(with_trail[195:206, 45:90]) > 0
    assert np.count_nonzero(after_cut[195:206, 125:170]) == 0


def test_render_frame_accepts_mapping_event_and_filters_inactive_event() -> None:
    frame = np.zeros((180, 480, 3), dtype=np.uint8)
    observation = _pose("fighter_a", BBox(20, 30, 180, 170))
    event = _event().to_dict()

    active = render_frame(frame, [observation], event, timestamp_ms=1000)
    inactive = render_frame(frame, [observation], event, timestamp_ms=5000)

    # Both have tracking/HUD overlays; the active frame additionally has the event banner.
    assert np.count_nonzero(active[6:36, :350]) > np.count_nonzero(inactive[6:36, :350])


def test_renderer_rejects_invalid_frames() -> None:
    renderer = FrameRenderer()

    for invalid in (
        np.zeros((20, 20), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.float32),
        np.zeros((0, 20, 3), dtype=np.uint8),
    ):
        try:
            renderer.draw(invalid, [])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid frame must raise ValueError")
