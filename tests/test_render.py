from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from boxing_vision.contracts import (
    BBox,
    IdentityState,
    Keypoint,
    PoseObservation,
    PunchEvent,
    RenderFrameContext,
    ReviewStatus,
    RoundScore,
)
from boxing_vision.render import (
    FIGHTER_COLOR_BY_ID,
    FrameRenderer,
    _fit_text,
    _font,
    render_frame,
)


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


def test_compact_hud_preserves_action_area_and_none_disables_hud() -> None:
    frame = np.full((360, 640, 3), 17, dtype=np.uint8)
    summary = {
        "metadata": {
            "scheduled_rounds": 3,
            "round_length_s": 180,
            "rest_length_s": 60,
        }
    }

    compact = FrameRenderer(hud_mode="compact").draw(
        frame,
        [],
        [_event()],
        summary,
        timestamp_ms=1000,
    )
    without_hud = FrameRenderer(hud_mode="none").draw(
        frame,
        [],
        [_event()],
        summary,
        timestamp_ms=1000,
    )

    assert np.count_nonzero(compact[:80] != frame[:80]) > 1000
    assert np.array_equal(compact[100:], frame[100:]), (
        "compact HUD must not reserve or darken a right-side panel"
    )
    assert np.array_equal(without_hud, frame)


def test_compact_hud_attaches_separate_stat_cards_to_both_fighters() -> None:
    frame = np.full((360, 640, 3), 17, dtype=np.uint8)
    observations = [
        _pose("fighter_a", BBox(270, 90, 410, 330)),
        _pose("fighter_b", BBox(210, 80, 350, 332)),
    ]
    summary = {
        "fighters": {
            "fighter_a": {
                "name": "Красный",
                "attempts": 18,
                "likely_landed": 8,
                "accuracy": 0.44,
            },
            "fighter_b": {
                "name": "Синий",
                "attempts": 16,
                "likely_landed": 7,
                "accuracy": 0.43,
            },
        },
        "metadata": {"scheduled_rounds": 3, "round_length_s": 180, "rest_length_s": 60},
    }

    rendered = FrameRenderer(hud_mode="compact").draw(
        frame, observations, [_event()], summary, timestamp_ms=1000
    )

    assert np.count_nonzero(rendered[80:155, 40:255] != frame[80:155, 40:255]) > 1000
    assert np.count_nonzero(rendered[75:150, 350:630] != frame[75:150, 350:630]) > 1000


def test_compact_hud_card_slots_do_not_follow_moving_boxes() -> None:
    frame = np.full((360, 640, 3), 17, dtype=np.uint8)
    summary = {
        "fighters": {
            "fighter_a": {"name": "Красный", "attempts": 4, "likely_landed": 2},
            "fighter_b": {"name": "Синий", "attempts": 5, "likely_landed": 3},
        }
    }
    renderer = FrameRenderer(hud_mode="compact")
    first = renderer.draw(
        frame,
        [
            _pose("fighter_a", BBox(90, 100, 240, 330)),
            _pose("fighter_b", BBox(390, 90, 560, 332)),
        ],
        summary=summary,
        timestamp_ms=1000,
    )
    second = renderer.draw(
        frame,
        [
            _pose("fighter_a", BBox(310, 90, 460, 330)),
            _pose("fighter_b", BBox(170, 95, 340, 332)),
        ],
        summary=summary,
        timestamp_ms=1033,
    )

    # The fixed card border and accent bars stay pixel-identical while only
    # their leader endpoints and fighter boxes move.
    assert np.array_equal(first[50:160, 10:14], second[50:160, 10:14])
    assert np.array_equal(first[50:160, 626:630], second[50:160, 626:630])


def test_compact_hud_drops_held_event_outside_active_fight() -> None:
    frame = np.full((360, 640, 3), 17, dtype=np.uint8)
    observations = [
        _pose("fighter_a", BBox(90, 100, 240, 330)),
        _pose("fighter_b", BBox(390, 90, 560, 332)),
    ]
    summary = {
        "fighters": {
            "fighter_a": {"name": "Красный"},
            "fighter_b": {"name": "Синий"},
        }
    }
    renderer = FrameRenderer(hud_mode="compact")

    renderer.draw(
        frame,
        observations,
        [_event()],
        summary,
        frame_context=RenderFrameContext(timestamp_ms=1000),
    )
    assert renderer._held_events

    renderer.draw(
        frame,
        observations,
        [],
        summary,
        frame_context=RenderFrameContext(
            timestamp_ms=1200,
            scene_state="BREAK",
        ),
    )

    assert renderer._held_events == {}


def test_fighter_colors_are_fixed_when_b_is_drawn_before_a() -> None:
    frame = np.zeros((300, 640, 3), dtype=np.uint8)
    observations = [
        _pose("fighter_b", BBox(330, 70, 560, 280)),
        _pose("fighter_a", BBox(30, 70, 260, 280)),
    ]

    rendered = FrameRenderer(hud_mode="none").draw(frame, observations)

    for fighter_id in ("fighter_a", "fighter_b"):
        color = np.asarray(FIGHTER_COLOR_BY_ID[fighter_id], dtype=np.uint8)
        assert np.count_nonzero(np.all(rendered == color, axis=2)) > 100


class RecordingRenderer(FrameRenderer):
    def _apply_text(self, image, operations):
        self.text_operations = list(operations)
        return image


@pytest.mark.parametrize(
    "quality",
    [
        None,
        {"identity_verified_coverage": 0.89},
        {"identity_verified_coverage": 1.0, "identity_swap_suspected": True},
        {"identity_verified_coverage": 1.0, "required_review_count": 1},
    ],
)
def test_direct_renderer_round_score_argument_cannot_bypass_result_gate(quality):
    renderer = RecordingRenderer(hud_mode="technical")
    summary = {
        "winner_id": "fighter_a",
        "round_scores": [{"round": 1, "fighter_a_points": 10, "fighter_b_points": 8}],
    }
    if quality is not None:
        summary["quality"] = quality
    score = RoundScore(1, 10, 8, 80.0, 50.0, 0.9, "test")
    renderer.draw(
        np.zeros((720, 1280, 3), np.uint8), [], summary=summary, round_scores=[score]
    )
    assert not any(
        "10 — 8" in item[1] or "Уверенность счёта" in item[1]
        for item in renderer.text_operations
    )
    assert summary["winner_id"] == "fighter_a"


def test_direct_renderer_shows_round_points_only_after_quality_pass():
    renderer = RecordingRenderer(hud_mode="technical")
    summary = {
        "quality": {
            "identity_verified_coverage": 0.99,
            "required_review_count": 0,
            "identity_swap_suspected": False,
        }
    }
    score = RoundScore(1, 10, 8, 80.0, 50.0, 0.9, "test")
    renderer.draw(
        np.zeros((720, 1280, 3), np.uint8), [], summary=summary, round_scores=[score]
    )
    assert any("10 — 8" in item[1] for item in renderer.text_operations)


@pytest.mark.parametrize("scene", ["BREAK", "REPLAY", "NON_FIGHT", "UNCERTAIN"])
def test_frame_context_suppresses_stale_active_pose_overlays(scene):
    frame = np.zeros((360, 640, 3), np.uint8)
    poses = [
        _pose("fighter_a", BBox(40, 60, 220, 340)),
        _pose("fighter_b", BBox(360, 60, 550, 340)),
    ]
    renderer = FrameRenderer(hud_mode="none")
    renderer.draw(frame, poses, timestamp_ms=1000)
    output = renderer.draw(
        frame,
        poses,
        [_event()],
        frame_context=RenderFrameContext(1100, scene_state=scene),
    )
    assert np.array_equal(output, frame)
    assert not renderer._trails


def test_explicit_unknown_clears_old_connector_event_and_trails_immediately():
    frame = np.zeros((360, 640, 3), np.uint8)
    pose = _pose("fighter_a", BBox(220, 80, 360, 340))
    summary = {"fighters": {"fighter_a": {"name": "A"}}}
    renderer = FrameRenderer(hud_mode="compact")
    renderer.draw(frame, [pose], [_event()], summary, timestamp_ms=1000)
    assert renderer._leader_points
    unknown = replace(
        pose,
        identity_state=IdentityState.UNKNOWN,
        review_status=ReviewStatus.NEEDS_REVIEW,
    )
    renderer.draw(frame, [unknown], summary=summary, timestamp_ms=1100)
    assert not renderer._leader_points
    assert not renderer._held_events
    assert not renderer._trails


def test_context_shot_change_resets_without_cut_flag_and_rejects_prior_shot():
    frame = np.zeros((360, 640, 3), np.uint8)
    pose = _pose("fighter_a", BBox(40, 60, 220, 340))
    renderer = FrameRenderer(hud_mode="none")
    renderer.draw(frame, [pose], frame_context=RenderFrameContext(1000, shot_id=0))
    output = renderer.draw(
        frame, [pose], frame_context=RenderFrameContext(1100, shot_id=1)
    )
    assert np.array_equal(output, frame)


def test_bbox_ema_is_temporal_and_resets_on_source_change():
    renderer = FrameRenderer(hud_mode="compact")
    first = replace(_pose("fighter_a", BBox(100, 30, 200, 300)), source_track_id=1)
    moved = replace(
        first, bbox=BBox(120, 30, 220, 300), detector_bbox=BBox(120, 30, 220, 300)
    )
    renderer._smooth_observation(first, 1000)
    smooth = renderer._smooth_observation(moved, 1033)
    assert 100 < smooth.bbox.x1 < 120
    assert moved.bbox.x1 == 120
    changed = renderer._smooth_observation(replace(moved, source_track_id=2), 1066)
    assert changed.bbox.x1 == 120


def test_clinch_hides_connector_even_if_line_does_not_intersect(monkeypatch):
    renderer = FrameRenderer(hud_mode="compact")
    frame = np.zeros((360, 640, 3), np.uint8)
    a = _pose("fighter_a", BBox(260, 90, 410, 330))
    b = _pose("fighter_b", BBox(280, 90, 430, 330))
    monkeypatch.setattr(
        renderer, "_polyline_intersects_observation", lambda *args: False
    )
    calls = []
    monkeypatch.setattr(cv2, "polylines", lambda *args, **kwargs: calls.append(args))
    renderer._draw_fighter_card(frame, "fighter_a", a, b, [], {}, 1000, [])
    assert calls == []


def test_compact_cards_fixed_for_300_frames_and_leader_jitter_below_three_pixels():
    renderer = RecordingRenderer(hud_mode="compact")
    frame = np.zeros((360, 640, 3), np.uint8)
    summary = {"fighters": {"fighter_a": {"name": "A"}, "fighter_b": {"name": "B"}}}
    coordinates, anchors = [], []
    for index in range(300):
        timestamp = index * 33
        jitter = 6.0 if index % 2 else -6.0
        pose = replace(
            _pose("fighter_a", BBox(230, 90, 360, 330), offset_x=jitter),
            timestamp_ms=timestamp,
        )
        renderer.draw(frame, [pose], summary=summary, timestamp_ms=timestamp)
        coordinates.append(
            tuple(item[0] for item in renderer.text_operations if item[1] in {"A", "B"})
        )
        anchors.append(renderer._leader_points["fighter_a"])
    assert len(set(coordinates)) == 1
    stable = np.array(anchors[20:])
    assert float(np.sqrt(np.mean((stable - stable.mean(axis=0)) ** 2))) <= 3.0


def test_long_cyrillic_name_fits_fixed_card_width_without_small_font():
    name = "Александр Владимирович Оченьдлиннаяфамилия"
    fitted = _fit_text(name, 166, 15, True)
    assert fitted.endswith("…")
    assert _font(15, True).getlength(fitted) <= 166


def test_target_class_does_not_invent_precise_image_contact_point(monkeypatch):
    renderer = FrameRenderer(hud_mode="compact")
    frame = np.zeros((360, 640, 3), np.uint8)
    b = _pose("fighter_b", BBox(360, 60, 550, 340))
    circles = []
    original = cv2.circle
    monkeypatch.setattr(
        cv2,
        "circle",
        lambda *args, **kwargs: circles.append(args) or original(*args, **kwargs),
    )
    renderer._draw_selected_contact(frame, {"fighter_b": b}, [_event()], 1000)
    assert circles == []
    event = _event()
    event.evidence = {
        "contact_point_image_norm": {"x": 0.7, "y": 0.25},
        "contact_point_image_confidence": 0.9,
    }
    renderer._draw_selected_contact(frame, {"fighter_b": b}, [event], 1000)
    assert len(circles) == 4
