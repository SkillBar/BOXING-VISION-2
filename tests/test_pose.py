from __future__ import annotations

import numpy as np
import pytest

from boxing_vision.contracts import BBox, Keypoint
from boxing_vision.identity import AppearancePart, IdentityState
from boxing_vision.pose import (
    RawPose,
    RTMLibPoseBackend,
    TwoFighterTracker,
    decode_yolox_people,
    extract_appearance_parts,
)

RED = (1.0, 0.0, 0.0)
BLUE = (0.0, 1.0, 0.0)
OTHER = (0.0, 0.0, 1.0)


def _keypoints(center_x: float) -> dict[str, Keypoint]:
    names = (
        "nose",
        "left_eye",
        "right_eye",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    return {
        name: Keypoint(center_x + (index % 3 - 1) * 10, 70 + index * 12, 0.95)
        for index, name in enumerate(names)
    }


def _pose(
    center_x: float,
    appearance: tuple[float, ...],
    *,
    width: float = 80,
    source_track_id: str | None = None,
) -> RawPose:
    return RawPose(
        BBox(center_x - width / 2, 60, center_x + width / 2, 300, 0.97),
        _keypoints(center_x),
        0.97,
        appearance,
        source_track_id=source_track_id,
    )


def _calibrated_tracker() -> TwoFighterTracker:
    tracker = TwoFighterTracker(
        anchors={"fighter_a": (100, 175), "fighter_b": (300, 175)}
    )
    observations = []
    for frame_index in range(3):
        observations = tracker.process(
            frame_index,
            frame_index * 100,
            [_pose(100, RED), _pose(300, BLUE), _pose(205, OTHER, width=135)],
        )
    assert {item.fighter_id for item in observations} == {"fighter_a", "fighter_b"}
    assert {item.identity_state for item in observations} == {
        IdentityState.FIGHTER_A,
        IdentityState.FIGHTER_B,
    }
    tracker.clear_anchors()
    return tracker


def _hsv_descriptor(
    *, red: float = 0.0, blue: float = 0.0, neutral: float = 1.0
) -> tuple[float, ...]:
    histogram = np.zeros((12, 4), dtype=np.float32)
    histogram[3, 0] = neutral
    histogram[0, 3] = red
    histogram[7, 3] = blue
    histogram /= histogram.sum()
    return tuple(float(value) for value in histogram.reshape(-1))


def test_auto_enrollment_requires_three_colour_consistent_frames() -> None:
    tracker = TwoFighterTracker()
    red = _hsv_descriptor(red=4.0)
    blue = _hsv_descriptor(blue=4.0)
    referee = _hsv_descriptor(neutral=4.0)

    first = tracker.process(
        0,
        0,
        [_pose(100, red), _pose(300, blue), _pose(205, referee, width=170)],
    )
    second = tracker.process(
        1,
        100,
        [_pose(102, red), _pose(298, blue), _pose(205, referee, width=170)],
    )
    third = tracker.process(
        2,
        200,
        [_pose(104, red), _pose(296, blue), _pose(205, referee, width=170)],
    )

    assert first == []
    assert second == []
    assert {item.fighter_id for item in third} == {"fighter_a", "fighter_b"}
    assert all(
        item.identity_margin is not None and item.identity_margin >= 0.12
        for item in third
    )
    assert len(tracker.negative_gallery) == 1


def test_auto_enrollment_refuses_neutral_people() -> None:
    tracker = TwoFighterTracker()
    neutral_a = _hsv_descriptor(neutral=4.0)
    neutral_b = _hsv_descriptor(neutral=3.0)

    for frame_index in range(5):
        observations = tracker.process(
            frame_index,
            frame_index * 100,
            [_pose(100, neutral_a), _pose(300, neutral_b)],
        )

    assert observations == []
    assert tracker.core_gallery == {}


def test_missing_a_stays_unknown_while_b_remains_stable() -> None:
    tracker = _calibrated_tracker()

    observations = tracker.process(
        1,
        100,
        [_pose(292, BLUE), _pose(170, OTHER, width=150)],
    )

    assert [item.fighter_id for item in observations] == ["fighter_b"]
    assert tracker.last_identity_states["fighter_a"] == IdentityState.UNKNOWN
    assert tracker.last_identity_states["fighter_b"] == IdentityState.FIGHTER_B


def test_large_referee_is_never_force_filled_into_missing_identity() -> None:
    tracker = _calibrated_tracker()

    observations = tracker.process(1, 100, [_pose(190, OTHER, width=190)])

    assert observations == []
    assert set(tracker.last_identity_states.values()) == {IdentityState.UNKNOWN}


def test_b_can_appear_first_without_becoming_fighter_a() -> None:
    tracker = TwoFighterTracker(
        anchors={"fighter_a": (100, 175), "fighter_b": (300, 175)}
    )

    observations = tracker.process(0, 0, [_pose(297, BLUE)])

    assert [item.fighter_id for item in observations] == ["fighter_b"]
    assert tracker.last_identity_states["fighter_a"] == IdentityState.UNKNOWN


def test_overlap_does_not_update_adaptive_gallery() -> None:
    tracker = _calibrated_tracker()
    for frame_index in range(1, 8):
        tracker.process(
            frame_index,
            frame_index * 100,
            [_pose(190, RED, width=120), _pose(215, BLUE, width=120)],
        )

    assert tracker.adaptive_gallery[IdentityState.FIGHTER_A] == ()
    assert tracker.adaptive_gallery[IdentityState.FIGHTER_B] == ()


def test_reset_preserves_custom_adaptive_gallery_policy() -> None:
    tracker = TwoFighterTracker(
        adaptive_identity_confidence_min=0.83,
        adaptive_identity_margin_min=0.29,
    )

    tracker.reset(keep_appearance=False)

    assert tracker.adaptive_identity_confidence_min == 0.83
    assert tracker.adaptive_identity_margin_min == 0.29
    assert tracker._gallery.adaptive_confidence_min == 0.83
    assert tracker._gallery.adaptive_margin_min == 0.29


def test_reacquisition_after_cut_requires_three_of_five_votes() -> None:
    tracker = _calibrated_tracker()
    frames = [
        tracker.process(
            index,
            index * 100,
            [
                _pose(300, RED, source_track_id="red-shot-2"),
                _pose(100, BLUE, source_track_id="blue-shot-2"),
            ],
            scene_cut=index == 1,
        )
        for index in range(1, 4)
    ]

    assert frames[0] == []
    assert frames[1] == []
    assert {item.fighter_id for item in frames[2]} == {"fighter_a", "fighter_b"}
    by_id = {item.fighter_id: item for item in frames[2]}
    assert by_id["fighter_a"].bbox.center[0] == 300
    assert by_id["fighter_b"].bbox.center[0] == 100
    assert tracker.shot_id == 1


def test_rtmlib_backend_preserves_detector_bbox_when_available() -> None:
    coordinates = np.asarray(
        [[[40.0 + index, 50.0 + index * 4] for index in range(17)]],
        dtype=np.float32,
    )
    scores = np.full((1, 17), 0.95, dtype=np.float32)

    class FakeBody:
        one_stage = False

        def det_model(self, frame: np.ndarray) -> np.ndarray:
            del frame
            return np.asarray([[10.0, 20.0, 120.0, 220.0]], dtype=np.float32)

        def pose_model(
            self,
            frame: np.ndarray,
            *,
            bboxes: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            del frame
            assert bboxes.shape == (1, 4)
            return coordinates, scores

    backend = RTMLibPoseBackend()
    backend._model = FakeBody()

    poses = backend.infer(np.zeros((240, 160, 3), dtype=np.uint8))

    assert len(poses) == 1
    assert (
        poses[0].bbox.x1,
        poses[0].bbox.y1,
        poses[0].bbox.x2,
        poses[0].bbox.y2,
    ) == (10.0, 20.0, 120.0, 220.0)
    assert poses[0].pose_bbox is not None
    assert poses[0].pose_bbox != poses[0].bbox


def test_rtmlib_backend_does_not_run_full_frame_pose_without_detections() -> None:
    class FakeBody:
        one_stage = False

        def det_model(self, frame: np.ndarray) -> np.ndarray:
            del frame
            return np.empty((0, 4), dtype=np.float32)

        def pose_model(self, frame: np.ndarray, *, bboxes: np.ndarray) -> object:
            del frame, bboxes
            raise AssertionError("pose_model must not run without a person bbox")

    backend = RTMLibPoseBackend()
    backend._model = FakeBody()

    assert backend.infer(np.zeros((240, 160, 3), dtype=np.uint8)) == []


def test_yolox_preserves_low_score_objectness_times_person_probability() -> None:
    predictions = np.zeros((1, 21, 6), dtype=np.float32)
    predictions[0, 5, 4:] = (0.5, 0.4)
    boxes, scores = decode_yolox_people(predictions, (32, 32), 0.5)
    assert len(boxes) == 1
    assert scores[0] == pytest.approx(0.2)
    assert boxes[0].tolist() == [8.0, 8.0, 24.0, 24.0]
    assert predictions[0, 5, :4].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_yolox_nms_retains_confidence_and_rejects_invalid_boxes() -> None:
    predictions = np.array(
        [
            [
                [0, 0, 100, 200, 0.2],
                [1, 1, 100, 200, 0.9],
                [200, 0, 300, 200, 0.15],
                [0, 0, 2, 2, 0.05],
                [np.nan, 0, 2, 2, 0.9],
            ]
        ],
        dtype=np.float32,
    )
    boxes, scores = decode_yolox_people(predictions, (640, 640), 1.0)
    assert len(boxes) == 2
    assert scores.tolist() == pytest.approx([0.9, 0.15])


def test_detector_confidence_is_not_mean_pose_and_pose_only_skips_detector() -> None:
    class FakeBody:
        one_stage = False
        detections = 0

        def det_model(self, frame):
            self.detections += 1
            return np.asarray([[10.0, 20.0, 120.0, 220.0, 0.2]], dtype=np.float32)

        def pose_model(self, frame, *, bboxes):
            coordinates = np.array(
                [[[40.0 + index, 50.0 + index * 4] for index in range(17)]],
                dtype=np.float32,
            )
            return coordinates, np.full((1, 17), 0.99, dtype=np.float32)

    backend = RTMLibPoseBackend()
    backend._model = FakeBody()
    frame = np.zeros((240, 160, 3), dtype=np.uint8)
    poses = backend.infer(frame)
    assert poses[0].detector_confidence == pytest.approx(0.2)
    assert poses[0].bbox.score == pytest.approx(0.2)
    assert poses[0].pose_confidence == pytest.approx(0.99)
    refined = backend.infer_in_boxes(frame, [poses[0].bbox])
    assert backend._model.detections == 1
    assert refined[0].bbox == poses[0].bbox
    assert refined[0].source_track_id is None


def test_appearance_masks_reject_occluder_pixels_and_hidden_parts() -> None:
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[:] = (255, 0, 0)
    frame[60:160, 90:210] = (0, 0, 255)
    points = {
        name: Keypoint(x, y, 0.99)
        for name, x, y in (
            ("left_shoulder", 100, 65),
            ("right_shoulder", 200, 65),
            ("left_hip", 105, 150),
            ("right_hip", 195, 150),
        )
    }
    box = BBox(70, 20, 230, 280, 0.9)
    parts = extract_appearance_parts(frame, box, points)
    assert set(parts) == {"torso", "waistband"}
    assert parts["torso"].histogram[3] == pytest.approx(0.65)
    assert extract_appearance_parts(frame, box, points, [box]) == {}
    # The same loose bbox, with the other person's actual pose located outside
    # this torso, must not erase empty background inside that person's box.
    other = {
        name: Keypoint(point.x + 130, point.y, point.score)
        for name, point in points.items()
    }
    local = extract_appearance_parts(
        frame, box, points, [box], occluder_keypoints=[other]
    )
    assert "torso" in local
    assert local["torso"].histogram == parts["torso"].histogram


def test_raw_pose_cache_roundtrip_preserves_part_and_detector_evidence() -> None:
    pose = _pose(100, RED, source_track_id="42")
    pose.detector_confidence = 0.21
    pose.pose_confidence = 0.96
    pose.appearance_parts = {"torso": AppearancePart((0.2, 0.8), 0.9)}
    restored = RawPose.from_dict(pose.to_dict())
    assert restored == pose


def test_profile_roundtrip_preserves_roi_override_and_core_is_write_once() -> None:
    tracker = TwoFighterTracker()
    tracker.enroll(
        {"fighter_a": [_pose(100, RED)] * 3, "fighter_b": [_pose(300, BLUE)] * 3},
        [_pose(200, OTHER)],
    )
    tracker.set_ring_roi(
        [(0.0, 0.0), (0.6, 0.0), (0.6, 1.0), (0.0, 1.0)],
        frame_size=(400, 400),
        shot_id=2,
    )
    tracker.set_identity_overrides({"shot-2-track-known": "FIGHTER_B"})
    restored = TwoFighterTracker()
    restored.import_identity_profile(tracker.export_identity_profile())
    assert (
        restored.score_candidate(_pose(300, BLUE), shot_id=2).reason == "outside_ring"
    )
    assert (
        restored.score_candidate(
            _pose(300, BLUE, source_track_id="known"), shot_id=2
        ).reason
        == "user_override"
    )
    with pytest.raises(RuntimeError):
        restored.import_identity_profile(tracker.export_identity_profile())


def test_legacy_reacquisition_does_not_combine_votes_from_different_people() -> None:
    tracker = _calibrated_tracker()
    decoded = [
        tracker.process(
            index + 3, (index + 3) * 100, [_pose(center, RED)], scene_cut=index == 0
        )
        for index, center in enumerate((100, 200, 300, 300, 300))
    ]
    assert all(not result for result in decoded[:4])
    assert [item.fighter_id for item in decoded[4]] == ["fighter_a"]
    assert decoded[4][0].source_track_id is None


def test_tiny_detection_is_not_promoted_to_pose_derived_person_box() -> None:
    backend = RTMLibPoseBackend()
    coordinates = np.asarray(
        [[[40.0 + i, 50.0 + i * 4] for i in range(17)]], dtype=np.float32
    )
    scores = np.full((1, 17), 0.99, dtype=np.float32)
    assert (
        backend._decode_poses(
            np.zeros((240, 160, 3), np.uint8),
            coordinates,
            scores,
            np.asarray([[10.0, 10.0, 20.0, 20.0]]),
            np.asarray([0.9]),
        )
        == []
    )
