from dataclasses import replace

import numpy as np
import pytest

from boxing_vision.contracts import BBox, Keypoint
from boxing_vision.detection_rescue import RescueController
from boxing_vision.pose import RawPose

IMAGE = np.zeros((480, 800, 3), dtype=np.uint8)
EXPECTED = BBox(200, 100, 300, 400, 0.9)


def prediction(now, *, source=5, shot=0, age=100, bbox=EXPECTED):
    return {
        "source_track_id": source,
        "shot_id": shot,
        "timestamp_ms": now - 67,
        "evidence_timestamp_ms": now - age,
        "bbox": bbox.to_dict(),
    }


class Backend:
    def __init__(self, output=None, fail=False):
        self.calls = []
        self.output = output
        self.fail = fail

    def infer(self, image):
        self.calls.append(image.shape)
        if self.fail:
            raise RuntimeError("test error")
        # Expansion1.6 crop is (170,10)..(330,480). Detector estimates the
        # original person in crop coordinates, not copied projected evidence.
        return (
            self.output
            if self.output is not None
            else [
                RawPose(
                    BBox(30, 90, 130, 390, 0.88),
                    {"left_wrist": Keypoint(50, 190, 0.81)},
                    detector_confidence=0.88,
                    pose_confidence=0.81,
                    pose_bbox=BBox(35, 95, 125, 380),
                    source_track_id=99,
                )
            ]
        )


def warm(controller, backend, count=9):
    for index in range(count):
        assert controller.maybe_infer(IMAGE, [], [], backend, index * 67, 0) == []


def test_roi_retry_adds_only_real_translated_measurement_without_forced_id():
    controller = RescueController()
    backend = Backend()
    warm(controller, backend)
    found = controller.maybe_infer(IMAGE, [], [prediction(603)], backend, 603, 0)
    assert len(found) == 1
    item = found[0]
    assert item.bbox == BBox(200, 100, 300, 400, 0.88)
    assert item.keypoints["left_wrist"] == Keypoint(220, 200, 0.81)
    assert item.pose_bbox == BBox(205, 105, 295, 390)
    assert item.source_track_id is None
    assert item.detector_confidence == 0.88
    assert controller.summary["inference_frames"] == 1
    assert controller.summary["frame_fraction"] == 0.1
    assert controller.last_diagnostics[0]["roi"] == [170, 10, 330, 480]


def test_visible_detector_box_with_zero_pose_and_no_appearance_never_triggers():
    controller = RescueController()
    backend = Backend()
    warm(controller, backend)
    visible = RawPose(
        EXPECTED, {}, detector_confidence=0.15, pose_confidence=0, appearance=None
    )
    result = controller.maybe_infer(
        IMAGE, [visible], [prediction(603)], backend, 603, 0
    )
    assert result == []
    assert backend.calls == []


def test_collapsed_detector_geometry_can_trigger_without_discarding_existing():
    controller = RescueController()
    backend = Backend()
    warm(controller, backend)
    fragment = RawPose(BBox(205, 105, 235, 170), {}, detector_confidence=0.9)
    original = fragment.to_dict()
    found = controller.maybe_infer(
        IMAGE, [fragment], [prediction(603)], backend, 603, 0
    )
    assert len(found) == 1
    assert fragment.to_dict() == original


@pytest.mark.parametrize(
    "invalid",
    [
        prediction(603, shot=1),
        prediction(603, age=1500),
        prediction(603, age=-10),
        prediction(603, bbox=BBox(900, 100, 1000, 400)),
        {"source_track_id": None, "shot_id": 0},
    ],
)
def test_no_search_for_stale_wrong_shot_or_offscreen_predictions(invalid):
    controller = RescueController()
    backend = Backend()
    warm(controller, backend)
    assert controller.maybe_infer(IMAGE, [], [invalid], backend, 603, 0) == []
    assert backend.calls == []


def test_rescue_never_exceeds_ten_percent_at_any_prefix_or_repeats_timestamp():
    controller = RescueController()
    backend = Backend(output=[])
    for index in range(120):
        now = index * 67
        controller.maybe_infer(IMAGE, [], [prediction(now)], backend, now, 0)
        controller.maybe_infer(IMAGE, [], [prediction(now)], backend, now, 0)
        assert controller.summary["frame_fraction"] <= 0.10
        assert controller.frames_seen == index + 1
    assert len(backend.calls) == 12


def test_invalid_detector_confidence_cannot_be_replaced_with_pose_score():
    outputs = [
        RawPose(
            BBox(30, 90, 130, 390), {}, detector_confidence=score, pose_confidence=0.99
        )
        for score in [None, 0.05, float("nan"), 1.2]
    ]
    controller, backend = RescueController(), Backend(outputs)
    warm(controller, backend)
    assert controller.maybe_infer(IMAGE, [], [prediction(603)], backend, 603, 0) == []


def test_roi_nms_removes_duplicates_and_keeps_highest_real_score():
    best = RawPose(BBox(30, 90, 130, 390, 0.9), {}, detector_confidence=0.9)
    controller, backend = (
        RescueController(),
        Backend([replace(best, detector_confidence=0.8), best]),
    )
    warm(controller, backend)
    found = controller.maybe_infer(IMAGE, [], [prediction(603)], backend, 603, 0)
    assert len(found) == 1
    assert found[0].detector_confidence == 0.9


def test_optional_roi_failure_is_diagnostic_and_does_not_retry_unbounded():
    controller, backend = RescueController(), Backend(fail=True)
    warm(controller, backend)
    assert controller.maybe_infer(IMAGE, [], [prediction(603)], backend, 603, 0) == []
    assert controller.last_diagnostics[0]["reason"] == "roi_inference_failed"
    assert controller.inference_frames == 1
    assert controller.maybe_infer(IMAGE, [], [prediction(670)], backend, 670, 0) == []
    assert len(backend.calls) == 1


def test_disabled_rescue_never_calls_model():
    controller, backend = RescueController(enabled=False), Backend()
    for index in range(20):
        assert (
            controller.maybe_infer(
                IMAGE, [], [prediction(index * 67)], backend, index * 67, 0
            )
            == []
        )
    assert backend.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_frame_fraction": 0.11},
        {"prediction_age_ms": 1500},
        {"detector_threshold": 0.05},
    ],
)
def test_unsafe_budget_and_evidence_thresholds_rejected(kwargs):
    with pytest.raises(ValueError):
        RescueController(**kwargs)
