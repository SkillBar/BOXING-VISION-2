from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxing_vision import refinement
from boxing_vision.contracts import BBox, PoseObservation, PunchEvent


def observation(stamp, role, source):
    return PoseObservation(
        round(stamp * 0.015),
        stamp,
        role,
        BBox(source * 100, 0, source * 100 + 80, 200),
        {},
        source_track_id=source,
        identity_confidence=0.95,
        identity_margin=0.3,
    )


def observations():
    return [
        observation(round(index * 1000 / 15), role, source)
        for index in range(31)
        for role, source in (("fighter_a", 1), ("fighter_b", 2))
    ]


def frames():
    return [
        {
            "timestamp_ms": round(index * 1000 / 15),
            "shot_id": 0,
            "scene_state": "ACTIVE_FIGHT",
            "is_scene_cut": False,
        }
        for index in range(31)
    ]


def event():
    return PunchEvent(
        "event",
        1,
        900,
        1000,
        1100,
        "fighter_a",
        "fighter_b",
        "left",
        "hook",
        "body",
        "unclear",
        0.7,
        50,
    )


class FakeCapture:
    def __init__(self):
        self.timestamp = 0
        self.released = False

    def set(self, key, value):
        self.timestamp = value

    def read(self):
        return True, np.zeros((220, 400, 3), np.uint8)

    def release(self):
        self.released = True


class PoseOnly:
    def __init__(self):
        self.calls = 0

    def infer(self, image):
        raise AssertionError("Refinement must never invoke person detection")

    def infer_in_boxes(self, image, boxes):
        self.calls += 1
        return [SimpleNamespace(keypoints={}, pose_confidence=0.9) for box in boxes]


@pytest.fixture
def capture(monkeypatch):
    capture = FakeCapture()
    monkeypatch.setattr(refinement.cv2, "VideoCapture", lambda _: capture)
    return capture


def test_dense_pass_calls_only_pose_in_confirmed_boxes(capture):
    backend = PoseOnly()
    result = refinement.dense_candidate_poses(
        Path("not-opened.mp4"),
        observations(),
        [event()],
        backend,
        frame_states=frames(),
    )
    assert len(result) == 54 and backend.calls == 27
    assert capture.released
    assert len({obs.timestamp_ms for obs in result}) == 27
    assert all(obs.detector_bbox == obs.bbox for obs in result)


@pytest.mark.parametrize(
    "kind",
    ["single_missing", "both_missing", "unknown", "source", "scene", "cut", "margin"],
)
def test_dense_does_not_bridge_missing_or_invalid_detector_frame(capture, kind):
    source, context = observations(), frames()
    if kind in {"single_missing", "both_missing"}:
        source = [
            obs
            for obs in source
            if not (
                obs.timestamp_ms == 1000
                and (kind == "both_missing" or obs.fighter_id == "fighter_a")
            )
        ]
    elif kind == "source":
        source = [
            replace(obs, source_track_id=3)
            if obs.timestamp_ms >= 1000 and obs.fighter_id == "fighter_a"
            else obs
            for obs in source
        ]
    elif kind in {"scene", "cut"}:
        context[15]["scene_state" if kind == "scene" else "is_scene_cut"] = (
            "BREAK" if kind == "scene" else True
        )
    else:
        source = [
            replace(
                obs,
                **(
                    {"identity_state": "UNKNOWN"}
                    if kind == "unknown"
                    else {"identity_margin": 0.01}
                ),
            )
            if obs.timestamp_ms == 1000 and obs.fighter_id == "fighter_a"
            else obs
            for obs in source
        ]
    result = refinement.dense_candidate_poses(
        Path("unused"), source, [event()], PoseOnly(), frame_states=context
    )
    stamps = {obs.timestamp_ms for obs in result}
    assert 967 not in stamps
    if kind != "source":
        assert 1000 not in stamps and 1033 not in stamps


def test_source_fps_fallback_refuses_one_fully_missing_15fps_frame(capture):
    source = [obs for obs in observations() if obs.timestamp_ms != 1000]
    result = refinement.dense_candidate_poses(
        Path("unused"), source, [event()], PoseOnly()
    )
    assert not {967, 1000, 1033} & {obs.timestamp_ms for obs in result}


def test_refinement_cancellation_releases_capture_without_detector(capture):
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise InterruptedError("cancel")

    with pytest.raises(InterruptedError):
        refinement.dense_candidate_poses(
            Path("unused"),
            observations(),
            [event()],
            PoseOnly(),
            check_cancel=cancel,
            frame_states=frames(),
        )
    assert capture.released


def test_empty_candidate_list_does_not_open_video(monkeypatch):
    monkeypatch.setattr(
        refinement.cv2,
        "VideoCapture",
        lambda _: pytest.fail("Video opened without candidates"),
    )
    assert (
        refinement.dense_candidate_poses(Path("unused"), observations(), [], PoseOnly())
        == []
    )
