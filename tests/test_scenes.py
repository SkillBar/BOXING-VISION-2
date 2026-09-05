from pathlib import Path

import numpy as np
import pytest
from scenedetect import FrameTimecode

from boxing_vision import scenes
from boxing_vision.video import VideoFrame


def test_scene_cut_uses_delayed_source_timestamp_and_flushes_detector(monkeypatch):
    seen = []

    class Detector:
        def __init__(self, **kwargs):
            pass

        def process_frame(self, timecode, image):
            assert image.shape[1] == 320
            return [FrameTimecode(2, fps=15.0)] if timecode.frame_num in {4, 5} else []

        def post_process(self, timecode):
            return [FrameTimecode(5, fps=15.0)]

    def source(path, **kwargs):
        seen.append(kwargs)
        for index in range(7):
            yield VideoFrame(
                index, round(index * 1000 / 15), np.zeros((16, 24, 3), np.uint8)
            )

    monkeypatch.setattr("scenedetect.detectors.AdaptiveDetector", Detector)
    monkeypatch.setattr(scenes, "iter_video_frames", source)
    assert scenes.detect_shot_boundaries(Path("unused"), max_duration_ms=10200) == [
        0,
        133,
        333,
    ]
    assert seen[0]["end_s"] == 10.2


def test_scene_scan_cancelled_before_opening_video(monkeypatch):
    monkeypatch.setattr(
        scenes,
        "iter_video_frames",
        lambda *args, **kwargs: pytest.fail("Decoded after cancellation"),
    )
    with pytest.raises(InterruptedError):
        scenes.detect_shot_boundaries(Path("unused"), cancelled=lambda: True)


def test_scene_scan_cancellation_during_stream(monkeypatch):
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 3

    source = [
        VideoFrame(index, index * 100, np.zeros((16, 24, 3), np.uint8))
        for index in range(10)
    ]
    monkeypatch.setattr(
        scenes, "iter_video_frames", lambda *args, **kwargs: iter(source)
    )
    with pytest.raises(InterruptedError):
        scenes.detect_shot_boundaries(Path("unused"), cancelled=cancel)


@pytest.mark.parametrize(
    "kwargs",
    [{"fps": 0}, {"max_duration_ms": -1}, {"end_s": 1, "max_duration_ms": 1000}],
)
def test_invalid_scene_scan_bounds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        scenes.detect_shot_boundaries(Path("unused"), **kwargs)
