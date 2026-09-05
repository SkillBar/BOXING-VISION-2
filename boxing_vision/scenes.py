"""Offline shot boundaries from the real PySceneDetect AdaptiveDetector."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2

from .video import iter_video_frames


def detect_shot_boundaries(
    video: Path,
    *,
    fps: float = 15,
    cancelled: Callable[[], bool] | None = None,
    end_s: float | None = None,
    max_duration_ms: int | None = None,
) -> list[int]:
    from scenedetect import FrameTimecode
    from scenedetect.detectors import AdaptiveDetector

    if max_duration_ms is not None:
        if end_s is not None:
            raise ValueError("Use either end_s or max_duration_ms")
        end_s = max_duration_ms / 1000
    if fps <= 0 or (end_s is not None and end_s < 0):
        raise ValueError("Scene scan requires fps > 0 and end_s >= 0")
    fps = float(fps)
    if cancelled and cancelled():
        raise InterruptedError("Определение сцен отменено")
    detector = AdaptiveDetector(
        adaptive_threshold=3.0,
        min_scene_len=max(3, round(fps * 0.5)),
        window_width=2,
        min_content_val=15.0,
    )
    timestamps: list[int] = []
    boundaries = [0]
    for index, frame in enumerate(
        iter_video_frames(video, target_fps=fps, end_s=end_s)
    ):
        if cancelled and cancelled():
            raise InterruptedError("Определение сцен отменено")
        timestamps.append(frame.timestamp_ms)
        h, w = frame.image.shape[:2]
        reduced = cv2.resize(frame.image, (320, max(2, round(h * 320 / w))))
        for cut in detector.process_frame(FrameTimecode(index, fps=fps), reduced):
            cut_index = cut.frame_num
            if 0 < cut_index < len(timestamps):
                boundaries.append(timestamps[cut_index])
    if timestamps:
        for cut in detector.post_process(FrameTimecode(len(timestamps) - 1, fps=fps)):
            cut_index = cut.frame_num
            if 0 < cut_index < len(timestamps):
                boundaries.append(timestamps[cut_index])
    return sorted(set(boundaries))
