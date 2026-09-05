"""Reproducible integration check on visually reviewed enrollment selections.

This is NOT a ground-truth accuracy benchmark. The choices below refer only to
the existing 12-second gym fixture; do not apply them to another video.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boxing_vision.calibration import calibration_backend, normalized_box
from boxing_vision.config import AnalysisConfig
from boxing_vision.pipeline import analyze_video


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "runs/20260904T183133387237Z-75b2e6e9/.render_cache/normalized.mp4"
    backend = calibration_backend()
    capture = cv2.VideoCapture(str(source))
    samples = []
    for timestamp, a, b in ((0, 1, 0), (2, 1, 0), (4, 0, 1)):
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, image = capture.read()
        assert ok
        poses = backend.infer(image)
        height, width = image.shape[:2]
        samples.append({"time_s": timestamp, "fighter_a": normalized_box(poses[a].bbox, width, height),
                        "fighter_b": normalized_box(poses[b].bbox, width, height)})
    capture.release()
    config = AnalysisConfig(enrollment_confirmed=True, enrollment_frames=(0, 2, 4),
        enrollment_samples=tuple(samples), scheduled_rounds=1, fight_end_s=12, hud_mode="compact",
        ring_rois=((.24,.69),(.80,.69),(.98,.97),(.02,.97)))
    result = analyze_video(source, config, lambda p, message: print(f"{p:.0%} {message}", flush=True))
    print(result.run_dir)
    print(result.summary["quality"])


if __name__ == "__main__":
    main()
