from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from boxing_vision.appearance_refresh import refresh_cached_appearance
from boxing_vision.contracts import BBox, Keypoint
from boxing_vision.pose import RawPose, TwoFighterTracker
from boxing_vision.tracking import TrackingFrame


def test_refresh_cache_reuses_real_detections_and_reenrolls_without_ml(
    tmp_path, monkeypatch
):
    source = tmp_path / "original" / ".render_cache"
    destination = tmp_path / "clone" / ".render_cache"
    source.mkdir(parents=True)

    def pose(x, descriptor, source_id):
        keys = {
            name: Keypoint(x + dx, y, 0.95)
            for name, dx, y in (
                ("left_shoulder", 0, 50),
                ("right_shoulder", 50, 50),
                ("left_hip", 0, 120),
                ("right_hip", 50, 120),
                ("left_knee", 0, 180),
                ("right_knee", 50, 180),
                ("nose", 25, 25),
            )
        }
        return RawPose(
            BBox(x - 10, 5, x + 70, 220, 0.84),
            keys,
            0.84,
            descriptor,
            source_track_id=source_id,
            detector_confidence=0.84,
            pose_confidence=0.9,
        )

    a, b = pose(60, (1.0, 0.0), 4), pose(240, (0.0, 1.0), 8)
    tracker = TwoFighterTracker()
    tracker.enroll({"fighter_a": [a] * 3, "fighter_b": [b] * 3})
    profile = tracker.export_identity_profile()
    profile["user_gallery_metadata"] = {"approved_view": "example"}
    (source / "first_pass.json").write_text(json.dumps({"identity_profile": profile}))

    def normalized(box):
        return [
            value / dimension
            for value, dimension in zip(
                (box.x1, box.y1, box.x2, box.y2), (400, 240, 400, 240)
            )
        ]

    samples = [
        {"time_s": t, "fighter_a": normalized(a.bbox), "fighter_b": normalized(b.bbox)}
        for t in (0.0, 0.1, 0.2)
    ]
    (source / "config.json").write_text(
        json.dumps({"config": {"enrollment_samples": samples}})
    )
    original = [TrackingFrame(i, i * 100, 0, [a, b]) for i in range(4)]
    with gzip.open(source / "detections.jsonl.gz", "wt") as handle:
        for frame in original:
            handle.write(json.dumps(frame.to_dict()) + "\n")
    before = (source / "detections.jsonl.gz").read_bytes()

    class Capture:
        def __init__(self, *_):
            self.image = np.zeros((240, 400, 3), np.uint8)
            self.image[:, :200] = (0, 0, 255)
            self.image[:, 200:] = (255, 0, 0)

        def isOpened(self):
            return True

        def get(self, _):
            return 10

        def read(self):
            return True, self.image.copy()

        def grab(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr("boxing_vision.appearance_refresh.cv2.VideoCapture", Capture)
    report = refresh_cached_appearance(source, destination)
    assert report["detector_rerun"] is report["pose_rerun"] is False
    assert (source / "detections.jsonl.gz").read_bytes() == before
    with gzip.open(destination / "detections.jsonl.gz", "rt") as handle:
        refreshed = [TrackingFrame.from_dict(json.loads(line)) for line in handle]
    for old, new in zip(original, refreshed):
        for previous_pose, current_pose in zip(old.poses, new.poses):
            assert previous_pose.bbox == current_pose.bbox
            assert previous_pose.keypoints == current_pose.keypoints
            assert previous_pose.source_track_id == current_pose.source_track_id
            assert current_pose.detector_confidence == 0.84
            assert {
                "torso",
                "shorts",
                "headgear",
                "waistband",
            } <= current_pose.appearance_parts.keys()
    refreshed_profile = json.loads((destination / "identity_profile.json").read_text())
    assert (
        refreshed_profile["user_gallery_metadata"] == profile["user_gallery_metadata"]
    )
    assert (
        refreshed_profile["core_parts"]["FIGHTER_A"]
        != profile["core_parts"]["FIGHTER_A"]
    )
    with pytest.raises(ValueError, match="Destination"):
        refresh_cached_appearance(source, destination)
    with pytest.raises(ValueError, match="отдельный"):
        refresh_cached_appearance(source, source.parent / "another-cache")
