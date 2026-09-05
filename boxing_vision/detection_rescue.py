"""Budgeted detector-only ROI retry before the single per-frame tracker update.

This module adds real person detections. It has no A/B assignment, appearance
gallery or pose-confidence gate: a visible box with weak wrists needs no retry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import ceil, floor, isfinite
from typing import Any

import numpy as np

from .contracts import BBox, Keypoint
from .pose import PoseBackend, PoseBackendUnavailable, RawPose


def _iou(first: BBox, second: BBox) -> float:
    intersection = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1)) * max(
        0.0, min(first.y2, second.y2) - max(first.y1, second.y1)
    )
    return intersection / max(first.area + second.area - intersection, 1e-6)


def _box(value: object) -> BBox | None:
    if isinstance(value, BBox):
        box = value
    elif isinstance(value, Mapping):
        try:
            box = BBox(**{key: float(value[key]) for key in ("x1", "y1", "x2", "y2")})
        except (KeyError, TypeError, ValueError):
            return None
    else:
        return None
    return (
        box
        if box.area > 0
        and all(isfinite(item) for item in (box.x1, box.y1, box.x2, box.y2))
        else None
    )


def _geometrically_present(expected: BBox, poses: Sequence[RawPose]) -> bool:
    """Confidence of identity/pose is deliberately irrelevant to this test."""
    for pose in poses:
        if _box(pose.bbox) is None:
            continue
        ratio = pose.bbox.area / expected.area
        center_distance = float(
            np.linalg.norm(np.asarray(pose.bbox.center) - expected.center)
        )
        if 0.4 <= ratio <= 2.5 and (
            _iou(expected, pose.bbox) >= 0.30
            or center_distance <= min(expected.width, expected.height) * 0.25
        ):
            return True
    return False


class RescueController:
    """One expanded crop on at most ten percent of analysed frames.

    ``predictions`` are the previous update's lost-track CMC snapshots. Passing
    previous snapshots avoids advancing a tracker twice at the same timestamp.
    Results have no source ID: only the following normal tracker update may
    associate them. RTMLib's Body backend is stateless between calls; no reset or
    second model instance is needed.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_frame_fraction: float = 0.10,
        prediction_age_ms: int = 1000,
        detector_threshold: float = 0.10,
    ) -> None:
        if not 0 < max_frame_fraction <= 0.10:
            raise ValueError(
                "Rescue frame fraction must be greater than zero and at most 0.10"
            )
        if not 0 < prediction_age_ms <= 1000:
            raise ValueError("Rescue predictions must be at most 1000 ms old")
        if not 0.10 <= detector_threshold <= 1:
            raise ValueError("Rescue requires real detector confidence >= 0.10")
        self.enabled = enabled
        self.max_frame_fraction = max_frame_fraction
        self.prediction_age_ms = prediction_age_ms
        self.detector_threshold = detector_threshold
        self.frames_seen = 0
        self.inference_frames = 0
        self.accepted_detections = 0
        self._last_inference_frame = -10_000
        self._shot_id: int | None = None
        self._last_timestamp_ms: int | None = None
        self._source_attempts: dict[str, int] = {}
        self.last_diagnostics: list[dict[str, Any]] = []

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "frames_seen": self.frames_seen,
            "inference_frames": self.inference_frames,
            "accepted_detections": self.accepted_detections,
            "frame_fraction": self.inference_frames / max(1, self.frames_seen),
            "maximum_frame_fraction": self.max_frame_fraction,
        }

    def maybe_infer(
        self,
        frame: np.ndarray,
        poses: Sequence[RawPose],
        predictions: Sequence[Mapping[str, Any]],
        backend: PoseBackend,
        timestamp_ms: int,
        shot_id: int,
    ) -> list[RawPose]:
        self.last_diagnostics = []
        if not self.enabled:
            return []
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms <= self._last_timestamp_ms
        ):
            # Repeated calls do not manufacture budget or run a crop twice.
            return []
        self._last_timestamp_ms = timestamp_ms
        self.frames_seen += 1
        if self._shot_id != shot_id:
            self._shot_id = shot_id
            self._source_attempts.clear()
        if self.inference_frames + 1 > floor(
            self.frames_seen * self.max_frame_fraction + 1e-9
        ):
            return []
        if self.frames_seen - self._last_inference_frame < ceil(
            1 / self.max_frame_fraction
        ):
            return []
        height, width = frame.shape[:2]
        candidates: list[tuple[int, str, BBox, Mapping[str, Any]]] = []
        for prediction in predictions:
            if (
                int(prediction.get("shot_id", -1)) != shot_id
                or prediction.get("source_track_id") is None
            ):
                continue
            evidence = int(prediction.get("evidence_timestamp_ms", -10_000))
            age = timestamp_ms - evidence
            expected = _box(prediction.get("bbox"))
            if expected is None or not 0 < age <= self.prediction_age_ms:
                continue
            cx, cy = expected.center
            if (
                not 0 <= cx < width
                or not 0 <= cy < height
                or _geometrically_present(expected, poses)
            ):
                continue
            source = str(prediction["source_track_id"])
            candidates.append(
                (self._source_attempts.get(source, 0), source, expected, prediction)
            )
        if not candidates:
            return []
        # Retry the least-served lost trajectory first. This is a search budget,
        # not a largest-person or left/right fighter assignment.
        attempts, source, expected, prediction = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        expansion = min(2.0, 1.6 + attempts * 0.2)
        pad_x, pad_y = (
            expected.width * (expansion - 1) / 2,
            expected.height * (expansion - 1) / 2,
        )
        x1, y1 = (
            max(0, floor(expected.x1 - pad_x + 1e-8)),
            max(0, floor(expected.y1 - pad_y + 1e-8)),
        )
        x2, y2 = (
            min(width, ceil(expected.x2 + pad_x - 1e-8)),
            min(height, ceil(expected.y2 + pad_y - 1e-8)),
        )
        if (
            x2 - x1 < 32
            or y2 - y1 < 32
            or (x2 - x1) * (y2 - y1) > width * height * 0.85
        ):
            return []
        self.inference_frames += 1
        self._last_inference_frame = self.frames_seen
        self._source_attempts[source] = attempts + 1
        diagnostic: dict[str, Any] = {
            "timestamp_ms": timestamp_ms,
            "shot_id": shot_id,
            "search_source_track_id": prediction["source_track_id"],
            "reason": "missing_or_collapsed_detector_bbox",
            "roi": [x1, y1, x2, y2],
            "accepted": 0,
        }
        self.last_diagnostics.append(diagnostic)
        try:
            crop_poses = backend.infer(frame[y1:y2, x1:x2].copy())
        except (PoseBackendUnavailable, RuntimeError, ValueError) as exc:
            # The main full-frame inference already succeeded. An optional ROI
            # failure should be visible diagnostically, not abort the whole job.
            diagnostic.update(
                reason="roi_inference_failed", error_type=type(exc).__name__
            )
            return []
        accepted: list[RawPose] = []
        for raw in sorted(
            crop_poses,
            key=lambda item: float(item.detector_confidence or 0),
            reverse=True,
        ):
            score = raw.detector_confidence
            if (
                score is None
                or not isfinite(score)
                or not self.detector_threshold <= score <= 1
            ):
                continue
            raw_box = _box(raw.bbox)
            if raw_box is None:
                continue
            box = BBox(
                max(x1, raw_box.x1 + x1),
                max(y1, raw_box.y1 + y1),
                min(x2, raw_box.x2 + x1),
                min(y2, raw_box.y2 + y1),
                score,
            )
            if box.area <= 0 or not 0.25 <= box.area / expected.area <= 4:
                continue
            if _iou(box, expected) < 0.10:
                continue
            if any(_iou(box, old.bbox) >= 0.55 for old in (*poses, *accepted)):
                continue
            keys = {
                name: Keypoint(point.x + x1, point.y + y1, point.score)
                for name, point in raw.keypoints.items()
                if all(isfinite(value) for value in (point.x, point.y, point.score))
            }
            pose_box = raw.pose_bbox
            if pose_box is not None:
                pose_box = BBox(
                    pose_box.x1 + x1,
                    pose_box.y1 + y1,
                    pose_box.x2 + x1,
                    pose_box.y2 + y1,
                    pose_box.score,
                )
            accepted.append(
                replace(
                    raw,
                    bbox=box,
                    pose_bbox=pose_box,
                    keypoints=keys,
                    source_track_id=None,
                )
            )
        diagnostic["accepted"] = len(accepted)
        self.accepted_detections += len(accepted)
        return accepted
