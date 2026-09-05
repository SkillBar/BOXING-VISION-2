"""Pose-only 30 fps second pass inside candidates, never new identities."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2

from .contracts import BBox, IdentityState, PoseObservation, PunchEvent, ReviewStatus


def dense_candidate_poses(
    video: Path,
    observations: list[PoseObservation],
    events: list[PunchEvent],
    backend: Any,
    *,
    check_cancel: Callable[[], None] = lambda: None,
    progress: Callable[[int, int], None] | None = None,
    frame_states: Sequence[dict[str, Any]] | None = None,
    source_fps: float = 15.0,
) -> list[PoseObservation]:
    """Only confirmed interpolation endpoints with the SAME local track qualify."""
    check_cancel()
    if source_fps <= 0:
        raise ValueError("source_fps must be positive")
    if not events or not observations:
        return []
    by_time: dict[int, dict[str, PoseObservation]] = defaultdict(dict)
    for observation in observations:
        if observation.fighter_id in by_time[observation.timestamp_ms]:
            raise ValueError("Duplicate fighter identity in one detector frame")
        by_time[observation.timestamp_ms][observation.fighter_id] = observation
    context = {int(row["timestamp_ms"]): row for row in (frame_states or [])}
    # Explicit first-pass frames include timestamps where BOTH identities were
    # lost. Compatibility fallback still refuses a skipped source-frame gap.
    times = sorted(context if frame_states is not None else by_time)
    max_gap_ms = min(150.0, 1.6 * 1000 / source_fps)
    samples = sorted(
        {
            round(event.peak_ms / 1000 * 30) + offset
            for event in events
            for offset in range(-13, 14)
            if round(event.peak_ms / 1000 * 30) + offset >= 0
        }
    )
    result = []
    capture = cv2.VideoCapture(str(video))
    try:
        for count, frame_number in enumerate(samples):
            check_cancel()
            stamp = round(frame_number * 1000 / 30)
            index = bisect_left(times, stamp)
            if index < len(times) and abs(times[index] - stamp) <= 3:
                left = right = times[index]
            elif 0 < index < len(times):
                left, right = times[index - 1], times[index]
            else:
                continue
            if right - left > max_gap_ms:
                continue
            if context and any(
                context.get(time, {}).get("scene_state") != "ACTIVE_FIGHT"
                or context.get(time, {}).get("is_scene_cut", False)
                for time in (left, right)
            ):
                continue
            anchors = []
            for role in ("fighter_a", "fighter_b"):
                before, after = by_time[left].get(role), by_time[right].get(role)
                if before is None or after is None:
                    continue
                expected = (
                    IdentityState.FIGHTER_A
                    if role == "fighter_a"
                    else IdentityState.FIGHTER_B
                )
                if (
                    after.timestamp_ms - before.timestamp_ms > 150
                    or before.shot_id != after.shot_id
                    or before.is_scene_cut
                    or after.is_scene_cut
                    or before.source_track_id is None
                    or before.source_track_id != after.source_track_id
                    or (
                        context
                        and any(
                            context[time].get("shot_id") != before.shot_id
                            for time in (left, right)
                        )
                    )
                    or any(
                        obs.identity_state != expected
                        or obs.scene_state != "ACTIVE_FIGHT"
                        or str(obs.review_status).upper()
                        in {ReviewStatus.NEEDS_REVIEW, ReviewStatus.REJECTED}
                        or float(obs.identity_confidence or 0) < 0.55
                        or float(obs.identity_margin or 0) < 0.12
                        for obs in (before, after)
                    )
                ):
                    continue
                ratio = (stamp - before.timestamp_ms) / max(
                    1, after.timestamp_ms - before.timestamp_ms
                )
                box = BBox(
                    *(
                        a + ratio * (b - a)
                        for a, b in zip(
                            (
                                before.bbox.x1,
                                before.bbox.y1,
                                before.bbox.x2,
                                before.bbox.y2,
                            ),
                            (
                                after.bbox.x1,
                                after.bbox.y1,
                                after.bbox.x2,
                                after.bbox.y2,
                            ),
                        )
                    ),
                    score=min(before.bbox.score, after.bbox.score),
                )
                anchors.append(
                    replace(
                        before,
                        timestamp_ms=stamp,
                        frame_index=frame_number,
                        bbox=box,
                        detector_bbox=box,
                        is_scene_cut=False,
                        identity_confidence=min(
                            before.identity_confidence, after.identity_confidence
                        ),
                        identity_margin=min(
                            before.identity_margin, after.identity_margin
                        ),
                    )
                )
            if (
                len(anchors) != 2
                or anchors[0].shot_id != anchors[1].shot_id
                or anchors[0].source_track_id == anchors[1].source_track_id
            ):
                continue
            capture.set(cv2.CAP_PROP_POS_MSEC, stamp)
            ok, image = capture.read()
            if not ok:
                continue
            check_cancel()
            poses = backend.infer_in_boxes(image, [obs.bbox for obs in anchors])
            if len(poses) != len(anchors):
                continue
            for anchor, pose in zip(anchors, poses):
                result.append(
                    replace(
                        anchor,
                        keypoints=pose.keypoints,
                        pose_confidence=pose.pose_confidence,
                    )
                )
            if progress and count % 30 == 0:
                progress(count, len(samples))
    finally:
        capture.release()
    return result
