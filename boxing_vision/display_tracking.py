"""Measured/predicted presentation tracks, isolated from analytical poses.

The display cache is derived from first-pass evidence. Nothing in this module
can produce a PoseObservation, update an identity gallery or count a punch.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from .contracts import (
    BBox,
    DisplayState,
    DisplayTrack,
    IdentityState,
    Keypoint,
    PoseObservation,
    RenderFrameContext,
)
from .identity import is_confirmed_identity

if TYPE_CHECKING:
    from .tracking import TrackingFrame


def display_track_from_dict(value: Mapping[str, Any]) -> DisplayTrack:
    """Read a v3 display record without legacy analytical identity inference."""
    fields = DisplayTrack.__dataclass_fields__
    result = {key: item for key, item in value.items() if key in fields}
    result["bbox"] = BBox(**value["bbox"])
    result["keypoints"] = {
        name: Keypoint(**point) for name, point in value.get("keypoints", {}).items()
    }
    return DisplayTrack(**result)


def _known(track: DisplayTrack) -> bool:
    return track.fighter_id in {"fighter_a", "fighter_b"} and str(
        track.identity_state
    ) == {"fighter_a": "FIGHTER_A", "fighter_b": "FIGHTER_B"}.get(track.fighter_id)


def _key(track: DisplayTrack) -> tuple[int, str, str]:
    # Segment boundaries stop interpolation even if BoT-SORT reused an ID.
    return track.shot_id, str(track.source_track_id), str(track.segment_id or "legacy")


def _bbox_values(box: BBox) -> np.ndarray:
    return np.asarray([box.x1, box.y1, box.x2, box.y2], dtype=float)


def _valid_box(box: BBox, frame_size: tuple[int, int] | None) -> bool:
    if not np.isfinite(_bbox_values(box)).all() or box.area <= 0:
        return False
    if frame_size is None:
        return box.x2 > 0 and box.y2 > 0
    width, height = frame_size
    cx, cy = box.center
    return 0 <= cx < width and 0 <= cy < height


def _continuous_box(previous: DisplayTrack, box: BBox, now: int) -> bool:
    """Continuity permits a neutral box, never an inferred fighter role."""
    if not 0 <= now - previous.evidence_timestamp_ms <= 1000:
        return False
    ratio = box.area / max(previous.bbox.area, 1.0)
    distance = float(np.linalg.norm(np.asarray(box.center) - previous.bbox.center))
    return (
        0.4 <= ratio <= 2.5 and distance <= max(box.height, previous.bbox.height) * 0.5
    )


def build_display_tracks(
    frames: Sequence[TrackingFrame],
    observations: Sequence[PoseObservation],
    diagnostics: Sequence[Mapping[str, Any]] = (),
    *,
    prediction_ms: int = 1000,
    frame_size: tuple[int, int] | None = None,
) -> list[DisplayTrack]:
    """Keep eligible measured people visible, never promote arbitrary people.

    A source becomes eligible only after an actual confirmed observation.
    Unlinked segment changes, OTHER and cuts revoke that eligibility. An
    identity conflict revokes its colour while retaining the measured person.
    Missing appearance alone retains a *neutral* measured box. A missing person
    may use the cached CMC/Kalman prediction; old caches use labelled bounded
    constant velocity. Neither branch fabricates joints.
    """
    if not 0 <= prediction_ms <= 1000:
        raise ValueError("prediction_ms must be between 0 and 1000")
    confirmed = {
        (item.shot_id, item.timestamp_ms, str(item.source_track_id)): item
        for item in observations
        if is_confirmed_identity(item) and item.source_track_id is not None
    }
    # Offline physical membership is different from identity at this instant.
    # A later confirmed member can establish that earlier measured boxes belong
    # to the same validated physical chain, without retrospectively colouring
    # them or inventing pose evidence. Ambiguous/rejected chains cannot seed it.
    physical_roles: dict[tuple[int, str], set[str]] = defaultdict(set)
    physical_confirmations: dict[tuple[int, str], list[int]] = defaultdict(list)
    for item in confirmed.values():
        if item.physical_track_id is not None:
            physical_roles[item.shot_id, str(item.physical_track_id)].add(item.fighter_id)
            physical_confirmations[item.shot_id, str(item.physical_track_id)].append(item.timestamp_ms)
    blocked_physical = {
        (int(item.get("shot_id", 0)), str(item.get("physical_track_id")))
        for item in diagnostics
        if str(item.get("identity_state")) == "OTHER"
        or item.get("reason") in {"user_rejected", "outside_ring", "geometry_jump"}
        or item.get("segment_reason") == "identity_conflict_boundary"
    }
    measured_times: dict[tuple[int, str], set[int]] = defaultdict(set)
    for item in diagnostics:
        if item.get("physical_track_id") is not None:
            measured_times[int(item.get("shot_id", 0)), str(item["physical_track_id"])].add(int(item["timestamp_ms"]))
    for key, times in physical_confirmations.items():
        measured_times[key].update(times)
    cuts = [(frame.shot_id, frame.timestamp_ms) for frame in frames if frame.is_scene_cut]
    for key, times in measured_times.items():
        ordered = sorted(times)
        if any(b - a > 1500 for a, b in pairwise(ordered)) or any(
            shot == key[0] and ordered[0] < timestamp <= ordered[-1] for shot, timestamp in cuts
        ):
            # Defend against malformed/legacy physical IDs crossing a barrier.
            blocked_physical.add(key)
    physical_members = {key: next(iter(roles)) for key, roles in physical_roles.items()
                        if len(roles) == 1 and key not in blocked_physical}
    diagnostic_by_source = {
        (
            int(item.get("shot_id", 0)),
            int(item["timestamp_ms"]),
            str(item.get("source_track_id")),
        ): item
        for item in diagnostics
        if "timestamp_ms" in item
    }
    result: list[DisplayTrack] = []
    last: dict[str, DisplayTrack] = {}
    preceding: dict[str, DisplayTrack] = {}
    ancestry: dict[str, DisplayTrack] = {}
    active_shot: int | None = None
    identity_conflicts = {
        "source_identity_conflict",
        "identity_conflict",
        "segment_identity_conflict",
        "opponent_color",
    }
    barriers = {
        "user_rejected",
        "geometry_jump",
        "scene_cut",
    }

    def stop(source: str, now: int, *, revoke: bool = False) -> None:
        previous = last.pop(source, None)
        preceding.pop(source, None)
        if revoke:
            for token, item in tuple(ancestry.items()):
                if str(item.source_track_id) == source:
                    ancestry.pop(token, None)
        if previous is not None:
            result.append(
                replace(
                    previous,
                    timestamp_ms=now,
                    display_state=DisplayState.LOST,
                    keypoints={},
                )
            )

    for frame in sorted(frames, key=lambda item: (item.timestamp_ms, item.shot_id)):
        now, shot = frame.timestamp_ms, frame.shot_id
        if (
            active_shot != shot
            or frame.is_scene_cut
            or str(frame.scene_state) != "ACTIVE_FIGHT"
        ):
            for source in tuple(last):
                stop(source, now)
            ancestry.clear()
            active_shot = shot
        if str(frame.scene_state) != "ACTIVE_FIGHT":
            continue
        present = {
            str(pose.source_track_id)
            for pose in frame.poses
            if pose.source_track_id is not None
        }
        known_here = {
            match.fighter_id: str(pose.source_track_id)
            for pose in frame.poses
            if (match := confirmed.get((shot, now, str(pose.source_track_id))))
            is not None
        }
        # A recovered physical participant must not also have an old forecast.
        for source, previous in tuple(last.items()):
            replacement = known_here.get(previous.fighter_id)
            if replacement is not None and replacement != source:
                stop(source, now, revoke=True)

        for pose in frame.poses:
            if pose.source_track_id is None:
                continue
            source = str(pose.source_track_id)
            match = confirmed.get((shot, now, source))
            diagnostic = diagnostic_by_source.get((shot, now, source), {})
            previous = last.get(source)
            segment = diagnostic.get("segment_id") or (
                match.segment_id if match else None
            )
            physical = diagnostic.get("physical_track_id") or (
                match.physical_track_id if match else None
            )
            if previous is None and physical is not None:
                linked_previous = ancestry.get(str(physical))
                if (
                    linked_previous is not None
                    and linked_previous.shot_id == shot
                    and 0 <= now - linked_previous.evidence_timestamp_ms <= 1500
                ):
                    # This linkage was decided from the physical-track graph,
                    # not from a nearest/largest-person presentation fallback.
                    previous = linked_previous
                    stop(str(previous.source_track_id), now)
            reason = str(diagnostic.get("reason", ""))
            hard_rejection = (
                str(diagnostic.get("identity_state", "")) == "OTHER"
                or reason in barriers
            )
            changed_segment = (
                previous is not None
                and segment is not None
                and str(segment) != str(previous.segment_id)
            )
            conflict = reason in identity_conflicts
            if conflict:
                # A detected, previously eligible person remains visible in
                # neutral grey. Its ambiguous role cannot power a forecast.
                match = None
            linked_segment = changed_segment and (
                (physical is not None and previous.physical_track_id == str(physical))
                or (
                    str(previous.source_track_id) == source
                    and _continuous_box(previous, pose.bbox, now)
                )
            )
            if (
                hard_rejection
                or (changed_segment and not linked_segment)
                or not _valid_box(pose.bbox, frame_size)
            ):
                stop(source, now, revoke=True)
                if physical is not None:
                    ancestry.pop(str(physical), None)
                previous = None
            elif linked_segment:
                # Terminate the old interpolation stream, then retain only a
                # neutral measurement unless the new segment was confirmed.
                stop(source, now)
            if hard_rejection or not _valid_box(pose.bbox, frame_size):
                continue
            physical_key = (shot, str(physical))
            physical_member = physical_members.get(physical_key) if physical is not None and now < max(physical_confirmations.get(physical_key, [now])) else None
            if match is None and previous is None and physical_member is None:
                continue
            identity = match.identity_state if match else IdentityState.UNKNOWN
            track = DisplayTrack(
                timestamp_ms=now,
                evidence_timestamp_ms=now,
                bbox=pose.bbox,
                keypoints=dict(pose.keypoints),
                shot_id=shot,
                source_track_id=pose.source_track_id,
                segment_id=str(segment)
                if segment is not None
                else (previous.segment_id if previous else None),
                physical_track_id=str(physical)
                if physical is not None
                else (previous.physical_track_id if previous else None),
                fighter_id=match.fighter_id if match else previous.fighter_id if previous else physical_member,
                identity_state=identity,
                identity_origin=(match.identity_origin or "verified_pose")
                if match
                else "measured_identity_uncertain" if previous else "physical_membership_only",
                detector_confidence=pose.detector_confidence,
                identity_confidence=match.identity_confidence if match else 0.0,
                identity_margin=match.identity_margin if match else 0.0,
                scene_state=frame.scene_state,
                is_scene_cut=frame.is_scene_cut,
            )
            result.append(track)
            if previous is not None:
                preceding[source] = previous
            last[source] = track
            if track.physical_track_id is not None:
                ancestry[track.physical_track_id] = track

        predictions = {
            str(item["source_track_id"]): item
            for item in getattr(frame, "tracker_predictions", ())
            if item.get("source_track_id") is not None
        }
        for source, previous in tuple(last.items()):
            if source in present:
                continue
            age = now - previous.evidence_timestamp_ms
            if not _known(previous) or age > prediction_ms or prediction_ms == 0:
                stop(source, now)
                continue
            prediction = predictions.get(source)
            origin = "legacy_motion_prediction"
            box = previous.bbox
            if prediction is not None and int(prediction.get("shot_id", shot)) == shot:
                raw_box = prediction.get("bbox", {})
                box = raw_box if isinstance(raw_box, BBox) else BBox(**raw_box)
                evidence = int(
                    prediction.get(
                        "evidence_timestamp_ms", previous.evidence_timestamp_ms
                    )
                )
                if now - evidence > prediction_ms:
                    stop(source, now)
                    continue
                origin = "botsort_cmc_prediction"
            else:
                before = preceding.get(source)
                if (
                    before is not None
                    and 0 < previous.timestamp_ms - before.timestamp_ms <= 250
                ):
                    delta = previous.timestamp_ms - before.timestamp_ms
                    velocity = (
                        _bbox_values(previous.bbox) - _bbox_values(before.bbox)
                    ) / delta
                    bound = max(1.0, previous.bbox.height) * 0.5 / 1000
                    velocity = np.clip(velocity, -bound, bound)
                    box = BBox(
                        *(_bbox_values(previous.bbox) + velocity * age),
                        score=previous.bbox.score,
                    )
            if not _valid_box(box, frame_size):
                stop(source, now)
                continue
            result.append(
                replace(
                    previous,
                    timestamp_ms=now,
                    bbox=box,
                    keypoints={},
                    display_state=DisplayState.PREDICTED,
                    identity_origin=origin,
                )
            )
    return sorted(
        result, key=lambda item: (item.timestamp_ms, str(item.source_track_id))
    )


class DisplayTrackSampler:
    """Pre-indexed, seek-safe sampling at output FPS without analytical poses."""

    def __init__(
        self, tracks: Sequence[DisplayTrack], *, max_gap_ms: int = 250
    ) -> None:
        if not 0 <= max_gap_ms <= 250:
            raise ValueError("max_gap_ms must be between 0 and 250")
        self.max_gap_ms = max_gap_ms
        grouped: dict[tuple[int, str, str], list[DisplayTrack]] = defaultdict(list)
        for track in tracks:
            grouped[_key(track)].append(track)
        self.tracks = {
            key: sorted(items, key=lambda item: item.timestamp_ms)
            for key, items in grouped.items()
        }
        self.timestamps = {
            key: [item.timestamp_ms for item in items]
            for key, items in self.tracks.items()
        }

    def sample(
        self, timestamp_ms: int, frame_context: RenderFrameContext | None = None
    ) -> list[DisplayTrack]:
        if (
            frame_context is not None
            and str(frame_context.scene_state) != "ACTIVE_FIGHT"
        ):
            return []
        result: list[DisplayTrack] = []
        for key, items in self.tracks.items():
            if frame_context is not None and key[0] != frame_context.shot_id:
                continue
            index = bisect_right(self.timestamps[key], timestamp_ms) - 1
            if index < 0:
                continue
            before = items[index]
            if str(before.display_state) == "LOST":
                continue
            age = timestamp_ms - before.timestamp_ms
            if age == 0:
                result.append(before)
                continue
            after = items[index + 1] if index + 1 < len(items) else None
            can_interpolate = (
                after is not None
                and 0 < after.timestamp_ms - before.timestamp_ms <= self.max_gap_ms
                and not after.is_scene_cut
                and str(before.display_state) == "OBSERVED"
                and str(after.display_state) == "OBSERVED"
                and before.identity_state == after.identity_state
                and before.fighter_id == after.fighter_id
            )
            if can_interpolate:
                ratio = age / (after.timestamp_ms - before.timestamp_ms)
                values = (
                    _bbox_values(before.bbox) * (1 - ratio)
                    + _bbox_values(after.bbox) * ratio
                )
                points = {
                    name: Keypoint(
                        point.x + (other.x - point.x) * ratio,
                        point.y + (other.y - point.y) * ratio,
                        min(point.score, other.score),
                    )
                    for name, point in before.keypoints.items()
                    if (other := after.keypoints.get(name)) is not None
                    and min(point.score, other.score) >= 0.25
                }
                result.append(
                    replace(
                        before,
                        timestamp_ms=timestamp_ms,
                        bbox=BBox(
                            *values, score=min(before.bbox.score, after.bbox.score)
                        ),
                        keypoints=points,
                        display_state=DisplayState.INTERPOLATED,
                    )
                )
            elif age <= min(100, self.max_gap_ms):
                # A single detector sample covers its output-frame interval, not
                # an unlimited tail. Predicted points are always absent.
                if (
                    str(before.display_state) == "PREDICTED"
                    and timestamp_ms - before.evidence_timestamp_ms > 1000
                ):
                    continue
                result.append(replace(before, timestamp_ms=timestamp_ms))
        return result


def sample_display_tracks(
    tracks: Sequence[DisplayTrack],
    timestamp_ms: int,
    frame_context: RenderFrameContext | None = None,
) -> list[DisplayTrack]:
    """Convenience wrapper; video loops should keep one DisplayTrackSampler."""
    return DisplayTrackSampler(tracks).sample(timestamp_ms, frame_context)
