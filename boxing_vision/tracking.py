"""Real shot-local motion tracks and a separate offline fighter-identity decoder.

The optional tracker dependency is loaded only when requested. Missing software
raises an actionable error; the legacy matcher is never labelled BoT-SORT.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .contracts import IdentityState, PoseObservation, ReviewStatus, SceneState
from .identity import IdentityMatch, part_appearance_distance
from .pose import RawPose, TwoFighterTracker


class TrackingBackendUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class TrackingFrame:
    frame_index: int
    timestamp_ms: int
    shot_id: int
    poses: list[RawPose]
    scene_state: str = "ACTIVE_FIGHT"
    is_scene_cut: bool = False
    tracker_predictions: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "shot_id": self.shot_id,
            "poses": [pose.to_dict() for pose in self.poses],
            "scene_state": str(self.scene_state),
            "is_scene_cut": self.is_scene_cut,
            "tracker_predictions": self.tracker_predictions,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrackingFrame:
        return cls(
            int(value["frame_index"]),
            int(value["timestamp_ms"]),
            int(value["shot_id"]),
            [RawPose.from_dict(pose) for pose in value["poses"]],
            str(value.get("scene_state", "UNCERTAIN")),
            bool(value.get("is_scene_cut", False)),
            list(value.get("tracker_predictions", [])),
        )


class ShotLocalBoTSORT:
    """Thin adapter over Roboflow's Apache-2.0 BoTSORTTracker with real scores."""

    name = "roboflow-botsort-cmc"

    def __init__(
        self,
        *,
        frame_rate: float = 15.0,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        activation_threshold: float = 0.7,
        lost_seconds: float = 1.0,
        enable_cmc: bool = True,
    ) -> None:
        if not 0.1 <= low_threshold < high_threshold <= activation_threshold <= 1.0:
            raise ValueError("Нужны 0.1 <= low < high <= activation <= 1")
        try:
            import supervision as sv
            from trackers import BoTSORTTracker
        except ImportError as exc:
            raise TrackingBackendUnavailable(
                "BoT-SORT недоступен. Установите project dependency trackers==2.6.0."
            ) from exc
        self._sv = sv
        self._tracker = BoTSORTTracker(
            frame_rate=frame_rate,
            high_conf_det_threshold=high_threshold,
            track_activation_threshold=activation_threshold,
            lost_track_buffer=round(lost_seconds * 30),
            minimum_consecutive_frames=2,
            instant_first_frame_activation=False,
            enable_cmc=enable_cmc,
            cmc_method="sparseOptFlow",
            cmc_downscale=2,
        )
        self.low_threshold = low_threshold
        self._shot_id: int | None = None
        self.last_diagnostics: list[dict[str, object]] = []
        self._tracklets: dict[tuple[int, int], dict[str, object]] = {}
        self._predictions: list[dict[str, object]] = []

    @property
    def predictions(self) -> list[dict[str, object]]:
        """Kalman/CMC snapshot only: reading it must never advance motion state."""
        return [dict(item) for item in self._predictions]

    @property
    def diagnostics(self) -> list[dict[str, object]]:
        return list(self.last_diagnostics)

    @property
    def tracklets(self) -> list[dict[str, object]]:
        return [dict(record) for record in self._tracklets.values()]

    def reset(self) -> None:
        self._tracker.reset()
        self._shot_id = None
        self.last_diagnostics = []
        self._predictions = []

    def update(
        self,
        poses: Sequence[RawPose],
        frame: np.ndarray,
        timestamp_ms: int,
        shot_id: int,
    ) -> list[RawPose]:
        if self._shot_id != shot_id:
            self._tracker.reset()
            self._shot_id = int(shot_id)
        for pose in poses:
            if pose.detector_confidence is None or not np.isfinite(
                pose.detector_confidence
            ):
                raise ValueError(
                    "BoT-SORT требует настоящий detector_confidence для каждого bbox"
                )
        indices = [
            index
            for index, pose in enumerate(poses)
            if float(pose.detector_confidence) >= self.low_threshold
            and pose.bbox.area > 0
        ]
        boxes = np.asarray(
            [
                [
                    poses[index].bbox.x1,
                    poses[index].bbox.y1,
                    poses[index].bbox.x2,
                    poses[index].bbox.y2,
                ]
                for index in indices
            ],
            dtype=np.float32,
        ).reshape(-1, 4)
        detections = self._sv.Detections(
            xyxy=boxes,
            confidence=np.asarray(
                [poses[index].detector_confidence for index in indices],
                dtype=np.float32,
            ),
            class_id=np.zeros(len(indices), dtype=int),
            data={"candidate_index": np.asarray(indices, dtype=int)},
        )
        tracked = self._tracker.update(
            detections, frame=frame, timestamp=timestamp_ms / 1000.0
        )
        assigned: dict[int, int] = {}
        if tracked.tracker_id is not None:
            for index, track_id in zip(
                tracked.data.get("candidate_index", []), tracked.tracker_id
            ):
                if int(track_id) >= 0:
                    assigned[int(index)] = int(track_id)
        result = [
            replace(pose, source_track_id=assigned.get(index))
            for index, pose in enumerate(poses)
        ]
        self._predictions = []
        for track in self._tracker.tracks:
            source = getattr(track, "tracker_id", None)
            if source is None or int(source) < 0:
                continue
            age_ms = round(float(track.time_since_update_seconds) * 1000)
            box = np.asarray(track.get_state_bbox(), dtype=float).reshape(-1)
            if (
                not 0 < age_ms <= 1000
                or box.size != 4
                or not np.all(np.isfinite(box))
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                continue
            self._predictions.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "evidence_timestamp_ms": timestamp_ms - age_ms,
                    "age_ms": age_ms,
                    "shot_id": shot_id,
                    "source_track_id": int(source),
                    "bbox": dict(zip(("x1", "y1", "x2", "y2"), box.tolist())),
                }
            )
        self.last_diagnostics = []
        for index, pose in enumerate(result):
            track_id = assigned.get(index)
            self.last_diagnostics.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "shot_id": shot_id,
                    "candidate_index": index,
                    "source_track_id": track_id,
                    "detector_bbox": pose.bbox.to_dict(),
                    "detector_confidence": pose.detector_confidence,
                    "motion_status": "tracked"
                    if track_id is not None
                    else "unconfirmed",
                }
            )
            if track_id is not None:
                record = self._tracklets.setdefault(
                    (shot_id, track_id),
                    {
                        "tracklet_id": f"shot-{shot_id}-track-{track_id}",
                        "shot_id": shot_id,
                        "source_track_id": track_id,
                        "start_ms": timestamp_ms,
                        "end_ms": timestamp_ms,
                        "frames": 0,
                    },
                )
                record["end_ms"] = timestamp_ms
                record["frames"] = int(record["frames"]) + 1
        return result


class OfflineIdentityDecoder:
    """Joint A/B Viterbi assignment inside a shot, with explicit unmatched states.

    The decoder never invents observations across missing detections. Its motion
    continuity term uses raw matches or bounded, nearby enrolled evidence; it
    cannot override negative-gallery, ring, or explicit user rejection.
    """

    version = "track-graph-v3"

    def __init__(
        self,
        identity_tracker: TwoFighterTracker,
        *,
        evidence_radius_ms: int = 1500,
        segment_identity_overrides: Mapping[str, str] | None = None,
    ) -> None:
        if not 0 <= evidence_radius_ms <= 1500:
            raise ValueError("Радиус identity evidence должен быть от 0 до 1500 мс")
        self.identity_tracker = identity_tracker
        self.evidence_radius_ms = int(evidence_radius_ms)
        self.diagnostics: list[dict[str, object]] = []
        self.tracklets: list[dict[str, object]] = []
        self.segment_identity_overrides = {
            str(key): IdentityState(value)
            for key, value in (segment_identity_overrides or {}).items()
        }
        self._segment_metadata: dict[tuple[int, str], dict[str, object]] = {}

    def _graph_matches(self, frames: Sequence[TrackingFrame]):
        """Split physical ambiguity locally and attach only evidence-backed edges.

        A continuously observed, non-overlapping trajectory can retain its
        enrolled identity through missing colour parts. Transfer to a different
        segment is a separate, bounded decision with actual appearance evidence.
        No source ID, screen side or box size establishes a fighter by itself.
        """
        tracker = self.identity_tracker
        states = (IdentityState.FIGHTER_A, IdentityState.FIGHTER_B)
        raw = [
            {
                str(p.source_track_id): tracker.score_candidate(p, shot_id=f.shot_id)
                for p in f.poses
                if p.source_track_id is not None
            }
            for f in frames
        ]
        # A single colour-negative can be caused by a dark torso turning to the
        # camera. Only persistent appearance negatives are a physical barrier;
        # user rejection and the supplied ROI remain immediate hard barriers.
        negative_history = defaultdict(lambda: deque(maxlen=5))
        previous_sources: set[str] = set()
        for index, frame in enumerate(frames):
            if frame.is_scene_cut or frame.scene_state != SceneState.ACTIVE_FIGHT:
                negative_history.clear()
            for absent in previous_sources - set(raw[index]):
                negative_history.pop(absent, None)
            for source, match in list(raw[index].items()):
                history = negative_history[source]
                is_negative = match.reason == "negative_gallery_closer"
                history.append(is_negative)
                if is_negative and sum(history) < 3:
                    raw[index][source] = replace(
                        match,
                        state=IdentityState.UNKNOWN,
                        accepted=False,
                        reason="negative_gallery_pending",
                    )
            previous_sources = set(raw[index])
        self._raw_graph_matches = raw
        pooled = self._pooled_matches(frames)
        groups = defaultdict(list)
        for index, frame in enumerate(frames):
            for pose in frame.poses:
                if pose.source_track_id is None:
                    continue
                overlap = any(
                    other is not pose
                    and 0.3 <= other.bbox.area / max(pose.bbox.area, 1) <= 3
                    and tracker._bbox_iou(other.bbox, pose.bbox) > 0.35
                    for other in frame.poses
                )
                groups[str(pose.source_track_id)].append((index, pose, overlap))

        segments = []
        self._segment_metadata = {}
        for source, samples in groups.items():
            components = []
            previous = None
            for item in samples:
                index, pose, overlap = item
                match = raw[index][source]
                hard = (
                    frames[index].scene_state != SceneState.ACTIVE_FIGHT
                    or match.state == IdentityState.OTHER
                    or (match.reason == "user_override" and not match.accepted)
                )
                reason = "start"
                if previous is not None:
                    pi, pp, po, ph = previous
                    reason = (
                        "scene_barrier"
                        if any(
                            f.is_scene_cut or f.scene_state != frames[pi].scene_state
                            for f in frames[pi + 1 : index + 1]
                        )
                        else "identity_barrier"
                        if hard != ph
                        else "detection_gap"
                        if frames[index].timestamp_ms - frames[pi].timestamp_ms > 250
                        else "geometry_jump"
                        if not self._geometry_continuous(pp, pose)
                        else "occlusion_boundary"
                        if po != overlap
                        else ""
                    )
                if reason:
                    components.append((reason, []))
                components[-1][1].append(item)
                previous = (index, pose, overlap, hard)

            for reason, component in components:
                # A stable opposite 3/5 run starts a new segment at its first
                # vote, not a retrospective ban on every earlier observation.
                boundaries = {0: reason}
                stable = None
                history = deque(maxlen=5)
                for position, (index, _, _) in enumerate(component):
                    match = raw[index][source]
                    vote = match.state if match.accepted else None
                    history.append((position, vote))
                    if (
                        vote in states
                        and sum(
                            v == vote and index - component[p][0] <= 4
                            for p, v in history
                        )
                        >= 3
                    ):
                        if stable is not None and stable != vote:
                            start = next(
                                p
                                for p, v in history
                                if v == vote and index - component[p][0] <= 4
                            )
                            boundaries[start] = "identity_conflict_boundary"
                            history = deque(
                                ((p, v) for p, v in history if p >= start), maxlen=5
                            )
                        stable = vote
                offsets = sorted(boundaries) + [len(component)]
                for start, end in itertools.pairwise(offsets):
                    items = component[start:end]
                    first = items[0][0]
                    segment_id = f"shot-{frames[first].shot_id}-track-{source}-segment-{frames[first].timestamp_ms}"
                    segments.append(
                        {
                            "id": segment_id,
                            "source": source,
                            "items": items,
                            "reason": boundaries[start],
                            "physical": segment_id,
                            "edge": None,
                        }
                    )

        segments.sort(key=lambda item: (item["items"][0][0], item["source"]))
        known_ids = {segment["id"] for segment in segments}
        # Decoder is called once per shot; reject unknown IDs for this shot only.
        prefix = f"shot-{frames[0].shot_id}-"
        invalid = [
            key
            for key in self.segment_identity_overrides
            if key.startswith(prefix) and key not in known_ids
        ]
        if invalid:
            raise ValueError(f"Неизвестный segment_id: {invalid[0]}")
        output = [{} for _ in frames]
        completed = []
        for segment in segments:
            source, items = segment["source"], segment["items"]
            override = self.segment_identity_overrides.get(segment["id"])
            label = None
            history = deque(maxlen=5)
            contrary_history = deque(maxlen=5)
            anchor = None
            anchor_position = None
            for index, _, _ in items:
                match = raw[index][source]
                history.append((index, match.state if match.accepted else None))
                if (
                    match.accepted
                    and sum(
                        state == match.state and index - observed_index <= 4
                        for observed_index, state in history
                    )
                    >= 3
                ):
                    label, anchor = match.state, match
                    anchor_position = next(
                        position
                        for position, item in enumerate(items)
                        if item[0] == index
                    )
                    break
            segment["label"] = label
            segment["anchor"] = anchor
            first_index, first_pose, first_overlap = items[0]
            first_match = raw[first_index][source]
            # User edits apply to this segment alone, and are never used as an
            # appearance template to label another segment implicitly.
            link = None
            if (
                override is None
                and not first_overlap
                and frames[first_index].scene_state == SceneState.ACTIVE_FIGHT
                and first_match.state != IdentityState.OTHER
                and first_match.reason != "user_override"
                and segment["reason"]
                not in {
                    "identity_barrier",
                    "identity_conflict_boundary",
                    "scene_barrier",
                }
            ):
                candidates = []
                for before in completed:
                    bi, bp, bo = before["items"][-1]
                    gap = frames[first_index].timestamp_ms - frames[bi].timestamp_ms
                    if (
                        not 0 < gap <= self.evidence_radius_ms
                        or before.get("last_anchor") is None
                        or bo
                        or before.get("override") is not None
                        or before.get("successor") is not None
                        or (label is not None and label != before["last_anchor"].state)
                        or any(
                            f.is_scene_cut or f.scene_state != SceneState.ACTIVE_FIGHT
                            for f in frames[bi + 1 : first_index + 1]
                        )
                    ):
                        continue
                    # An old ancestor cannot be revisited after this measured
                    # trajectory crossed an explicit identity barrier. Checking
                    # just the endpoints would jump over OTHER during a clinch.
                    expected = before["last_anchor"].state
                    if any(
                        match is not None
                        and (
                            match.state == IdentityState.OTHER
                            or match.reason == "user_override"
                            or (match.accepted and match.state != expected)
                        )
                        for index in range(bi + 1, first_index + 1)
                        for candidate_source in {before["source"], source}
                        for match in [raw[index].get(candidate_source)]
                    ):
                        continue
                    if not self._geometry_continuous(bp, first_pose):
                        continue
                    # Compare actual visible endpoints within the bounded gap
                    # neighbourhood. Missing parts alone never create an edge.
                    appearance = []
                    for ni, np_, _ in items:
                        if (
                            frames[ni].timestamp_ms - frames[first_index].timestamp_ms
                            > 1000
                        ):
                            break
                        for oi, op, _ in before["items"][-5:]:
                            if (
                                frames[ni].timestamp_ms - frames[oi].timestamp_ms
                                > self.evidence_radius_ms
                            ):
                                continue
                            value = part_appearance_distance(
                                op.appearance_parts,
                                np_.appearance_parts,
                                allow_partial=True,
                            )
                            if (
                                value is None
                                and not op.appearance_parts
                                and not np_.appearance_parts
                            ):
                                from .identity import appearance_distance

                                if not tracker.gallery._core_parts:
                                    value = appearance_distance(
                                        op.appearance, np_.appearance
                                    )
                            if value is not None:
                                appearance.append(value)
                    if (
                        not appearance
                        or float(np.median(appearance)) > tracker.gallery.max_distance
                    ):
                        continue
                    displacement = float(
                        np.linalg.norm(
                            np.asarray(first_pose.bbox.center)
                            - np.asarray(bp.bbox.center)
                        )
                    )
                    cost = 0.65 * float(
                        np.median(appearance)
                    ) + 0.35 * displacement / max(
                        bp.bbox.height, first_pose.bbox.height, 1
                    )
                    candidates.append((cost, before))
                candidates.sort(key=lambda item: item[0])
                if candidates and (
                    len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= 0.12
                ):
                    cost, link = candidates[0]
                    link["successor"] = segment["id"]
                    segment["physical"] = link["physical"]
                    segment["edge"] = {
                        "predecessor_segment_id": link["id"],
                        "cost": cost,
                        "kind": "appearance_motion",
                        "gap_ms": frames[first_index].timestamp_ms
                        - frames[link["items"][-1][0]].timestamp_ms,
                    }

            # Physical continuity is not an identity decision. Retain the
            # provenance of a real, continuously observed source across an
            # occlusion boundary even when neither side has enough appearance
            # for a coloured A/B box. The display layer can show it neutrally.
            if (
                link is None
                and override is None
                and segment["reason"]
                not in {
                    "identity_barrier",
                    "identity_conflict_boundary",
                    "scene_barrier",
                    "geometry_jump",
                }
                and first_match.state != IdentityState.OTHER
                and frames[first_index].scene_state == SceneState.ACTIVE_FIGHT
            ):
                for before in reversed(completed):
                    if before["source"] != source:
                        continue
                    bi, bp, _ = before["items"][-1]
                    if (
                        0
                        < frames[first_index].timestamp_ms - frames[bi].timestamp_ms
                        <= 250
                        and self._geometry_continuous(bp, first_pose)
                        and raw[bi][source].state != IdentityState.OTHER
                        and before.get("override") is None
                        and (
                            label is None
                            or before.get("physical_identity") is None
                            or label == before["physical_identity"]
                        )
                        and not any(
                            f.is_scene_cut or f.scene_state != SceneState.ACTIVE_FIGHT
                            for f in frames[bi + 1 : first_index + 1]
                        )
                    ):
                        segment["physical"] = before["physical"]
                        before["successor"] = segment["id"]
                        segment["edge"] = {
                            "predecessor_segment_id": before["id"],
                            "kind": "physical_continuity",
                            "identity_transferred": False,
                            "gap_ms": frames[first_index].timestamp_ms
                            - frames[bi].timestamp_ms,
                        }
                        segment["physical_identity"] = before.get("physical_identity")
                    break

            segment["physical_identity"] = (
                label
                or segment.get("physical_identity")
                or (link.get("physical_identity") if link is not None else None)
            )

            history = deque(maxlen=5)
            active_anchor = link.get("last_anchor") if link is not None else None
            previous_pose = None
            last_strong_parts = None
            segment["override"] = override
            for index, pose, overlap in items:
                match = raw[index][source]
                effective = match
                hard = (
                    frames[index].scene_state != SceneState.ACTIVE_FIGHT
                    or match.state == IdentityState.OTHER
                    or (match.reason == "user_override" and not match.accepted)
                )
                if override is not None:
                    effective = IdentityMatch(
                        override, 0, 1, 1, override in states, "segment_user_override"
                    )
                    active_anchor = None
                elif hard:
                    active_anchor = None
                    history.clear()
                    contrary_history.clear()
                else:
                    history.append((index, match.state if match.accepted else None))
                    if (
                        match.accepted
                        and sum(
                            state == match.state and index - observed_index <= 4
                            for observed_index, state in history
                        )
                        >= 3
                    ):
                        active_anchor = match
                        last_strong_parts = pose.appearance_parts
                    if active_anchor is not None and not match.accepted and not overlap:
                        distances, _ = tracker.gallery.distances(
                            pose.appearance, pose.appearance_parts
                        )
                        other = (
                            states[1] if active_anchor.state == states[0] else states[0]
                        )
                        # A genuinely closer opponent is contradictory evidence.
                        # Missing/incomparable parts are not assigned distance=0.
                        partial = part_appearance_distance(
                            pose.appearance_parts,
                            pose.appearance_parts,
                            allow_partial=True,
                        )
                        comparable = partial is not None
                        contradiction = (
                            comparable
                            and distances[other] <= tracker.gallery.max_distance
                            and distances[active_anchor.state] - distances[other]
                            >= tracker.gallery.min_margin
                        )
                        local_distance = (
                            part_appearance_distance(
                                previous_pose.appearance_parts,
                                pose.appearance_parts,
                                allow_partial=True,
                            )
                            if previous_pose is not None
                            else None
                        )
                        anchor_distance = (
                            part_appearance_distance(
                                last_strong_parts,
                                pose.appearance_parts,
                                allow_partial=True,
                            )
                            if last_strong_parts
                            else None
                        )
                        appearance_jump = (
                            local_distance is not None
                            and local_distance > tracker.gallery.max_distance
                        )
                        contrary_history.append((index, contradiction))
                        if contradiction:
                            effective = replace(
                                match,
                                state=IdentityState.UNKNOWN,
                                accepted=False,
                                reason="identity_contradiction_pending",
                            )
                            if (
                                sum(
                                    contrary and index - observed_index <= 4
                                    for observed_index, contrary in contrary_history
                                )
                                >= 3
                            ):
                                active_anchor = None
                                history.clear()
                        elif appearance_jump:
                            active_anchor = None
                            history.clear()
                        elif previous_pose is not None and self._geometry_continuous(
                            previous_pose, pose
                        ):
                            # No unbounded chain of colour adaptation: continuous
                            # observations are the evidence, with immutable anchor.
                            effective = IdentityMatch(
                                active_anchor.state,
                                active_anchor.distance,
                                active_anchor.margin,
                                active_anchor.negative_distance,
                                True,
                                "observed_continuity",
                            )
                        elif (
                            link is not None
                            and anchor_distance is not None
                            and anchor_distance <= tracker.gallery.max_distance
                        ):
                            effective = IdentityMatch(
                                active_anchor.state,
                                active_anchor.distance,
                                active_anchor.margin,
                                active_anchor.negative_distance,
                                True,
                                "source_reacquired",
                            )
                    elif (
                        active_anchor is not None
                        and match.accepted
                        and match.state != active_anchor.state
                    ):
                        contrary_history.append((index, True))
                        effective = replace(
                            match,
                            state=IdentityState.UNKNOWN,
                            accepted=False,
                            reason="identity_contradiction_pending",
                        )
                        if (
                            sum(
                                contrary and index - observed_index <= 4
                                for observed_index, contrary in contrary_history
                            )
                            >= 3
                        ):
                            active_anchor = None
                            history.clear()
                    else:
                        contrary_history.append((index, False))
                    if effective is match and not match.accepted:
                        pooled_match = pooled[index].get(source)
                        if pooled_match is not None and (
                            label is None or pooled_match.state == label
                        ):
                            effective = pooled_match
                if effective is not match or match.reason == "negative_gallery_pending":
                    output[index][source] = effective
                self._segment_metadata[index, source] = {
                    "segment_id": segment["id"],
                    "physical_track_id": segment["physical"],
                    "segment_reason": segment["reason"],
                    "link_evidence": segment["edge"],
                    "identity_origin": "user_confirmed"
                    if override is not None
                    else "reacquired"
                    if link is not None
                    else "continuous"
                    if effective.reason == "observed_continuity"
                    else "enrollment",
                }
                previous_pose = pose
            segment["last_anchor"] = active_anchor
            # Offline inference may use a later *real* 3/5 enrollment match on
            # the same uninterrupted, non-overlapping measured segment. The
            # 1500ms limit constrains transfers over gaps, not the lifetime of
            # a person who stays observed. Do not cross even one missing part,
            # competing role, OTHER, override or incompatible local transition.
            if (
                override is None
                and anchor is not None
                and anchor_position is not None
                and not any(overlap for _, _, overlap in items)
            ):
                next_index, next_pose, _ = items[anchor_position]
                for position in range(anchor_position - 1, -1, -1):
                    index, pose, _ = items[position]
                    match = raw[index][source]
                    competing = any(
                        other_source != source
                        and other_match.accepted
                        and other_match.state == anchor.state
                        for other_source, other_match in raw[index].items()
                    )
                    local_distance = part_appearance_distance(
                        pose.appearance_parts,
                        next_pose.appearance_parts,
                        allow_partial=True,
                    )
                    if (
                        match.state == IdentityState.OTHER
                        or match.reason == "user_override"
                        or (match.accepted and match.state != anchor.state)
                        or competing
                        or local_distance is None
                        or local_distance > tracker.gallery.max_distance
                        or frames[next_index].timestamp_ms - frames[index].timestamp_ms
                        > 250
                        or not self._geometry_continuous(pose, next_pose)
                    ):
                        break
                    if not match.accepted:
                        output[index][source] = IdentityMatch(
                            anchor.state,
                            anchor.distance,
                            anchor.margin,
                            anchor.negative_distance,
                            True,
                            "offline_tracklet_evidence",
                        )
                        self._segment_metadata[index, source].update(
                            identity_origin="offline_confirmed",
                            evidence_timestamp_ms=frames[
                                items[anchor_position][0]
                            ].timestamp_ms,
                        )
                    next_index, next_pose = index, pose
            completed.append(segment)
        return output

    def _pooled_matches(
        self, frames: Sequence[TrackingFrame]
    ) -> list[dict[str, IdentityMatch]]:
        """Borrow nearby strong evidence, never the identity of an entire shot.

        Every intervening detection must retain visible, coherent body features.
        A contradictory strong role splits the evidence window rather than
        poisoning all earlier and later samples of the same motion track.
        """
        tracker = self.identity_tracker
        output: list[dict[str, IdentityMatch]] = [{} for _ in frames]
        groups = defaultdict(list)
        for index, frame in enumerate(frames):
            if frame.scene_state != SceneState.ACTIVE_FIGHT:
                continue
            for pose in frame.poses:
                if pose.source_track_id is not None:
                    match = tracker.score_candidate(pose, shot_id=frame.shot_id)
                    groups[str(pose.source_track_id)].append((index, pose, match))
        for source, samples in groups.items():
            components: list[list[tuple[int, RawPose, IdentityMatch]]] = []
            previous_item = None
            for item in samples:
                frame_index, pose, match = item
                visible = (
                    part_appearance_distance(
                        pose.appearance_parts, pose.appearance_parts
                    )
                    is not None
                )
                if (
                    match.state == IdentityState.OTHER
                    or match.reason == "user_override"
                    or not visible
                ):
                    previous_item = None
                    continue
                connected = False
                if previous_item is not None:
                    previous_index, previous, _ = previous_item
                    distance = part_appearance_distance(
                        previous.appearance_parts, pose.appearance_parts
                    )
                    connected = bool(
                        frame_index == previous_index + 1
                        and not frames[frame_index].is_scene_cut
                        and frames[frame_index].timestamp_ms
                        - frames[previous_index].timestamp_ms
                        <= 200
                        and self._geometry_continuous(previous, pose)
                        and distance is not None
                        and distance <= tracker.gallery.max_distance
                    )
                if not connected:
                    components.append([])
                components[-1].append(item)
                previous_item = item
            for component in components:
                for position, (frame_index, pose, raw_match) in enumerate(component):
                    if raw_match.accepted:
                        continue
                    distances, _ = tracker.gallery.distances(
                        pose.appearance, pose.appearance_parts
                    )
                    ordered = sorted(distances.items(), key=lambda item: item[1])
                    state, own_distance = ordered[0]
                    if own_distance >= min(1.0, ordered[1][1]):
                        continue
                    # Never cross a strong contradictory role, even when the
                    # motion ID stays unchanged. Genuine swaps require review.
                    neighbors = [(frame_index, raw_match)]
                    for direction in (-1, 1):
                        cursor = position + direction
                        while 0 <= cursor < len(component):
                            index, _, match = component[cursor]
                            if abs(
                                frames[index].timestamp_ms
                                - frames[frame_index].timestamp_ms
                            ) > self.evidence_radius_ms or (
                                match.accepted and match.state != state
                            ):
                                break
                            neighbors.append((index, match))
                            cursor += direction
                    strong = sorted(
                        (index, match)
                        for index, match in neighbors
                        if match.accepted and match.state == state
                    )
                    # Count consecutive input-frame positions, not a filtered
                    # list in which three sparse votes could look consecutive.
                    if not any(
                        strong[i + 2][0] - strong[i][0] <= 4
                        for i in range(len(strong) - 2)
                    ):
                        continue
                    evidence = [match for _, match in strong]
                    output[frame_index][source] = IdentityMatch(
                        state,
                        float(np.median([m.distance for m in evidence])),
                        float(np.median([m.margin for m in evidence])),
                        float(np.median([m.negative_distance for m in evidence])),
                        True,
                        "tracklet_evidence",
                    )
        return output

    @staticmethod
    def _geometry_continuous(previous: RawPose, current: RawPose) -> bool:
        """A reused motion ID may not bridge a large bbox jump/scale change."""
        if min(previous.bbox.area, current.bbox.area) <= 0:
            return False
        area_ratio = current.bbox.area / previous.bbox.area
        displacement = float(
            np.linalg.norm(
                np.asarray(current.bbox.center) - np.asarray(previous.bbox.center)
            )
        )
        return bool(
            0.4 <= area_ratio <= 2.5
            and displacement <= 0.5 * max(previous.bbox.height, current.bbox.height)
        )

    def decode(self, frames: Sequence[TrackingFrame]) -> list[PoseObservation]:
        if not frames:
            return []
        if len({frame.shot_id for frame in frames}) != 1:
            raise ValueError("decode принимает один shot; разделите кадры по shot_id")
        if any(b.timestamp_ms <= a.timestamp_ms for a, b in itertools.pairwise(frames)):
            raise ValueError("Кадры shot должны иметь строго возрастающие timestamps")
        tracker = self.identity_tracker
        fighter_ids = tracker.fighter_ids
        pooled_matches = self._graph_matches(frames)
        costs: list[dict[tuple[int | None, int | None], float]] = []
        backpointers: list[
            dict[tuple[int | None, int | None], tuple[int | None, int | None]]
        ] = []
        for time_index, frame in enumerate(frames):
            options: list[list[int | None]] = [[None], [None]]
            distances: dict[int, float] = {}
            if frame.scene_state == SceneState.ACTIVE_FIGHT:
                for role_index, state in enumerate(
                    (IdentityState.FIGHTER_A, IdentityState.FIGHTER_B)
                ):
                    matches = []
                    for index, pose in enumerate(frame.poses):
                        match = pooled_matches[time_index].get(
                            str(pose.source_track_id)
                        ) or tracker.score_candidate(pose, shot_id=frame.shot_id)
                        if (
                            pose.source_track_id is not None
                            and match.accepted
                            and match.state == state
                        ):
                            matches.append((match.distance, index))
                            distances[index] = match.distance
                    options[role_index].extend(
                        index for _, index in sorted(matches)[:2]
                    )
            states = [
                state
                for state in itertools.product(*options)
                if state[0] is None or state[1] is None or state[0] != state[1]
            ]
            emissions = {
                state: sum(
                    0.40 if index is None else distances[index] for index in state
                )
                for state in states
            }
            current: dict[tuple[int | None, int | None], float] = {}
            links = {}
            for state in states:
                if not time_index:
                    current[state] = emissions[state]
                    continue

                def transition(
                    previous: tuple[int | None, int | None],
                    selected=state,
                    before=frames[time_index - 1],
                    after=frame,
                ) -> float:
                    penalty = 0.0
                    for old_index, new_index in zip(previous, selected):
                        if old_index is None or new_index is None:
                            penalty += 0.02 if old_index != new_index else 0
                        elif (
                            before.poses[old_index].source_track_id
                            != after.poses[new_index].source_track_id
                        ):
                            penalty += 0.12
                    return penalty

                best = min(
                    costs[-1],
                    key=lambda previous: costs[-1][previous] + transition(previous),
                )
                current[state] = emissions[state] + costs[-1][best] + transition(best)
                links[state] = best
            costs.append(current)
            backpointers.append(links)
        path = [min(costs[-1], key=costs[-1].get)]
        for index in range(len(frames) - 1, 0, -1):
            path.append(backpointers[index][path[-1]])
        path.reverse()
        tracker.reset(keep_appearance=True)
        histories = {role: deque(maxlen=5) for role in fighter_ids}
        observations: list[PoseObservation] = []
        self.diagnostics = []
        tracks: dict[str, list[dict[str, object]]] = defaultdict(list)
        for time_index, (frame, selection) in enumerate(zip(frames, path)):
            temporal_break = bool(
                frame.is_scene_cut
                or frame.scene_state != SceneState.ACTIVE_FIGHT
                or (
                    time_index
                    and (
                        frames[time_index - 1].scene_state != frame.scene_state
                        or frame.timestamp_ms - frames[time_index - 1].timestamp_ms
                        > 200
                    )
                )
            )
            if temporal_break:
                for history in histories.values():
                    history.clear()
            roles: dict[str, str] = {}
            visible_sources = {
                str(pose.source_track_id): pose
                for pose in frame.poses
                if pose.source_track_id is not None
            }
            current_matches = {
                source: tracker.score_candidate(pose, shot_id=frame.shot_id)
                for source, pose in visible_sources.items()
            }
            for source, match in self._raw_graph_matches[time_index].items():
                if match.reason == "negative_gallery_pending":
                    current_matches[source] = match
            for source, effective in pooled_matches[time_index].items():
                if effective.reason in {
                    "segment_user_override",
                    "identity_contradiction_pending",
                }:
                    current_matches[source] = effective
            safe_pooled_matches = {}
            for source, pooled in pooled_matches[time_index].items():
                pose = visible_sources.get(source)
                raw = current_matches.get(source)
                if pooled.reason in {
                    "segment_user_override",
                    "negative_gallery_pending",
                    "identity_contradiction_pending",
                }:
                    safe_pooled_matches[source] = pooled
                    continue
                if (
                    raw is None
                    or raw.state == IdentityState.OTHER
                    or raw.reason == "user_override"
                    or (raw.accepted and raw.state != pooled.state)
                ):
                    continue
                distances, _ = tracker.gallery.distances(
                    pose.appearance, pose.appearance_parts
                )
                other = (
                    IdentityState.FIGHTER_B
                    if pooled.state == IdentityState.FIGHTER_A
                    else IdentityState.FIGHTER_A
                )
                if (
                    pooled.reason == "segment_user_override"
                    or (
                        pooled.reason
                        in {
                            "observed_continuity",
                            "source_reacquired",
                            "offline_tracklet_evidence",
                        }
                        and not raw.accepted
                        and (
                            tracker.gallery._core_parts
                            or distances[pooled.state] <= distances[other]
                        )
                        and (
                            part_appearance_distance(
                                pose.appearance_parts,
                                pose.appearance_parts,
                                allow_partial=True,
                            )
                            is None
                            or (
                                distances[pooled.state] - distances[other]
                                < tracker.gallery.min_margin
                                or distances[other] > tracker.gallery.max_distance
                            )
                        )
                    )
                    or distances.get(pooled.state, 1.0) < min(1.0, distances[other])
                ):
                    safe_pooled_matches[source] = pooled
            for role_index, (role, candidate_index) in enumerate(
                zip(fighter_ids, selection)
            ):
                history = histories[role]
                expected_state = (IdentityState.FIGHTER_A, IdentityState.FIGHTER_B)[
                    role_index
                ]
                # A weak *present* observation is an empty vote, not a reset of
                # every earlier vote. Absence/OTHER/contradiction remains a hard
                # reset; a source can never inherit another source's votes.
                # Check EVERY retained source, not just the latest contender:
                # otherwise [old, old, new, new, old] can reuse stale votes after
                # old disappeared while the decoder considered another person.
                for previous_source in set(history) - {None}:
                    pose = visible_sources.get(previous_source)
                    previous_pose = (
                        next(
                            (
                                previous
                                for previous in frames[time_index - 1].poses
                                if str(previous.source_track_id) == previous_source
                            ),
                            None,
                        )
                        if time_index
                        else None
                    )
                    match = current_matches.get(previous_source)
                    if (
                        match is None
                        or previous_pose is None
                        or not self._geometry_continuous(previous_pose, pose)
                        or match.state == IdentityState.OTHER
                        or (match.accepted and match.state != expected_state)
                        or (match.reason == "user_override" and not match.accepted)
                        or (
                            self._segment_metadata.get(
                                (time_index, previous_source), {}
                            ).get("segment_id")
                            != self._segment_metadata.get(
                                (time_index - 1, previous_source), {}
                            ).get("segment_id")
                            and not (
                                self._segment_metadata.get(
                                    (time_index, previous_source), {}
                                ).get("physical_track_id")
                                == self._segment_metadata.get(
                                    (time_index - 1, previous_source), {}
                                ).get("physical_track_id")
                                and match.accepted
                                and match.state == expected_state
                            )
                        )
                        or (
                            pose.appearance_parts
                            and part_appearance_distance(
                                pose.appearance_parts, pose.appearance_parts
                            )
                            is None
                            and previous_source not in safe_pooled_matches
                        )
                        or (
                            not pose.appearance_parts
                            and not match.accepted
                            and previous_source not in safe_pooled_matches
                        )
                    ):
                        for position, value in enumerate(history):
                            if value == previous_source:
                                history[position] = None
                token = (
                    str(frame.poses[candidate_index].source_track_id)
                    if candidate_index is not None
                    else None
                )
                effective = safe_pooled_matches.get(token) or current_matches.get(token)
                if (
                    effective is None
                    or not effective.accepted
                    or effective.state != expected_state
                ):
                    token = None
                history.append(token)
                if token is None:
                    continue
                if Counter(history)[token] >= 3:
                    roles[token] = role
            # Voting is already performed above. Reusing process produces the
            # same confidence, visibility and contamination guards as live use.
            frame_observations = tracker.process(
                frame.frame_index,
                frame.timestamp_ms,
                frame.poses,
                scene_cut=temporal_break,
                shot_id=frame.shot_id,
                active_fight=frame.scene_state == SceneState.ACTIVE_FIGHT,
                decoded_source_roles=roles,
                identity_votes_confirmed=True,
                decoded_matches=safe_pooled_matches,
            )
            observations.extend(
                replace(
                    item,
                    scene_state=frame.scene_state,
                    is_scene_cut=frame.is_scene_cut,
                    **{
                        key: value
                        for key, value in self._segment_metadata.get(
                            (time_index, str(item.source_track_id)), {}
                        ).items()
                        if key in {"segment_id", "physical_track_id", "identity_origin"}
                    },
                )
                for item in frame_observations
            )
            for diagnostic in tracker.last_diagnostics:
                metadata = self._segment_metadata.get(
                    (time_index, str(diagnostic["source_track_id"])), {}
                )
                diagnostic = {
                    **diagnostic,
                    **metadata,
                    "scene_state": str(frame.scene_state),
                }
                self.diagnostics.append(diagnostic)
                if diagnostic["source_track_id"] is not None:
                    tracks[str(metadata["segment_id"])].append(diagnostic)
        self.tracklets = []
        conflicting_segments = set()
        for tracklet_id, samples in tracks.items():
            states = Counter(str(sample["identity_state"]) for sample in samples)
            conflicting = all(
                states[state]
                for state in (
                    str(IdentityState.FIGHTER_A),
                    str(IdentityState.FIGHTER_B),
                )
            )
            if conflicting:
                conflicting_segments.add(tracklet_id)
            state = states.most_common(1)[0][0]
            fighter_candidate = any(
                sample.get("selected_fighter_id") is not None
                or (
                    min(sample["a_distance"], sample["b_distance"])
                    <= tracker.gallery.max_distance
                    and sample["reason"]
                    not in {"outside_ring", "negative_gallery_closer"}
                )
                for sample in samples
            )
            has_confirmed = any(
                states[str(role)]
                for role in (IdentityState.FIGHTER_A, IdentityState.FIGHTER_B)
            )
            boundary_suspected = samples[0].get(
                "segment_reason"
            ) == "identity_conflict_boundary" and not all(
                sample.get("identity_origin") == "user_confirmed" for sample in samples
            )
            required_review = (
                conflicting
                or boundary_suspected
                or (not has_confirmed and fighter_candidate)
            )
            self.tracklets.append(
                {
                    "tracklet_id": tracklet_id,
                    "segment_id": tracklet_id,
                    "physical_track_id": samples[0].get("physical_track_id"),
                    "segment_reason": samples[0].get("segment_reason"),
                    "link_evidence": samples[0].get("link_evidence"),
                    "fighter_candidate": fighter_candidate,
                    "eligible_for_review": fighter_candidate,
                    "required_review": required_review,
                    "source_track_id": samples[0]["source_track_id"],
                    "shot_id": frames[0].shot_id,
                    "start_ms": samples[0]["timestamp_ms"],
                    "end_ms": samples[-1]["timestamp_ms"],
                    "frames": len(samples),
                    "identity_state": str(IdentityState.UNKNOWN)
                    if conflicting
                    else state,
                    "identity_confidence": float(
                        np.mean([sample["confidence"] for sample in samples])
                    ),
                    "identity_margin": float(
                        np.median([sample["margin"] for sample in samples])
                    ),
                    "identity_conflict": conflicting or boundary_suspected,
                    "conflict_boundary_only": boundary_suspected and not conflicting,
                    "review_status": str(
                        ReviewStatus.NEEDS_REVIEW
                        if required_review
                        else ReviewStatus.AUTO_CONFIRMED
                    ),
                }
            )
        if conflicting_segments:
            # Residual conflicts quarantine only the deterministic segment,
            # never unrelated good portions of the same reused motion source.
            observations = [
                item
                for item in observations
                if item.segment_id not in conflicting_segments
            ]
            for diagnostic in self.diagnostics:
                if diagnostic.get("segment_id") in conflicting_segments:
                    diagnostic.update(
                        identity_state=str(IdentityState.UNKNOWN),
                        selected_fighter_id=None,
                        confidence=0.0,
                        reason="segment_identity_conflict",
                    )
        return observations
