"""Ready-made punch classifier, isolated from identity and event detection.

The four-way source model has no background class and no contact classifier.
It is consequently used only to refine already proposed, identity-verified
events. Its softmax values are explicitly uncalibrated model scores.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .contracts import (
    IdentityState,
    Keypoint,
    PoseObservation,
    PunchEvent,
    ReviewStatus,
    SceneState,
)
from .model_registry import ACM_BUNDLE_ID, ACM_JOINTS, ACM_LABELS, validate_acm_bundle


@dataclass(frozen=True)
class PunchClassification:
    label: str
    score: float
    margin: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class ClassificationWindow:
    features: np.ndarray | None
    input_kind: str
    reason: str | None = None


class PunchClassifier(Protocol):
    def predict(self, features: np.ndarray) -> PunchClassification: ...


def normalize_acm_pose(keypoints: Mapping[str, Keypoint]) -> np.ndarray | None:
    """Source-exact 8 joint order, midpoint centering and shoulder-width scale.

    The source requires shoulders >= .5 but zeros individual joints <= .5.
    Keep that strict distinction instead of silently changing model inputs.
    """
    points = [keypoints.get(name) for name in ACM_JOINTS]
    shoulders = points[:2]
    if any(point is None or point.score < 0.5 for point in shoulders):
        return None
    if any(
        point is not None and not np.isfinite((point.x, point.y, point.score)).all()
        for point in points
    ):
        return None
    left, right = shoulders
    assert left is not None and right is not None
    left_xy = np.asarray((left.x, left.y), dtype=np.float32)
    right_xy = np.asarray((right.x, right.y), dtype=np.float32)
    scale = float(np.linalg.norm(left_xy - right_xy))
    if scale < 1e-4:
        return None
    center = (left_xy + right_xy) / 2
    normalized = np.zeros((8, 2), dtype=np.float32)
    for index, point in enumerate(points):
        if point is not None and point.score > 0.5:
            normalized[index] = (np.asarray((point.x, point.y), dtype=np.float32) - center) / scale
    return normalized.reshape(16)


def _confirmed(observation: PoseObservation) -> bool:
    expected = {
        "fighter_a": IdentityState.FIGHTER_A,
        "fighter_b": IdentityState.FIGHTER_B,
    }.get(observation.fighter_id)
    return bool(
        expected is not None and observation.identity_state == expected
        and observation.scene_state == SceneState.ACTIVE_FIGHT
        and float(observation.identity_confidence or 0) >= 0.55
        and float(observation.identity_margin if observation.identity_margin is not None else 1) >= 0.12
        and observation.review_status not in {ReviewStatus.NEEDS_REVIEW, ReviewStatus.REJECTED}
    )


class _WindowIndex:
    def __init__(self, observations: Sequence[PoseObservation]):
        self.by_fighter: dict[str, list[PoseObservation]] = defaultdict(list)
        for observation in observations:
            self.by_fighter[observation.fighter_id].append(observation)
        for values in self.by_fighter.values():
            values.sort(key=lambda value: value.timestamp_ms)
        self.times = {key: [obs.timestamp_ms for obs in values] for key, values in self.by_fighter.items()}

    def sample(self, fighter: str, timestamp: float) -> tuple[dict[str, Keypoint] | None, int | None, bool, str | None]:
        values, times = self.by_fighter.get(fighter, []), self.times.get(fighter, [])
        insertion = bisect_left(times, timestamp)
        neighbors = [i for i in (insertion - 1, insertion) if 0 <= i < len(values)]
        nearest_index = min(neighbors, key=lambda i: abs(times[i] - timestamp)) if neighbors else None
        if nearest_index is None:
            return None, None, False, "missing_fighter"
        nearest = values[nearest_index]
        if abs(nearest.timestamp_ms - timestamp) <= 8:
            if not _confirmed(nearest) or nearest.is_scene_cut:
                return None, None, False, "identity_or_scene_boundary"
            return nearest.keypoints, nearest.shot_id, True, None
        if insertion <= 0 or insertion >= len(values):
            return None, None, False, "incomplete_window"
        previous, following = values[insertion - 1], values[insertion]
        gap = following.timestamp_ms - previous.timestamp_ms
        if gap <= 0 or gap > 150:
            return None, None, False, "pose_gap"
        if (
            not _confirmed(previous) or not _confirmed(following)
            or previous.shot_id != following.shot_id or following.is_scene_cut
        ):
            return None, None, False, "identity_or_scene_boundary"
        fraction = (timestamp - previous.timestamp_ms) / gap
        points = {}
        for name in ACM_JOINTS:
            first, second = previous.keypoints.get(name), following.keypoints.get(name)
            if first is not None and second is not None:
                points[name] = Keypoint(
                    first.x + fraction * (second.x - first.x),
                    first.y + fraction * (second.y - first.y),
                    min(first.score, second.score),
                )
        return points, previous.shot_id, False, None

    def window(self, event: PunchEvent) -> ClassificationWindow:
        for fighter in (event.attacker_id, event.defender_id):
            values = self.by_fighter.get(fighter, [])
            times = self.times.get(fighter, [])
            start = bisect_left(times, event.peak_ms - 408)
            end = bisect_left(times, event.peak_ms + 409)
            span = values[start:end]
            track_ids = {obs.source_track_id for obs in span if obs.source_track_id is not None}
            if len(track_ids) > 1:
                return ClassificationWindow(None, "unavailable", "tracklet_boundary")
            if any(not _confirmed(obs) or obs.is_scene_cut for obs in span):
                return ClassificationWindow(None, "unavailable", "identity_or_scene_boundary")
        rows, shots, native = [], set(), True
        for timestamp in event.peak_ms + (np.arange(25) - 12) * (1000 / 30):
            attacker, shot, is_native, reason = self.sample(event.attacker_id, float(timestamp))
            _, defender_shot, defender_native, defender_reason = self.sample(event.defender_id, float(timestamp))
            if reason or defender_reason or attacker is None or shot != defender_shot:
                return ClassificationWindow(None, "unavailable", reason or defender_reason or "shot_mismatch")
            shots.add(shot)
            if len(shots) > 1:
                return ClassificationWindow(None, "unavailable", "scene_cut")
            row = normalize_acm_pose(attacker)
            visible = sum(attacker.get(name) is not None and attacker[name].score > .5 for name in ACM_JOINTS)
            if row is None or visible < 6:
                return ClassificationWindow(None, "unavailable", "insufficient_pose_visibility")
            rows.append(row)
            native = native and is_native and defender_native
        return ClassificationWindow(np.stack(rows)[None], "native_30fps" if native else "resampled")


def build_classification_window(event: PunchEvent, observations: Sequence[PoseObservation]) -> ClassificationWindow:
    return _WindowIndex(observations).window(event)


class AcmPunchClassifier:
    def __init__(self, bundle_dir: str | Path, providers: Sequence[str] | None = None):
        import onnxruntime as ort

        self.manifest = validate_acm_bundle(bundle_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(Path(bundle_dir) / "model.onnx"), options,
            providers=list(providers or ["CPUExecutionProvider"]),
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 25, 16] or inputs[0].type != "tensor(float)":
            raise ValueError("Unexpected punch model input")
        if len(outputs) != 1 or outputs[0].shape != [1, 4]:
            raise ValueError("Unexpected punch model output")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def predict(self, features: np.ndarray) -> PunchClassification:
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (1, 25, 16) or not np.isfinite(features).all():
            raise ValueError("Expected finite float32 punch features [1, 25, 16]")
        logits = self.session.run([self.output_name], {self.input_name: features})[0][0]
        if logits.shape != (4,) or not np.isfinite(logits).all():
            raise ValueError("Non-finite punch model output")
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        order = np.argsort(probabilities)[::-1]
        winner, alternative = int(order[0]), int(order[1])
        return PunchClassification(
            ACM_LABELS[winner], float(probabilities[winner]),
            float(probabilities[winner] - probabilities[alternative]),
            tuple(float(value) for value in probabilities),
        )


def _family(technique: str) -> str:
    return "straight" if technique in {"jab", "cross", "straight"} else technique


def _straight_label(hand: str, stance: str | None) -> str:
    if stance in {"orthodox", "southpaw"}:
        return "jab" if hand == ("left" if stance == "orthodox" else "right") else "cross"
    return "straight"


def classify_candidate_events(
    events: Sequence[PunchEvent], observations: Sequence[PoseObservation],
    classifier: PunchClassifier, *, stances: Mapping[str, str] | None = None,
    progress: Callable[[int, int], Any] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[PunchEvent]:
    """Refine existing candidates without inventing hand, contact or identity.

    Sparse resampled input is diagnostic only. A confident contradiction on
    native input abstains. Filling a geometric unknown requires manual review.
    The composite baseline confidence is never inflated by raw model scores.
    """
    index = _WindowIndex(observations)
    result = []
    for number, original in enumerate(events):
        if cancelled and cancelled():
            raise InterruptedError("Punch classification cancelled")
        event = replace(original, evidence=dict(original.evidence))
        window = index.window(event)
        event.evidence.update({"classification_input": window.input_kind, "classification_model": ACM_BUNDLE_ID})
        if window.features is None:
            event.evidence["classification_abstention"] = window.reason
        elif event.is_replay or event.review_status == "rejected":
            event.evidence["classification_abstention"] = "excluded_event"
        else:
            prediction = classifier.predict(window.features)
            event.evidence.update({
                "model_technique": prediction.label,
                "model_score_uncalibrated": prediction.score,
                "model_margin": prediction.margin,
                "model_probabilities": dict(zip(ACM_LABELS, prediction.probabilities)),
                "geometry_technique": event.technique,
            })
            family = _family(prediction.label)
            if window.input_kind != "native_30fps":
                event.evidence["classification_abstention"] = "sparse_input_diagnostic_only"
            elif prediction.score < .70 or prediction.margin < .15:
                event.evidence["classification_abstention"] = "weak_model_evidence"
            elif event.technique != "unknown" and family != _family(event.technique):
                event.technique = "unknown"
                event.classification_confidence = None
                event.review_status = "needs_review"
                event.evidence["classification_abstention"] = "model_geometry_disagreement"
            elif event.technique == "unknown":
                if prediction.score >= .80 and prediction.margin >= .20:
                    event.technique = _straight_label(event.hand, (stances or {}).get(event.attacker_id)) if family == "straight" else family
                    event.classification_confidence = None
                    event.review_status = "needs_review"
                    event.evidence["classification_consensus"] = "model_proposal_requires_review"
                else:
                    event.evidence["classification_abstention"] = "weak_unknown_refinement"
            else:
                event.classification_confidence = None
                event.evidence["classification_consensus"] = "geometry_model_agree"
            event.model_version = "temporal-fsm-v2+" + ACM_BUNDLE_ID
        result.append(event)
        if progress:
            progress(number + 1, len(events))
    return result
