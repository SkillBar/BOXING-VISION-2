"""Confidence-aware, pose-based boxing event candidates.

This is intentionally a conservative heuristic baseline, not a learned judge.
It detects a punch-shaped wrist trajectory and always retains an ``unclear`` /
``unknown`` path when pose geometry does not support a stronger conclusion.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .config import AnalysisConfig
from .contracts import Keypoint, PoseObservation, PunchEvent

OUTCOMES = {"likely_landed", "blocked", "missed", "unclear"}
TECHNIQUES = {"straight", "jab", "cross", "hook", "uppercut", "unknown"}
TARGETS = {"head", "body", "unknown"}


@dataclass(frozen=True, slots=True)
class PunchDetectionConfig:
    pose_score_threshold: float = 0.25
    min_wrist_speed: float = 0.85
    min_outward_speed: float = 0.30
    min_reach_gain: float = 0.16
    min_duration_ms: int = 80
    max_duration_ms: int = 900
    refractory_ms: int = 260
    event_confidence_threshold: float = 0.45
    contact_distance: float = 0.72
    block_distance: float = 0.52
    miss_distance: float = 1.25
    combo_window_ms: int = 1_400
    counter_window_ms: int = 950
    fight_start_s: float = 0.0
    round_length_s: int = 180
    rest_length_s: int = 60
    scheduled_rounds: int = 12

    @classmethod
    def from_analysis_config(cls, config: AnalysisConfig) -> PunchDetectionConfig:
        return cls(
            pose_score_threshold=config.pose_score_threshold,
            event_confidence_threshold=max(0.25, min(0.85, config.confidence_threshold)),
            fight_start_s=config.fight_start_s,
            round_length_s=config.round_length_s,
            rest_length_s=int(getattr(config, "rest_length_s", 60)),
            scheduled_rounds=config.scheduled_rounds,
        )


@dataclass(slots=True)
class _MotionSample:
    observation: PoseObservation
    defender: PoseObservation | None
    hand: str
    shoulder: Keypoint
    elbow: Keypoint
    wrist: Keypoint
    relative_wrist: np.ndarray
    torso_scale: float
    extension: float
    elbow_angle: float
    visibility: float
    speed: float = 0.0
    outward_speed: float = 0.0
    target_speed: float = 0.0
    acceleration: float = 0.0


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, value))


def _point(observation: PoseObservation | None, name: str, threshold: float) -> Keypoint | None:
    if observation is None:
        return None
    point = observation.keypoints.get(name)
    if point is None or point.score < threshold:
        return None
    if not all(math.isfinite(value) for value in (point.x, point.y, point.score)):
        return None
    return point


def _distance(first: Keypoint, second: Keypoint) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _torso_scale(observation: PoseObservation, threshold: float) -> float:
    left_shoulder = _point(observation, "left_shoulder", threshold)
    right_shoulder = _point(observation, "right_shoulder", threshold)
    shoulder_width = _distance(left_shoulder, right_shoulder) if left_shoulder and right_shoulder else 0.0
    left_hip = _point(observation, "left_hip", threshold)
    right_hip = _point(observation, "right_hip", threshold)
    torso_height = 0.0
    if left_hip and right_hip and left_shoulder and right_shoulder:
        shoulder_midpoint = Keypoint(
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            1.0,
        )
        hip_midpoint = Keypoint((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2, 1.0)
        torso_height = _distance(shoulder_midpoint, hip_midpoint) * 0.70
    return max(12.0, shoulder_width, torso_height, observation.bbox.height * 0.20)


def _angle(first: Keypoint, vertex: Keypoint, third: Keypoint) -> float:
    first_vector = np.array([first.x - vertex.x, first.y - vertex.y], dtype=np.float64)
    third_vector = np.array([third.x - vertex.x, third.y - vertex.y], dtype=np.float64)
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(third_vector))
    if denominator <= 1e-6:
        return 0.0
    cosine = float(np.dot(first_vector, third_vector) / denominator)
    return math.degrees(math.acos(_clamp(cosine, -1.0, 1.0)))


def _closest_observation(
    observations_by_time: Mapping[int, PoseObservation],
    timestamp_ms: int,
    *,
    tolerance_ms: int = 100,
    sorted_timestamps: Sequence[int] | None = None,
) -> PoseObservation | None:
    exact = observations_by_time.get(timestamp_ms)
    if exact is not None:
        return exact
    if not observations_by_time:
        return None
    timestamps = sorted_timestamps if sorted_timestamps is not None else sorted(observations_by_time)
    insertion = bisect_left(timestamps, timestamp_ms)
    candidates = timestamps[max(0, insertion - 1) : min(len(timestamps), insertion + 1)]
    if not candidates:
        return None
    nearest_timestamp = min(candidates, key=lambda value: abs(value - timestamp_ms))
    return observations_by_time[nearest_timestamp] if abs(nearest_timestamp - timestamp_ms) <= tolerance_ms else None


def _defender_target_center(observation: PoseObservation | None, threshold: float) -> np.ndarray | None:
    if observation is None:
        return None
    names = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
    points = [_point(observation, name, threshold) for name in names]
    visible = [point for point in points if point is not None]
    if not visible:
        return np.asarray(observation.bbox.center, dtype=np.float64)
    return np.asarray(
        [np.mean([point.x for point in visible]), np.mean([point.y for point in visible])],
        dtype=np.float64,
    )


def _build_motion_series(
    attacker_observations: Sequence[PoseObservation],
    defender_by_time: Mapping[int, PoseObservation],
    hand: str,
    config: PunchDetectionConfig,
) -> list[_MotionSample]:
    samples: list[_MotionSample] = []
    defender_timestamps = sorted(defender_by_time)
    for observation in attacker_observations:
        shoulder = _point(observation, f"{hand}_shoulder", config.pose_score_threshold)
        elbow = _point(observation, f"{hand}_elbow", config.pose_score_threshold)
        wrist = _point(observation, f"{hand}_wrist", config.pose_score_threshold)
        if not (shoulder and elbow and wrist):
            continue
        scale = _torso_scale(observation, config.pose_score_threshold)
        relative = np.asarray([(wrist.x - shoulder.x) / scale, (wrist.y - shoulder.y) / scale])
        visibility = min(
            shoulder.score,
            elbow.score,
            wrist.score,
            observation.track_confidence,
        )
        samples.append(
            _MotionSample(
                observation=observation,
                defender=_closest_observation(
                    defender_by_time,
                    observation.timestamp_ms,
                    sorted_timestamps=defender_timestamps,
                ),
                hand=hand,
                shoulder=shoulder,
                elbow=elbow,
                wrist=wrist,
                relative_wrist=relative,
                torso_scale=scale,
                extension=float(np.linalg.norm(relative)),
                elbow_angle=_angle(shoulder, elbow, wrist),
                visibility=visibility,
            )
        )
    for index in range(1, len(samples)):
        previous = samples[index - 1]
        current = samples[index]
        elapsed_s = (current.observation.timestamp_ms - previous.observation.timestamp_ms) / 1000.0
        if elapsed_s <= 0 or elapsed_s > 0.35 or current.observation.is_scene_cut:
            continue
        delta = current.relative_wrist - previous.relative_wrist
        current.speed = float(np.linalg.norm(delta)) / elapsed_s
        current.outward_speed = (current.extension - previous.extension) / elapsed_s
        target_center = _defender_target_center(current.defender, config.pose_score_threshold)
        if target_center is not None:
            direction = target_center - np.asarray([previous.wrist.x, previous.wrist.y])
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-6:
                wrist_delta_pixels = np.asarray(
                    [current.wrist.x - previous.wrist.x, current.wrist.y - previous.wrist.y],
                    dtype=np.float64,
                )
                current.target_speed = float(np.dot(wrist_delta_pixels / current.torso_scale, direction / direction_norm)) / elapsed_s
        current.acceleration = abs(current.speed - previous.speed) / elapsed_s
    return samples


def _candidate_peaks(samples: Sequence[_MotionSample], config: PunchDetectionConfig) -> list[int]:
    if len(samples) < 3:
        return []
    salience = np.asarray(
        [
            sample.speed
            * (0.72 + 0.28 * _clamp(max(sample.outward_speed, sample.target_speed) / 1.5))
            for sample in samples
        ],
        dtype=np.float64,
    )
    candidates: list[int] = []
    for index in range(1, len(samples) - 1):
        sample = samples[index]
        directional_speed = max(sample.outward_speed, sample.target_speed)
        if sample.speed < config.min_wrist_speed or directional_speed < config.min_outward_speed:
            continue
        if salience[index] + 1e-9 < salience[index - 1] or salience[index] < salience[index + 1]:
            continue
        candidates.append(index)
    # Non-maximum suppression in time, retaining the stronger trajectory.
    selected: list[int] = []
    for index in sorted(candidates, key=lambda candidate: salience[candidate], reverse=True):
        timestamp = samples[index].observation.timestamp_ms
        if any(abs(timestamp - samples[chosen].observation.timestamp_ms) < config.refractory_ms for chosen in selected):
            continue
        selected.append(index)
    return sorted(selected)


def _event_window(
    samples: Sequence[_MotionSample],
    peak_index: int,
    config: PunchDetectionConfig,
) -> tuple[int, int]:
    peak_timestamp = samples[peak_index].observation.timestamp_ms
    start = peak_index - 1
    while start > 0:
        current = samples[start]
        if peak_timestamp - current.observation.timestamp_ms >= min(450, config.max_duration_ms):
            break
        if current.observation.is_scene_cut or current.speed < config.min_wrist_speed * 0.28:
            break
        start -= 1
    # Include the guard frame immediately before acceleration.
    start = max(0, start)

    end = peak_index + 1
    while end < len(samples) - 1:
        current = samples[end]
        elapsed = current.observation.timestamp_ms - peak_timestamp
        if elapsed >= min(500, config.max_duration_ms):
            break
        if current.observation.is_scene_cut:
            break
        if current.speed < config.min_wrist_speed * 0.32 or current.outward_speed < -config.min_outward_speed * 0.7:
            break
        end += 1
    return start, min(len(samples) - 1, end)


def _head_points(observation: PoseObservation | None, threshold: float) -> list[Keypoint]:
    return [
        point
        for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        if (point := _point(observation, name, threshold)) is not None
    ]


def _body_points(observation: PoseObservation | None, threshold: float) -> list[Keypoint]:
    if observation is None:
        return []
    points = [
        point
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if (point := _point(observation, name, threshold)) is not None
    ]
    left_shoulder = _point(observation, "left_shoulder", threshold)
    right_shoulder = _point(observation, "right_shoulder", threshold)
    left_hip = _point(observation, "left_hip", threshold)
    right_hip = _point(observation, "right_hip", threshold)
    if left_shoulder and right_shoulder and left_hip and right_hip:
        points.append(
            Keypoint(
                (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4,
                (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4,
                min(left_shoulder.score, right_shoulder.score, left_hip.score, right_hip.score),
            )
        )
    return points


def _minimum_distance(wrist: Keypoint, points: Sequence[Keypoint], scale: float) -> float:
    if not points:
        return math.inf
    return min(_distance(wrist, point) for point in points) / max(1.0, scale)


def _contact_geometry(
    window: Sequence[_MotionSample],
    config: PunchDetectionConfig,
) -> tuple[str, str, float, float, float]:
    best_target = "unknown"
    best_target_distance = math.inf
    best_block_distance = math.inf
    defender_visibility: list[float] = []
    for sample in window:
        defender = sample.defender
        if defender is None:
            continue
        defender_scale = _torso_scale(defender, config.pose_score_threshold)
        head_distance = _minimum_distance(
            sample.wrist,
            _head_points(defender, config.pose_score_threshold),
            defender_scale,
        )
        body_distance = _minimum_distance(
            sample.wrist,
            _body_points(defender, config.pose_score_threshold),
            defender_scale,
        )
        if head_distance < best_target_distance and head_distance <= body_distance * 1.05:
            best_target = "head"
            best_target_distance = head_distance
        if body_distance < best_target_distance:
            best_target = "body"
            best_target_distance = body_distance
        guard_points = [
            point
            for name in ("left_wrist", "right_wrist", "left_elbow", "right_elbow")
            if (point := _point(defender, name, config.pose_score_threshold)) is not None
        ]
        best_block_distance = min(
            best_block_distance,
            _minimum_distance(sample.wrist, guard_points, defender_scale),
        )
        visible_scores = [point.score for point in _head_points(defender, config.pose_score_threshold)]
        visible_scores.extend(point.score for point in _body_points(defender, config.pose_score_threshold))
        if visible_scores:
            defender_visibility.append(float(np.mean(visible_scores)) * defender.track_confidence)

    visibility = float(np.mean(defender_visibility)) if defender_visibility else 0.0
    if not math.isfinite(best_target_distance):
        return "unknown", "unclear", math.inf, math.inf, visibility
    guard_intercepts = (
        best_block_distance <= config.block_distance
        and best_block_distance <= best_target_distance + 0.22
    )
    if best_target_distance <= config.contact_distance:
        if guard_intercepts:
            outcome = "blocked"
        else:
            outcome = "likely_landed"
    elif guard_intercepts and best_target_distance <= config.miss_distance:
        outcome = "blocked"
    elif best_target_distance >= config.miss_distance and visibility >= 0.50:
        outcome = "missed"
    else:
        outcome = "unclear"
    if best_target_distance > config.miss_distance:
        best_target = "unknown"
    return best_target, outcome, best_target_distance, best_block_distance, visibility


def _known_straight_label(hand: str, stance: str | None) -> str:
    normalized = (stance or "unknown").lower().strip()
    if normalized in {"orthodox", "правша", "right-handed", "right_handed"}:
        return "jab" if hand == "left" else "cross"
    if normalized in {"southpaw", "левша", "left-handed", "left_handed"}:
        return "jab" if hand == "right" else "cross"
    return "straight"


def _classify_technique(
    start: _MotionSample,
    peak: _MotionSample,
    *,
    stance: str | None,
) -> str:
    displacement = np.asarray([peak.wrist.x - start.wrist.x, peak.wrist.y - start.wrist.y]) / peak.torso_scale
    horizontal, vertical = abs(float(displacement[0])), float(displacement[1])
    reach_gain = peak.extension - start.extension
    if vertical < -0.24 and abs(vertical) > horizontal * 0.62 and peak.elbow_angle < 155:
        return "uppercut"
    if peak.elbow_angle >= 145 or reach_gain >= 0.38:
        return _known_straight_label(peak.hand, stance)
    if peak.elbow_angle <= 142 and horizontal >= abs(vertical) * 0.72:
        return "hook"
    return "unknown"


def _round_number(timestamp_ms: int, config: PunchDetectionConfig) -> int | None:
    elapsed_ms = max(0, timestamp_ms - int(config.fight_start_s * 1000))
    cycle_ms = max(1, (config.round_length_s + config.rest_length_s) * 1000)
    if elapsed_ms % cycle_ms >= config.round_length_s * 1000:
        return None
    result = elapsed_ms // cycle_ms + 1
    if result > config.scheduled_rounds:
        return None
    return int(max(1, result))


def _impact_proxy(
    peak_speed: float,
    peak_acceleration: float,
    reach_gain: float,
    visibility: float,
    outcome: str,
) -> int:
    speed_score = _clamp((peak_speed - 0.65) / 3.2)
    acceleration_score = _clamp((peak_acceleration - 1.0) / 13.0)
    reach_score = _clamp(reach_gain / 0.75)
    estimate = 100.0 * (
        0.52 * speed_score + 0.18 * acceleration_score + 0.20 * reach_score + 0.10 * visibility
    )
    caps = {"missed": 35.0, "blocked": 58.0, "unclear": 48.0}
    estimate = min(estimate, caps.get(outcome, 100.0))
    return round(_clamp(estimate, 0.0, 100.0))


def _make_event(
    samples: Sequence[_MotionSample],
    start_index: int,
    peak_index: int,
    end_index: int,
    *,
    attacker_id: str,
    defender_id: str,
    stance: str | None,
    config: PunchDetectionConfig,
) -> PunchEvent | None:
    start, peak, end = samples[start_index], samples[peak_index], samples[end_index]
    duration_ms = end.observation.timestamp_ms - start.observation.timestamp_ms
    if duration_ms < config.min_duration_ms or duration_ms > config.max_duration_ms:
        return None
    reach_gain = peak.extension - start.extension
    target_progress = max(
        sample.target_speed for sample in samples[max(start_index + 1, 0) : peak_index + 1]
    ) if peak_index > start_index else 0.0
    if reach_gain < config.min_reach_gain and target_progress < config.min_outward_speed * 1.4:
        return None
    window = samples[start_index : end_index + 1]
    target, outcome, target_distance, block_distance, defender_visibility = _contact_geometry(window, config)
    peak_speed = max(sample.speed for sample in window)
    peak_acceleration = max(sample.acceleration for sample in window)
    motion_confidence = _clamp((peak_speed - config.min_wrist_speed * 0.72) / 2.3)
    reach_confidence = _clamp(max(reach_gain, target_progress * 0.16) / 0.55)
    pose_confidence = float(np.mean([sample.visibility for sample in window]))
    geometry_confidence = {
        "likely_landed": 0.85,
        "blocked": 0.78,
        "missed": 0.70,
        "unclear": 0.38,
    }[outcome]
    if outcome != "unclear":
        geometry_confidence *= 0.55 + 0.45 * defender_visibility
    confidence = _clamp(
        0.34 * motion_confidence
        + 0.24 * reach_confidence
        + 0.24 * pose_confidence
        + 0.18 * geometry_confidence
    )
    if confidence < config.event_confidence_threshold:
        return None
    technique = _classify_technique(start, peak, stance=stance)
    impact = _impact_proxy(peak_speed, peak_acceleration, max(0.0, reach_gain), pose_confidence, outcome)
    evidence = {
        "peak_wrist_speed_torso_s": round(peak_speed, 4),
        "peak_acceleration_torso_s2": round(peak_acceleration, 4),
        "reach_gain_torso": round(reach_gain, 4),
        "target_distance_torso": round(target_distance, 4) if math.isfinite(target_distance) else -1.0,
        "guard_distance_torso": round(block_distance, 4) if math.isfinite(block_distance) else -1.0,
        "pose_visibility": round(pose_confidence, 4),
    }
    round_number = _round_number(peak.observation.timestamp_ms, config)
    if round_number is None:
        return None
    return PunchEvent(
        event_id="pending",
        round=round_number,
        start_ms=start.observation.timestamp_ms,
        peak_ms=peak.observation.timestamp_ms,
        end_ms=end.observation.timestamp_ms,
        attacker_id=attacker_id,
        defender_id=defender_id,
        hand=peak.hand,
        technique=technique if technique in TECHNIQUES else "unknown",
        target=target if target in TARGETS else "unknown",
        outcome=outcome if outcome in OUTCOMES else "unclear",
        confidence=round(confidence, 4),
        impact_proxy_0_100=impact,
        evidence=evidence,
    )


def _deduplicate(events: Sequence[PunchEvent], refractory_ms: int) -> list[PunchEvent]:
    selected: list[PunchEvent] = []
    quality = lambda event: event.confidence * (1.0 + event.impact_proxy_0_100 / 200.0)
    for event in sorted(events, key=lambda item: quality(item), reverse=True):
        duplicate = any(
            event.attacker_id == existing.attacker_id
            and abs(event.peak_ms - existing.peak_ms) < refractory_ms
            for existing in selected
        )
        if not duplicate:
            selected.append(event)
    return sorted(selected, key=lambda item: (item.peak_ms, item.attacker_id))


def annotate_exchanges(
    events: Sequence[PunchEvent],
    *,
    combo_window_ms: int = 1_400,
    counter_window_ms: int = 950,
) -> list[PunchEvent]:
    """Mutate event metadata with deterministic combo and counter labels."""

    ordered = sorted(events, key=lambda event: (event.peak_ms, event.attacker_id))
    combo_number = 1
    for attacker_id in sorted({event.attacker_id for event in ordered}):
        attacker_events = [event for event in ordered if event.attacker_id == attacker_id and not event.is_replay]
        group: list[PunchEvent] = []
        for event in attacker_events:
            if group and event.peak_ms - group[-1].peak_ms > combo_window_ms:
                if len(group) >= 2:
                    combo_id = f"combo_{combo_number:04d}"
                    combo_number += 1
                    for grouped_event in group:
                        grouped_event.combo_id = combo_id
                group = []
            group.append(event)
        if len(group) >= 2:
            combo_id = f"combo_{combo_number:04d}"
            combo_number += 1
            for grouped_event in group:
                grouped_event.combo_id = combo_id

    for index, event in enumerate(ordered):
        prior_opponent_events = [
            prior
            for prior in ordered[:index]
            if prior.attacker_id == event.defender_id
            and 0 < event.peak_ms - prior.peak_ms <= counter_window_ms
            and not prior.is_replay
        ]
        event.is_counter = bool(prior_opponent_events)
    return ordered


def annotate_possible_knockdowns(
    events: Sequence[PunchEvent],
    observations: Sequence[PoseObservation],
    *,
    lookback_ms: int = 900,
    reaction_window_ms: int = 2_200,
) -> list[PunchEvent]:
    """Flag a conservative post-impact fall pattern for manual review.

    A monocular broadcast cannot confirm a knockdown. We only mark a likely
    landed, reasonably strong event when its defender changes from an upright
    pose to a clearly horizontal pose shortly afterwards. The flag never
    changes the 10-point score without the user's separate confirmation.
    """

    by_fighter: dict[str, list[PoseObservation]] = defaultdict(list)
    for observation in observations:
        by_fighter[observation.fighter_id].append(observation)
    for fighter_observations in by_fighter.values():
        fighter_observations.sort(key=lambda observation: observation.timestamp_ms)
    timestamps_by_fighter = {
        fighter_id: [observation.timestamp_ms for observation in fighter_observations]
        for fighter_id, fighter_observations in by_fighter.items()
    }

    for event in events:
        if (
            event.is_replay
            or _canonical_outcome_for_fall(event.outcome) != "likely_landed"
            or event.confidence < 0.58
            or event.impact_proxy_0_100 < 42
        ):
            continue
        defender = by_fighter.get(event.defender_id, [])
        timestamps = timestamps_by_fighter.get(event.defender_id, [])
        before = defender[
            bisect_left(timestamps, event.peak_ms - lookback_ms) : bisect_right(
                timestamps, event.peak_ms + 250
            )
        ]
        after = defender[
            bisect_right(timestamps, event.peak_ms + 250) : bisect_right(
                timestamps, event.peak_ms + reaction_window_ms
            )
        ]
        upright = any(
            observation.bbox.height >= observation.bbox.width * 1.18
            and observation.track_confidence >= 0.30
            for observation in before
        )
        horizontal = [
            observation
            for observation in after
            if observation.bbox.width >= observation.bbox.height * 1.16
            and observation.track_confidence >= 0.30
        ]
        if not upright or len(horizontal) < 2:
            continue
        fall_confidence = min(
            0.92,
            0.38
            + 0.24 * event.confidence
            + 0.18 * (event.impact_proxy_0_100 / 100.0)
            + 0.04 * min(3, len(horizontal)),
        )
        event.evidence["possible_knockdown"] = round(fall_confidence, 4)
        if event.review_status == "unreviewed":
            event.review_status = "needs_review"
    return list(events)


def _canonical_outcome_for_fall(value: str) -> str:
    normalized = value.lower().strip().replace("-", "_")
    return "likely_landed" if normalized in {"likely_landed", "landed", "likely landed"} else normalized


def _normalize_config(config: AnalysisConfig | PunchDetectionConfig | None) -> PunchDetectionConfig:
    if config is None:
        return PunchDetectionConfig()
    if isinstance(config, PunchDetectionConfig):
        return config
    return PunchDetectionConfig.from_analysis_config(config)


def detect_punch_events(
    observations: Iterable[PoseObservation],
    config: AnalysisConfig | PunchDetectionConfig | None = None,
    *,
    stances: Mapping[str, str] | None = None,
    replay_intervals: Sequence[tuple[int, int]] | None = None,
) -> list[PunchEvent]:
    """Detect punch candidates from a flat stream of two-fighter poses."""

    detection_config = _normalize_config(config)
    ordered_observations = sorted(
        observations,
        key=lambda observation: (observation.timestamp_ms, observation.fighter_id),
    )
    by_fighter: dict[str, list[PoseObservation]] = defaultdict(list)
    for observation in ordered_observations:
        by_fighter[observation.fighter_id].append(observation)
    fighter_ids = sorted(by_fighter)
    if len(fighter_ids) < 2:
        return []

    candidates: list[PunchEvent] = []
    # The UI configures exactly two fighters.  If an upstream adapter happens
    # to pass more tracks, the two longest tracks are the safest choice.
    fighter_ids = sorted(fighter_ids, key=lambda fighter_id: len(by_fighter[fighter_id]), reverse=True)[:2]
    for attacker_id, defender_id in ((fighter_ids[0], fighter_ids[1]), (fighter_ids[1], fighter_ids[0])):
        defender_by_time = {
            observation.timestamp_ms: observation for observation in by_fighter[defender_id]
        }
        for hand in ("left", "right"):
            samples = _build_motion_series(
                by_fighter[attacker_id],
                defender_by_time,
                hand,
                detection_config,
            )
            for peak_index in _candidate_peaks(samples, detection_config):
                start_index, end_index = _event_window(samples, peak_index, detection_config)
                event = _make_event(
                    samples,
                    start_index,
                    peak_index,
                    end_index,
                    attacker_id=attacker_id,
                    defender_id=defender_id,
                    stance=(stances or {}).get(attacker_id),
                    config=detection_config,
                )
                if event is not None:
                    candidates.append(event)

    events = _deduplicate(candidates, detection_config.refractory_ms)
    intervals = list(replay_intervals or [])
    for index, event in enumerate(events, start=1):
        event.event_id = f"evt_{index:05d}"
        event.is_replay = any(start_ms <= event.peak_ms <= end_ms for start_ms, end_ms in intervals)
    annotated = annotate_exchanges(
        events,
        combo_window_ms=detection_config.combo_window_ms,
        counter_window_ms=detection_config.counter_window_ms,
    )
    return annotate_possible_knockdowns(annotated, ordered_observations)


class PunchEventDetector:
    """Small buffering facade for frame-by-frame pipelines."""

    def __init__(
        self,
        config: AnalysisConfig | PunchDetectionConfig | None = None,
        *,
        stances: Mapping[str, str] | None = None,
    ) -> None:
        self.config = _normalize_config(config)
        self.stances = dict(stances or {})
        self._observations: list[PoseObservation] = []
        self._replay_intervals: list[tuple[int, int]] = []

    def process(self, observations: PoseObservation | Iterable[PoseObservation]) -> list[PunchEvent]:
        """Buffer one frame; finalized events are returned by :meth:`flush`."""

        if isinstance(observations, PoseObservation):
            self._observations.append(observations)
        else:
            self._observations.extend(observations)
        return []

    def add_replay_interval(self, start_ms: int, end_ms: int) -> None:
        if end_ms < start_ms:
            raise ValueError("Конец повтора должен быть позже начала")
        self._replay_intervals.append((start_ms, end_ms))

    def flush(self, *, clear: bool = True) -> list[PunchEvent]:
        events = detect_punch_events(
            self._observations,
            self.config,
            stances=self.stances,
            replay_intervals=self._replay_intervals,
        )
        if clear:
            self.reset()
        return events

    def reset(self) -> None:
        self._observations.clear()
        self._replay_intervals.clear()


# Backwards-friendly concise alias.
PunchDetector = PunchEventDetector
