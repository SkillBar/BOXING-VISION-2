"""Pose inference and stable assignment of the two configured fighters."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .contracts import COCO_KEYPOINT_NAMES, BBox, Keypoint, PoseObservation

_LIGHTWEIGHT_MODEL_HASHES = {
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip": (
        "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx",
        "ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250",
    ),
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip": (
        "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx",
        "9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d",
    ),
}


class PoseBackendUnavailable(RuntimeError):
    """Raised with an actionable message when RTMLib cannot be used."""


@dataclass(slots=True)
class RawPose:
    """Backend-neutral pose returned before a fighter identity is assigned."""

    bbox: BBox
    keypoints: dict[str, Keypoint]
    confidence: float = 1.0
    appearance: tuple[float, ...] | None = None


class PoseBackend(Protocol):
    name: str

    def infer(self, frame: np.ndarray) -> list[RawPose]: ...

    def reset(self) -> None: ...


class NullPoseBackend:
    """Non-crashing fallback used to keep the UI alive and show the reason."""

    name = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def infer(self, frame: np.ndarray) -> list[RawPose]:
        del frame
        return []

    def reset(self) -> None:
        return None


def _auto_device() -> str:
    """Choose RTMLib's most reliable ONNX Runtime provider.

    CoreML currently rejects a dynamic-rank YOLOX node on some Apple Silicon
    builds.  CPU is already fast enough for the offline lightweight model and
    avoids a first-run failure.  Advanced users can still request ``mps``
    explicitly after validating their local ONNX Runtime version.
    """

    return "cpu"


class RTMLibPoseBackend:
    """RTMPose/YOLOX body-pose backend with no Ultralytics dependency.

    RTMLib downloads its Apache-licensed ONNX model files on first use.  Model
    construction is lazy so importing the web application never unexpectedly
    downloads hundreds of megabytes.
    """

    name = "rtmlib-rtmpose"

    def __init__(
        self,
        *,
        mode: str = "lightweight",
        device: str = "auto",
        backend: str = "onnxruntime",
        pose_score_threshold: float = 0.25,
    ) -> None:
        if mode not in {"lightweight", "balanced", "performance"}:
            raise ValueError("RTMLib mode должен быть lightweight, balanced или performance")
        self.mode = mode
        self.device = _auto_device() if device == "auto" else device
        self.backend = backend
        self.pose_score_threshold = pose_score_threshold
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from rtmlib import Body
            from rtmlib.tools.file import download_checkpoint
        except Exception as exc:
            raise PoseBackendUnavailable(
                "RTMLib не установлен. Выполните установку зависимостей проекта."
            ) from exc
        try:
            if self.mode == "lightweight":
                self._prefetch_and_verify_lightweight_models(download_checkpoint)
            # Body performs detection with YOLOX and body pose with RTMPose.  We
            # intentionally do our own identity assignment below: RTMLib's IoU
            # tracker is not robust to a referee crossing between the fighters.
            self._model = Body(
                mode=self.mode,
                to_openpose=False,
                backend=self.backend,
                device=self.device,
            )
        except Exception as exc:
            raise PoseBackendUnavailable(
                "Не удалось загрузить RTMPose. Проверьте сеть/кэш моделей или выберите CPU: "
                f"{exc}"
            ) from exc
        return self._model

    @staticmethod
    def _prefetch_and_verify_lightweight_models(download_checkpoint: object) -> None:
        if not callable(download_checkpoint):
            raise PoseBackendUnavailable("RTMLib download helper недоступен")
        torch_home = os.getenv("TORCH_HOME")
        if torch_home:
            cache_root = Path(torch_home).expanduser()
        else:
            xdg_cache = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
            cache_root = xdg_cache / "rtmlib"
        checkpoint_dir = cache_root / "hub" / "checkpoints"
        for url, (filename, expected) in _LIGHTWEIGHT_MODEL_HASHES.items():
            model_path = checkpoint_dir / filename
            if not model_path.is_file():
                downloaded = Path(str(download_checkpoint(url)))
                if downloaded.name != filename:
                    raise PoseBackendUnavailable(
                        f"RTMLib вернул неожиданный checkpoint: {downloaded.name}"
                    )
                model_path = downloaded
            if not model_path.is_file():
                raise PoseBackendUnavailable(
                    f"RTMLib не сохранил ожидаемый checkpoint: {filename}"
                )
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if digest != expected:
                raise PoseBackendUnavailable(
                    f"Контрольная сумма checkpoint не совпала: {filename}"
                )

    def reset(self) -> None:
        # Body itself has no temporal state.
        return None

    @staticmethod
    def _appearance_descriptor(frame: np.ndarray, bbox: BBox) -> tuple[float, ...] | None:
        """Compact HSV clothing descriptor used only for post-cut re-ID."""

        height, width = frame.shape[:2]
        x1 = int(max(0, min(width - 1, bbox.x1 + bbox.width * 0.08)))
        x2 = int(max(x1 + 1, min(width, bbox.x2 - bbox.width * 0.08)))
        y1 = int(max(0, min(height - 1, bbox.y1 + bbox.height * 0.20)))
        y2 = int(max(y1 + 1, min(height, bbox.y2 - bbox.height * 0.08)))
        crop = frame[y1:y2, x1:x2]
        if crop.size < 96:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).reshape(-1)
        total = float(histogram.sum())
        if total <= 1e-6:
            return None
        histogram = histogram.astype(np.float32) / total
        return tuple(float(value) for value in histogram)

    def infer(self, frame: np.ndarray) -> list[RawPose]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("RTMPose ожидает BGR-кадр H×W×3")
        model = self._ensure_model()
        try:
            keypoints_array, scores_array = model(frame)  # type: ignore[operator]
        except PoseBackendUnavailable:
            raise
        except Exception as exc:
            raise PoseBackendUnavailable(f"Ошибка RTMPose inference: {exc}") from exc
        keypoints_array = np.asarray(keypoints_array)
        scores_array = np.asarray(scores_array)
        if keypoints_array.size == 0:
            return []
        if keypoints_array.ndim == 2:
            keypoints_array = keypoints_array[None, ...]
        if scores_array.ndim == 1:
            scores_array = scores_array[None, ...]

        height, width = frame.shape[:2]
        poses: list[RawPose] = []
        for coordinates, scores in zip(keypoints_array, scores_array):
            point_count = min(len(COCO_KEYPOINT_NAMES), len(coordinates), len(scores))
            mapped: dict[str, Keypoint] = {}
            valid_xy: list[tuple[float, float]] = []
            valid_scores: list[float] = []
            for index in range(point_count):
                x, y = float(coordinates[index][0]), float(coordinates[index][1])
                score = float(scores[index])
                if not all(math.isfinite(value) for value in (x, y, score)):
                    continue
                score = min(1.0, max(0.0, score))
                mapped[COCO_KEYPOINT_NAMES[index]] = Keypoint(x=x, y=y, score=score)
                if score >= self.pose_score_threshold:
                    valid_xy.append((x, y))
                    valid_scores.append(score)
            if len(valid_xy) < 5:
                continue
            xs = [point[0] for point in valid_xy]
            ys = [point[1] for point in valid_xy]
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)
            pad_x = max(4.0, x_span * 0.12)
            pad_y = max(4.0, y_span * 0.08)
            bbox = BBox(
                x1=max(0.0, min(xs) - pad_x),
                y1=max(0.0, min(ys) - pad_y),
                x2=min(float(width - 1), max(xs) + pad_x),
                y2=min(float(height - 1), max(ys) + pad_y),
                score=float(np.mean(valid_scores)),
            )
            if bbox.area < 400:
                continue
            poses.append(
                RawPose(
                    bbox=bbox,
                    keypoints=mapped,
                    confidence=bbox.score,
                    appearance=self._appearance_descriptor(frame, bbox),
                )
            )
        return poses


def create_pose_backend(
    backend: str = "auto",
    *,
    strict: bool = True,
    mode: str = "lightweight",
    device: str = "auto",
    pose_score_threshold: float = 0.25,
) -> PoseBackend:
    """Create the configured backend or an explicit non-crashing fallback."""

    normalized = backend.strip().lower()
    if normalized not in {"auto", "rtmlib", "rtmpose", "none"}:
        raise ValueError(f"Неизвестный pose backend: {backend}")
    if normalized == "none":
        return NullPoseBackend("Pose-анализ отключён")
    if importlib.util.find_spec("rtmlib") is None:
        message = "RTMLib не установлен. Установите зависимости из pyproject.toml."
        if strict:
            raise PoseBackendUnavailable(message)
        return NullPoseBackend(message)
    return RTMLibPoseBackend(
        mode=mode,
        device=device,
        pose_score_threshold=pose_score_threshold,
    )


def _coerce_keypoint(value: object) -> Keypoint:
    if isinstance(value, Keypoint):
        return value
    if isinstance(value, Mapping):
        return Keypoint(float(value["x"]), float(value["y"]), float(value.get("score", 1.0)))
    if isinstance(value, Sequence) and len(value) >= 2:
        score = float(value[2]) if len(value) >= 3 else 1.0
        return Keypoint(float(value[0]), float(value[1]), score)
    raise TypeError(f"Неподдерживаемая ключевая точка: {value!r}")


def _bbox_from_keypoints(keypoints: Mapping[str, Keypoint]) -> BBox:
    visible = [point for point in keypoints.values() if point.score > 0]
    if not visible:
        return BBox(0, 0, 0, 0, 0)
    xs = [point.x for point in visible]
    ys = [point.y for point in visible]
    return BBox(min(xs), min(ys), max(xs), max(ys), float(np.mean([point.score for point in visible])))


def coerce_raw_pose(value: RawPose | PoseObservation | Mapping[str, object]) -> RawPose:
    """Accept lightweight dictionaries as well as the internal dataclasses."""

    if isinstance(value, RawPose):
        return value
    if isinstance(value, PoseObservation):
        return RawPose(value.bbox, value.keypoints, value.track_confidence)
    if not isinstance(value, Mapping):
        raise TypeError(f"Неподдерживаемый формат pose: {type(value).__name__}")
    raw_keypoints = value.get("keypoints")
    if not isinstance(raw_keypoints, Mapping):
        raise TypeError("Pose должен содержать keypoints")
    keypoints = {str(name): _coerce_keypoint(point) for name, point in raw_keypoints.items()}
    raw_bbox = value.get("bbox")
    if isinstance(raw_bbox, BBox):
        bbox = raw_bbox
    elif isinstance(raw_bbox, Mapping):
        bbox = BBox(
            float(raw_bbox["x1"]),
            float(raw_bbox["y1"]),
            float(raw_bbox["x2"]),
            float(raw_bbox["y2"]),
            float(raw_bbox.get("score", 1.0)),
        )
    elif isinstance(raw_bbox, Sequence) and len(raw_bbox) >= 4:
        bbox = BBox(
            float(raw_bbox[0]),
            float(raw_bbox[1]),
            float(raw_bbox[2]),
            float(raw_bbox[3]),
            float(raw_bbox[4]) if len(raw_bbox) >= 5 else 1.0,
        )
    else:
        bbox = _bbox_from_keypoints(keypoints)
    confidence = float(value.get("confidence", bbox.score))
    raw_appearance = value.get("appearance")
    appearance = (
        tuple(float(item) for item in raw_appearance)
        if isinstance(raw_appearance, Sequence) and not isinstance(raw_appearance, (str, bytes))
        else None
    )
    return RawPose(bbox, keypoints, min(1.0, max(0.0, confidence)), appearance)


@dataclass(slots=True)
class _TrackState:
    bbox: BBox
    center: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    missed_frames: int = 0
    appearance: tuple[float, ...] | None = None


class TwoFighterTracker:
    """Assign at most two persistent fighter IDs and suppress the referee.

    The tracker uses predicted centre, scale consistency and pose confidence.
    It is intentionally deterministic, which makes it easy to validate on
    synthetic sequences and avoids a heavyweight native tracking dependency.
    """

    def __init__(
        self,
        fighter_ids: tuple[str, str] = ("fighter_a", "fighter_b"),
        *,
        anchors: Mapping[str, tuple[float, float]] | None = None,
        max_missing_frames: int = 12,
        max_assignment_cost: float = 3.0,
        candidate_limit: int = 6,
    ) -> None:
        if len(fighter_ids) != 2 or fighter_ids[0] == fighter_ids[1]:
            raise ValueError("Нужны два разных fighter_id")
        self.fighter_ids = fighter_ids
        self.anchors = dict(anchors or {})
        self.max_missing_frames = max_missing_frames
        self.max_assignment_cost = max_assignment_cost
        self.candidate_limit = max(2, candidate_limit)
        self._tracks: dict[str, _TrackState] = {}
        self._appearance_profiles: dict[str, tuple[float, ...]] = {}

    def reset(self, *, keep_anchors: bool = True, keep_appearance: bool = True) -> None:
        self._tracks.clear()
        if not keep_anchors:
            self.anchors.clear()
        if not keep_appearance:
            self._appearance_profiles.clear()

    def set_anchors(self, fighter_a: tuple[float, float], fighter_b: tuple[float, float]) -> None:
        self.anchors = {self.fighter_ids[0]: fighter_a, self.fighter_ids[1]: fighter_b}
        self._tracks.clear()

    def clear_anchors(self) -> None:
        """Keep active tracks but stop reusing first-shot screen positions."""

        self.anchors.clear()

    @staticmethod
    def _quality(pose: RawPose) -> float:
        visible = sum(point.score >= 0.25 for point in pose.keypoints.values())
        return pose.bbox.area * max(0.05, pose.confidence) * min(1.0, visible / 10.0)

    @staticmethod
    def _assignment_cost(state: _TrackState, candidate: RawPose) -> float:
        predicted = (state.center[0] + state.velocity[0], state.center[1] + state.velocity[1])
        center = candidate.bbox.center
        reference_scale = max(25.0, state.bbox.height, candidate.bbox.height)
        distance = math.dist(predicted, center) / reference_scale
        area_ratio = (candidate.bbox.area + 1.0) / (state.bbox.area + 1.0)
        scale_penalty = abs(math.log(area_ratio)) * 0.28
        confidence_penalty = (1.0 - candidate.confidence) * 0.25
        appearance_penalty = TwoFighterTracker._appearance_distance(
            state.appearance,
            candidate.appearance,
        )
        return distance + scale_penalty + confidence_penalty + 0.72 * appearance_penalty

    @staticmethod
    def _appearance_distance(
        first: tuple[float, ...] | None,
        second: tuple[float, ...] | None,
    ) -> float:
        if first is None or second is None or len(first) != len(second):
            return 0.45
        first_array = np.asarray(first, dtype=np.float32)
        second_array = np.asarray(second, dtype=np.float32)
        coefficient = float(np.sum(np.sqrt(np.maximum(0.0, first_array * second_array))))
        return min(1.0, max(0.0, math.sqrt(max(0.0, 1.0 - coefficient))))

    def _profile_assignment(
        self,
        candidates: list[RawPose],
        candidate_indices: list[int],
    ) -> dict[str, int] | None:
        profiled_ids = [
            fighter_id
            for fighter_id in self.fighter_ids
            if fighter_id in self._appearance_profiles
        ]
        if not profiled_ids or not candidate_indices:
            return None
        assignment_count = min(len(profiled_ids), len(candidate_indices))
        best: tuple[float, dict[str, int]] | None = None
        for assigned_ids in itertools.permutations(profiled_ids, assignment_count):
            for indices in itertools.permutations(candidate_indices, assignment_count):
                mapping = dict(zip(assigned_ids, indices))
                cost = sum(
                    self._appearance_distance(
                        self._appearance_profiles[fighter_id],
                        candidates[index].appearance,
                    )
                    for fighter_id, index in mapping.items()
                )
                if best is None or cost < best[0]:
                    best = (cost, mapping)
        if best is None or best[0] / assignment_count > 0.62:
            return None
        return best[1]

    def _initial_assignment(self, candidates: list[RawPose]) -> dict[str, int]:
        quality_order = sorted(
            range(len(candidates)),
            key=lambda index: self._quality(candidates[index]),
            reverse=True,
        )
        # Manual anchors are stronger evidence than apparent box area and let
        # us reject a large referee standing in the foreground.
        selected_indices = quality_order if self.anchors else quality_order[:2]
        if not selected_indices:
            return {}
        if self.anchors:
            assignment_count = min(len(self.fighter_ids), len(selected_indices))
            best: tuple[float, dict[str, int]] | None = None
            for assigned_ids in itertools.permutations(self.fighter_ids, assignment_count):
                for indices in itertools.permutations(selected_indices, assignment_count):
                    mapping = dict(zip(assigned_ids, indices))
                    cost = sum(
                        math.dist(self.anchors[fighter_id], candidates[index].bbox.center)
                        for fighter_id, index in mapping.items()
                        if fighter_id in self.anchors
                    )
                    if best is None or cost < best[0]:
                        best = (cost, mapping)
            result = dict(best[1]) if best is not None else {}
            available = set(selected_indices) - set(result.values())
            for fighter_id in self.fighter_ids:
                if fighter_id not in result and available:
                    result[fighter_id] = available.pop()
            return result
        profile_candidates = [
            index
            for index in quality_order[: self.candidate_limit]
            if candidates[index].appearance is not None
        ]
        profile_result = self._profile_assignment(candidates, profile_candidates)
        if profile_result:
            available = set(quality_order) - set(profile_result.values())
            for fighter_id in self.fighter_ids:
                if fighter_id not in profile_result and available:
                    chosen = max(available, key=lambda index: self._quality(candidates[index]))
                    profile_result[fighter_id] = chosen
                    available.remove(chosen)
            return profile_result
        selected_indices.sort(key=lambda index: candidates[index].bbox.center[0])
        return {fighter_id: index for fighter_id, index in zip(self.fighter_ids, selected_indices)}

    def _tracked_assignment(self, candidates: list[RawPose]) -> dict[str, int]:
        active_ids = [fighter_id for fighter_id in self.fighter_ids if fighter_id in self._tracks]
        if not active_ids:
            return self._initial_assignment(candidates)
        best: tuple[float, dict[str, int]] | None = None
        # Exhaustive search is tiny (at most 2 × 6) and avoids scipy in the hot
        # path.  It prevents both IDs from claiming the same person.
        assignment_count = min(len(active_ids), len(candidates))
        for assigned_ids in itertools.permutations(active_ids, assignment_count):
            for indices in itertools.permutations(range(len(candidates)), assignment_count):
                mapping = dict(zip(assigned_ids, indices))
                total = sum(
                    self._assignment_cost(self._tracks[fighter_id], candidates[index])
                    for fighter_id, index in mapping.items()
                )
                if best is None or total < best[0]:
                    best = (total, mapping)
        assignments = dict(best[1]) if best is not None else {}
        assignments = {
            fighter_id: index
            for fighter_id, index in assignments.items()
            if self._assignment_cost(self._tracks[fighter_id], candidates[index]) <= self.max_assignment_cost
        }
        available = set(range(len(candidates))) - set(assignments.values())
        for fighter_id in self.fighter_ids:
            if fighter_id in assignments or fighter_id in self._tracks or not available:
                continue
            # A previously unseen second fighter may enter after the first.
            anchor = self.anchors.get(fighter_id)
            if anchor is not None:
                assignments[fighter_id] = min(
                    available,
                    key=lambda index: math.dist(anchor, candidates[index].bbox.center),
                )
            else:
                assignments[fighter_id] = max(
                    available,
                    key=lambda index: self._quality(candidates[index]),
                )
            available.remove(assignments[fighter_id])
        return assignments

    def process(
        self,
        frame_index: int,
        timestamp_ms: int,
        poses: Sequence[RawPose | PoseObservation | Mapping[str, object]],
        scene_cut: bool = False,
    ) -> list[PoseObservation]:
        if scene_cut:
            self.reset(keep_anchors=True)
        candidates = [coerce_raw_pose(pose) for pose in poses]
        candidates = [candidate for candidate in candidates if candidate.bbox.area > 0]
        candidates.sort(key=self._quality, reverse=True)
        candidates = candidates[: self.candidate_limit]
        assignments = self._tracked_assignment(candidates) if self._tracks else self._initial_assignment(candidates)

        observations: list[PoseObservation] = []
        assigned_ids = set(assignments)
        for fighter_id, candidate_index in assignments.items():
            candidate = candidates[candidate_index]
            previous = self._tracks.get(fighter_id)
            center = candidate.bbox.center
            if previous is None:
                velocity = (0.0, 0.0)
                assignment_confidence = 0.75 if len(candidates) > 2 else 0.9
            else:
                displacement = (center[0] - previous.center[0], center[1] - previous.center[1])
                velocity = (
                    previous.velocity[0] * 0.55 + displacement[0] * 0.45,
                    previous.velocity[1] * 0.55 + displacement[1] * 0.45,
                )
                cost = self._assignment_cost(previous, candidate)
                assignment_confidence = math.exp(-0.55 * cost)
            track_confidence = min(1.0, max(0.0, candidate.confidence * assignment_confidence))
            self._tracks[fighter_id] = _TrackState(candidate.bbox, center, velocity, 0)
            if candidate.appearance is not None:
                prior_profile = self._appearance_profiles.get(fighter_id)
                if prior_profile is None or len(prior_profile) != len(candidate.appearance):
                    profile = candidate.appearance
                else:
                    profile = tuple(
                        0.82 * old + 0.18 * new
                        for old, new in zip(prior_profile, candidate.appearance)
                    )
                    total = sum(profile)
                    if total > 1e-9:
                        profile = tuple(value / total for value in profile)
                self._appearance_profiles[fighter_id] = profile
                self._tracks[fighter_id].appearance = profile
            observations.append(
                PoseObservation(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    fighter_id=fighter_id,
                    bbox=candidate.bbox,
                    keypoints=candidate.keypoints,
                    track_confidence=track_confidence,
                    is_scene_cut=scene_cut,
                )
            )

        for fighter_id in list(self._tracks):
            if fighter_id in assigned_ids:
                continue
            state = self._tracks[fighter_id]
            state.missed_frames += 1
            state.center = (state.center[0] + state.velocity[0], state.center[1] + state.velocity[1])
            state.velocity = (state.velocity[0] * 0.85, state.velocity[1] * 0.85)
            if state.missed_frames > self.max_missing_frames:
                del self._tracks[fighter_id]

        observations.sort(key=lambda observation: self.fighter_ids.index(observation.fighter_id))
        return observations


# Friendly aliases used by integration code and notebooks.
StableTwoFighterTracker = TwoFighterTracker
PoseEstimator = RTMLibPoseBackend
