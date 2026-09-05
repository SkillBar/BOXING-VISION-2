"""Pose inference and stable assignment of the two configured fighters."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import math
import os
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .contracts import (
    COCO_KEYPOINT_NAMES,
    BBox,
    Keypoint,
    PoseObservation,
    ReviewStatus,
    SceneState,
)
from .identity import (
    AppearancePart,
    IdentityGallery,
    IdentityMatch,
    IdentityState,
    appearance_distance,
)

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
    # The pose bbox remains available for diagnostics, while ``bbox`` is the
    # original person-detector box whenever RTMLib exposes it.
    pose_bbox: BBox | None = None
    source_track_id: str | int | None = None
    detector_confidence: float | None = None
    pose_confidence: float | None = None
    appearance_parts: dict[str, AppearancePart] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "bbox": self.bbox.to_dict(),
            "keypoints": {
                name: point.to_dict() for name, point in self.keypoints.items()
            },
            "confidence": self.confidence,
            "appearance": list(self.appearance)
            if self.appearance is not None
            else None,
            "pose_bbox": self.pose_bbox.to_dict()
            if self.pose_bbox is not None
            else None,
            "source_track_id": self.source_track_id,
            "detector_confidence": self.detector_confidence,
            "pose_confidence": self.pose_confidence,
            "appearance_parts": {
                name: {
                    "histogram": list(part.histogram),
                    "reliability": part.reliability,
                }
                for name, part in self.appearance_parts.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RawPose:
        return coerce_raw_pose(value)


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


def decode_yolox_people(
    outputs: np.ndarray,
    input_size: tuple[int, int],
    ratio: float,
    *,
    score_threshold: float = 0.1,
    nms_threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw YOLOX person scores without RTMLib's lossy box-only API."""
    predictions = np.asarray(outputs, dtype=np.float32)
    if predictions.ndim != 3 or predictions.shape[0] != 1 or ratio <= 0:
        raise PoseBackendUnavailable("Неизвестный формат выходов YOLOX")
    if predictions.shape[-1] == 5:
        boxes = predictions[0, :, :4].copy() / ratio
        scores = predictions[0, :, 4].copy()
    elif predictions.shape[-1] >= 6:
        grids, strides = [], []
        for stride in (8, 16, 32):
            y, x = np.mgrid[: input_size[0] // stride, : input_size[1] // stride]
            grid = np.stack((x, y), axis=-1).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((len(grid), 1), stride, dtype=np.float32))
        grid, scale = np.concatenate(grids), np.concatenate(strides)
        raw = predictions[0]
        if len(raw) != len(grid):
            raise PoseBackendUnavailable("YOLOX grid не совпадает с формой ONNX output")
        center = (raw[:, :2] + grid) * scale
        size = np.exp(np.clip(raw[:, 2:4], -16, 16)) * scale
        boxes = np.concatenate((center - size / 2, center + size / 2), axis=1) / ratio
        # HumanArt has one class; COCO's person is class 0 as well.
        scores = raw[:, 4] * raw[:, 5]
    else:
        raise PoseBackendUnavailable("YOLOX output не содержит confidence человека")
    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(scores)
        & (scores >= score_threshold)
    )
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], np.clip(scores[valid], 0, 1)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        remaining = order[1:]
        if not remaining.size:
            break
        top_left = np.maximum(boxes[index, :2], boxes[remaining, :2])
        bottom_right = np.minimum(boxes[index, 2:], boxes[remaining, 2:])
        intersection = np.prod(np.maximum(0, bottom_right - top_left), axis=1)
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        iou = intersection / np.maximum(
            1e-9, areas[index] + areas[remaining] - intersection
        )
        order = remaining[iou <= nms_threshold]
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)


def extract_appearance_parts(
    frame: np.ndarray,
    bbox: BBox,
    keypoints: Mapping[str, Keypoint],
    occluders: Sequence[BBox] = (),
    *,
    occluder_keypoints: Sequence[Mapping[str, Keypoint]] | None = None,
) -> dict[str, AppearancePart]:
    """Compare actual foreground intersections, not entire loose person boxes.

    Bbox-only occluders remain a conservative compatibility path for old callers.
    The inference backend supplies pose-local masks, so a referee's surrounding
    empty bbox no longer erases a boxer's visible shorts/torso during a clinch.
    """
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    masks: dict[str, tuple[np.ndarray, float]] = {}

    def polygon(name: str, names: tuple[str, ...]) -> None:
        points = [keypoints.get(key) for key in names]
        if any(point is None or point.score < 0.35 for point in points):
            return
        xy = np.asarray([(point.x, point.y) for point in points], dtype=np.float32)
        center = xy.mean(axis=0)
        xy = center + (xy - center) * 0.78
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(xy.astype(np.int32)), 255)
        masks[name] = mask, min(float(point.score) for point in points)

    polygon("torso", ("left_shoulder", "right_shoulder", "right_hip", "left_hip"))
    shoulders = [keypoints.get(key) for key in ("left_shoulder", "right_shoulder")]
    hips = [keypoints.get(key) for key in ("left_hip", "right_hip")]
    if all(point is not None and point.score >= 0.4 for point in shoulders + hips):
        ls, rs, lh, rh = [np.array([point.x, point.y]) for point in shoulders + hips]
        xy = np.array([lh, rh, rh + 0.20 * (rs - rh), lh + 0.20 * (ls - lh)])
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(xy.astype(np.int32)), 255)
        masks["waistband"] = mask, min(point.score for point in shoulders + hips)
    nose = keypoints.get("nose")
    if nose is not None and nose.score >= 0.5:
        head_width = max(4, bbox.width * 0.13)
        ears = [keypoints.get(key) for key in ("left_ear", "right_ear")]
        if all(point is not None and point.score >= 0.4 for point in ears):
            head_width = max(
                4, math.dist((ears[0].x, ears[0].y), (ears[1].x, ears[1].y)) * 0.5
            )
        mask = np.zeros((height, width), np.uint8)
        cv2.ellipse(
            mask,
            (round(nose.x), round(nose.y - head_width * 0.55)),
            (round(head_width), round(head_width * 0.7)),
            0,
            0,
            360,
            255,
            -1,
        )
        masks["headgear"] = mask, nose.score
    points = [
        keypoints.get(key)
        for key in ("left_hip", "right_hip", "left_knee", "right_knee")
    ]
    if all(point is not None and point.score >= 0.35 for point in points):
        lh, rh, lk, rk = [np.array([point.x, point.y]) for point in points]
        xy = np.array([lh, rh, rh + 0.60 * (rk - rh), lh + 0.60 * (lk - lh)])
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(xy.astype(np.int32)), 255)
        masks["shorts"] = mask, min(float(point.score) for point in points)
    for hand in ("left", "right"):
        wrist, elbow = keypoints.get(f"{hand}_wrist"), keypoints.get(f"{hand}_elbow")
        if wrist is None or elbow is None or min(wrist.score, elbow.score) < 0.45:
            continue
        radius = round(
            max(
                3,
                min(
                    bbox.height * 0.055,
                    math.dist((wrist.x, wrist.y), (elbow.x, elbow.y)) * 0.35,
                ),
            )
        )
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (round(wrist.x), round(wrist.y)), radius, 255, -1)
        masks[f"{hand}_glove"] = mask, min(wrist.score, elbow.score)
    exclusion = np.zeros((height, width), np.uint8)
    if occluder_keypoints is not None:
        for other in occluder_keypoints:
            core = [
                other.get(key)
                for key in ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
            ]
            if all(point is not None and point.score >= 0.4 for point in core):
                xy = np.array([(point.x, point.y) for point in core], np.int32)
                cv2.fillConvexPoly(exclusion, cv2.convexHull(xy), 255)
            for side in ("left", "right"):
                for first, second in (
                    ("shoulder", "elbow"),
                    ("elbow", "wrist"),
                    ("hip", "knee"),
                    ("knee", "ankle"),
                ):
                    a, b = other.get(f"{side}_{first}"), other.get(f"{side}_{second}")
                    if a is not None and b is not None and min(a.score, b.score) >= 0.4:
                        radius = max(3, round(math.dist((a.x, a.y), (b.x, b.y)) * 0.13))
                        cv2.line(
                            exclusion,
                            (round(a.x), round(a.y)),
                            (round(b.x), round(b.y)),
                            255,
                            radius * 2,
                        )
    else:
        for other in occluders:
            x1, x2 = max(0, int(other.x1)), min(width, math.ceil(other.x2))
            y1, y2 = max(0, int(other.y1)), min(height, math.ceil(other.y2))
            exclusion[y1:y2, x1:x2] = 255
    result: dict[str, AppearancePart] = {}
    for name, (mask, score) in masks.items():
        original_area = int(np.count_nonzero(mask))
        if original_area < 32:
            continue
        mask[exclusion > 0] = 0
        remaining_area = int(np.count_nonzero(mask))
        reliability = score * remaining_area / original_area
        if remaining_area < 32 or reliability < 0.25:
            continue
        hs = cv2.calcHist([hsv], [0, 1], mask, [12, 4], [0, 180, 0, 256]).reshape(-1)
        ab = cv2.calcHist([lab], [1, 2], mask, [8, 8], [0, 256, 0, 256]).reshape(-1)
        histogram = np.concatenate((hs / hs.sum() * 0.65, ab / ab.sum() * 0.35))
        result[name] = AppearancePart(
            tuple(float(v) for v in histogram), float(reliability)
        )
    return result


def refresh_appearance_parts(
    frame: np.ndarray, poses: Sequence[RawPose]
) -> list[RawPose]:
    """Re-extract colours from cached real poses, without detector/pose inference.

    Enrollment samples must be refreshed with the same extractor before using
    these descriptors. Returns new records; historical detector caches stay intact.
    """
    return [
        replace(
            pose,
            appearance_parts=extract_appearance_parts(
                frame,
                pose.bbox,
                pose.keypoints,
                occluder_keypoints=[
                    other.keypoints for other in poses if other is not pose
                ],
            ),
        )
        for pose in poses
    ]


class RTMLibPoseBackend:
    """RTMPose/YOLOX body-pose backend with no Ultralytics dependency.

    RTMLib downloads ONNX model files on first use; weight provenance is
    recorded separately from the library's code licence. Model
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
        detector_score_threshold: float = 0.10,
        detector_nms_threshold: float = 0.60,
    ) -> None:
        if mode not in {"lightweight", "balanced", "performance"}:
            raise ValueError(
                "RTMLib mode должен быть lightweight, balanced или performance"
            )
        self.mode = mode
        self.device = _auto_device() if device == "auto" else device
        self.backend = backend
        self.pose_score_threshold = pose_score_threshold
        self.detector_score_threshold = detector_score_threshold
        self.detector_nms_threshold = detector_nms_threshold
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
    def _appearance_descriptor(
        frame: np.ndarray, bbox: BBox
    ) -> tuple[float, ...] | None:
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
        histogram = cv2.calcHist(
            [hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]
        ).reshape(-1)
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
            detector_boxes: np.ndarray | None = None
            detector_scores: np.ndarray | None = None
            detector = getattr(model, "det_model", None)
            pose_model = getattr(model, "pose_model", None)
            if (
                not bool(getattr(model, "one_stage", False))
                and callable(detector)
                and callable(pose_model)
            ):
                if callable(getattr(detector, "preprocess", None)) and callable(
                    getattr(detector, "inference", None)
                ):
                    detector_input, ratio = detector.preprocess(frame)
                    outputs = detector.inference(detector_input)
                    detector_boxes, detector_scores = decode_yolox_people(
                        np.asarray(outputs[0]),
                        detector.model_input_size,
                        ratio,
                        score_threshold=self.detector_score_threshold,
                        nms_threshold=self.detector_nms_threshold,
                    )
                else:
                    # Compatibility for non-YOLOX test/custom adapters. Missing
                    # confidence remains explicitly unavailable, never pose-derived.
                    detector_boxes = np.asarray(detector(frame), dtype=np.float32)
                    if detector_boxes.ndim == 2 and detector_boxes.shape[1] >= 5:
                        detector_scores = detector_boxes[:, 4].copy()
                        detector_boxes = detector_boxes[:, :4]
                if detector_boxes.size == 0:
                    return []
                if detector_boxes.ndim == 1 and detector_boxes.size:
                    detector_boxes = detector_boxes[None, ...]
                keypoints_array, scores_array = pose_model(  # type: ignore[operator]
                    frame,
                    bboxes=detector_boxes,
                )
            else:
                keypoints_array, scores_array = model(frame)  # type: ignore[operator]
        except PoseBackendUnavailable:
            raise
        except Exception as exc:
            raise PoseBackendUnavailable(f"Ошибка RTMPose inference: {exc}") from exc
        return self._decode_poses(
            frame, keypoints_array, scores_array, detector_boxes, detector_scores
        )

    def infer_in_boxes(
        self, frame: np.ndarray, bboxes: Sequence[BBox]
    ) -> list[RawPose]:
        """Pose refinement only; supplied confirmed boxes never run detection again."""
        if not bboxes:
            return []
        model = self._ensure_model()
        pose_model = getattr(model, "pose_model", None)
        if not callable(pose_model):
            raise PoseBackendUnavailable(
                "Pose-only refinement требует RTMLib pose_model"
            )
        boxes = np.asarray(
            [[box.x1, box.y1, box.x2, box.y2] for box in bboxes], dtype=np.float32
        )
        coordinates, scores = pose_model(frame, bboxes=boxes)
        return self._decode_poses(
            frame,
            coordinates,
            scores,
            boxes,
            np.asarray([box.score for box in bboxes], dtype=np.float32),
        )

    def _decode_poses(
        self,
        frame: np.ndarray,
        keypoints_array: np.ndarray,
        scores_array: np.ndarray,
        detector_boxes: np.ndarray | None,
        detector_scores: np.ndarray | None,
    ) -> list[RawPose]:
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
        for pose_index, (coordinates, scores) in enumerate(
            zip(keypoints_array, scores_array)
        ):
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
            if len(valid_xy) < 5 and detector_boxes is None:
                continue
            if not valid_xy:
                valid_xy = [(0.0, 0.0)]
            xs = [point[0] for point in valid_xy]
            ys = [point[1] for point in valid_xy]
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)
            pad_x = max(4.0, x_span * 0.12)
            pad_y = max(4.0, y_span * 0.08)
            pose_bbox = BBox(
                x1=max(0.0, min(xs) - pad_x),
                y1=max(0.0, min(ys) - pad_y),
                x2=min(float(width - 1), max(xs) + pad_x),
                y2=min(float(height - 1), max(ys) + pad_y),
                score=float(np.mean(valid_scores)) if valid_scores else 0.0,
            )
            bbox = pose_bbox
            if detector_boxes is not None and pose_index < len(detector_boxes):
                raw_box = np.asarray(detector_boxes[pose_index]).reshape(-1)
                if raw_box.size >= 4 and np.all(np.isfinite(raw_box[:4])):
                    detector_bbox = BBox(
                        x1=max(0.0, min(float(width - 1), float(raw_box[0]))),
                        y1=max(0.0, min(float(height - 1), float(raw_box[1]))),
                        x2=max(0.0, min(float(width - 1), float(raw_box[2]))),
                        y2=max(0.0, min(float(height - 1), float(raw_box[3]))),
                        score=(
                            float(detector_scores[pose_index])
                            if detector_scores is not None
                            else 0.0
                        ),
                    )
                    # Never expand a small detector box into an unrelated
                    # pose-derived person. Keep the genuine detection or drop it.
                    bbox = detector_bbox
            if bbox.area < 400:
                continue
            poses.append(
                RawPose(
                    bbox=bbox,
                    keypoints=mapped,
                    confidence=bbox.score,
                    appearance=self._appearance_descriptor(frame, bbox),
                    pose_bbox=pose_bbox,
                    detector_confidence=(
                        float(detector_scores[pose_index])
                        if detector_scores is not None
                        else None
                    ),
                    pose_confidence=float(np.mean(valid_scores))
                    if valid_scores
                    else 0.0,
                )
            )
        for pose in poses:
            pose.appearance_parts = extract_appearance_parts(
                frame,
                pose.bbox,
                pose.keypoints,
                [other.bbox for other in poses if other is not pose],
                occluder_keypoints=[
                    other.keypoints for other in poses if other is not pose
                ],
            )
        return poses


def create_pose_backend(
    backend: str = "auto",
    *,
    strict: bool = True,
    mode: str = "lightweight",
    device: str = "auto",
    pose_score_threshold: float = 0.25,
    detector_score_threshold: float = 0.10,
    detector_nms_threshold: float = 0.60,
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
        detector_score_threshold=detector_score_threshold,
        detector_nms_threshold=detector_nms_threshold,
    )


def _coerce_keypoint(value: object) -> Keypoint:
    if isinstance(value, Keypoint):
        return value
    if isinstance(value, Mapping):
        return Keypoint(
            float(value["x"]), float(value["y"]), float(value.get("score", 1.0))
        )
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
    return BBox(
        min(xs),
        min(ys),
        max(xs),
        max(ys),
        float(np.mean([point.score for point in visible])),
    )


def coerce_raw_pose(value: RawPose | PoseObservation | Mapping[str, object]) -> RawPose:
    """Accept lightweight dictionaries as well as the internal dataclasses."""

    if isinstance(value, RawPose):
        return value
    if isinstance(value, PoseObservation):
        return RawPose(
            value.bbox,
            value.keypoints,
            value.track_confidence,
            source_track_id=value.source_track_id,
            detector_confidence=getattr(value, "detector_confidence", None),
            pose_confidence=getattr(value, "pose_confidence", None),
        )
    if not isinstance(value, Mapping):
        raise TypeError(f"Неподдерживаемый формат pose: {type(value).__name__}")
    raw_keypoints = value.get("keypoints")
    if not isinstance(raw_keypoints, Mapping):
        raise TypeError("Pose должен содержать keypoints")
    keypoints = {
        str(name): _coerce_keypoint(point) for name, point in raw_keypoints.items()
    }
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
        if isinstance(raw_appearance, Sequence)
        and not isinstance(raw_appearance, (str, bytes))
        else None
    )
    raw_pose_bbox = value.get("pose_bbox")
    pose_bbox: BBox | None = None
    if isinstance(raw_pose_bbox, BBox):
        pose_bbox = raw_pose_bbox
    elif isinstance(raw_pose_bbox, Mapping):
        pose_bbox = BBox(
            float(raw_pose_bbox["x1"]),
            float(raw_pose_bbox["y1"]),
            float(raw_pose_bbox["x2"]),
            float(raw_pose_bbox["y2"]),
            float(raw_pose_bbox.get("score", 1.0)),
        )
    source_track_id = value.get("source_track_id")
    if not isinstance(source_track_id, (str, int)):
        source_track_id = None
    parts: dict[str, AppearancePart] = {}
    for name, part in (value.get("appearance_parts") or {}).items():
        if isinstance(part, AppearancePart):
            parts[str(name)] = part
        elif isinstance(part, Mapping):
            histogram = tuple(float(item) for item in part["histogram"])
            reliability = float(part["reliability"])
            if not np.isfinite(histogram).all() or not 0 <= reliability <= 1:
                raise ValueError("Некорректный appearance part")
            parts[str(name)] = AppearancePart(histogram, reliability)
    return RawPose(
        bbox,
        keypoints,
        min(1.0, max(0.0, confidence)),
        appearance,
        pose_bbox,
        source_track_id,
        float(value["detector_confidence"])
        if value.get("detector_confidence") is not None
        else None,
        float(value["pose_confidence"])
        if value.get("pose_confidence") is not None
        else None,
        parts,
    )


@dataclass(slots=True)
class _TrackState:
    bbox: BBox
    center: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    missed_frames: int = 0
    appearance: tuple[float, ...] | None = None
    source_track_id: str | int | None = None
    stable_frames: int = 1


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
        gallery_max_distance: float = 0.35,
        gallery_min_margin: float = 0.12,
        adaptive_identity_confidence_min: float = 0.90,
        adaptive_identity_margin_min: float = 0.20,
        reacquisition_votes: int = 3,
        reacquisition_window: int = 5,
        core_appearances: Mapping[str | IdentityState, Sequence[Sequence[float]]]
        | None = None,
        negative_appearances: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if len(fighter_ids) != 2 or fighter_ids[0] == fighter_ids[1]:
            raise ValueError("Нужны два разных fighter_id")
        self.fighter_ids = fighter_ids
        self.anchors = dict(anchors or {})
        self.max_missing_frames = max_missing_frames
        self.max_assignment_cost = max_assignment_cost
        self.candidate_limit = max(2, candidate_limit)
        self.gallery_max_distance = gallery_max_distance
        self.gallery_min_margin = gallery_min_margin
        self.adaptive_identity_confidence_min = adaptive_identity_confidence_min
        self.adaptive_identity_margin_min = adaptive_identity_margin_min
        self.reacquisition_votes = max(1, int(reacquisition_votes))
        self.reacquisition_window = max(
            self.reacquisition_votes,
            int(reacquisition_window),
        )
        self._tracks: dict[str, _TrackState] = {}
        # CLI/cache v1 may not carry a motion source ID. These short-lived,
        # geometry-matched vote tokens are private and never exported as BoT-SORT IDs.
        self._legacy_vote_tracks: dict[int, tuple[RawPose, int]] = {}
        self._legacy_vote_candidates: dict[int, int] = {}
        self._legacy_vote_serial = 0
        normalized_core: dict[IdentityState, Sequence[Sequence[float]]] = {}
        for raw_identity, descriptors in (core_appearances or {}).items():
            if raw_identity in {self.fighter_ids[0], IdentityState.FIGHTER_A}:
                state = IdentityState.FIGHTER_A
            elif raw_identity in {self.fighter_ids[1], IdentityState.FIGHTER_B}:
                state = IdentityState.FIGHTER_B
            else:
                raise ValueError(
                    f"Неизвестная identity для core gallery: {raw_identity}"
                )
            normalized_core[state] = descriptors
        self._gallery = IdentityGallery(
            core=normalized_core,
            negative=negative_appearances,
            max_distance=gallery_max_distance,
            min_margin=gallery_min_margin,
            adaptive_confidence_min=adaptive_identity_confidence_min,
            adaptive_margin_min=adaptive_identity_margin_min,
        )
        self._shot_id = 0
        self._needs_reacquisition: set[str] = set()
        self._reacquisition_history: dict[str, deque[str | int | None]] = {
            fighter_id: deque(maxlen=self.reacquisition_window)
            for fighter_id in self.fighter_ids
        }
        self._last_identity_states: dict[str, IdentityState] = {
            fighter_id: IdentityState.UNKNOWN for fighter_id in self.fighter_ids
        }
        # Automatic enrollment is deliberately colour-led and multi-frame.
        # It never falls back to largest-person or left/right ordering.  Until
        # both red/blue candidates agree for three frames, both roles remain
        # UNKNOWN and no punch event can be emitted.
        self._auto_enrollment_samples: dict[IdentityState, deque[tuple[float, ...]]] = {
            IdentityState.FIGHTER_A: deque(maxlen=5),
            IdentityState.FIGHTER_B: deque(maxlen=5),
        }
        self._auto_enrollment_required = 3
        self._ring_rois: dict[int, tuple[tuple[float, float], ...]] = {}
        self._identity_overrides: dict[str, IdentityState] = {}
        self.last_diagnostics: list[dict[str, object]] = []

    @property
    def gallery(self) -> IdentityGallery:
        return self._gallery

    def enroll(
        self,
        samples: Mapping[str, Sequence[RawPose]],
        negatives: Sequence[RawPose] = (),
    ) -> None:
        """Freeze three or more confirmed samples per role, independent of screen side."""
        first, second = (samples.get(role, ()) for role in self.fighter_ids)
        if any(a is b for a in first for b in second):
            raise ValueError("Один detection нельзя выбрать для обоих бойцов")
        for fighter_id in self.fighter_ids:
            items = list(samples.get(fighter_id, ()))
            if len(items) < 3 or any(item.appearance is None for item in items):
                raise ValueError(
                    "Калибровка требует три пригодных образца каждого бойца"
                )
            state = self._identity_state(fighter_id)
            if state in self._gallery.core:
                raise RuntimeError(
                    "Core gallery уже зафиксирована; создайте новую калибровку"
                )
        for fighter_id in self.fighter_ids:
            items = list(samples[fighter_id])
            state = self._identity_state(fighter_id)
            self._gallery.set_core_once(state, [item.appearance for item in items])
            self._gallery.set_core_parts_once(
                state, [item.appearance_parts for item in items]
            )
        for item in negatives:
            if item.appearance is not None:
                self._gallery.add_negative(item.appearance)
            self._gallery.add_negative_parts(item.appearance_parts)
        self.anchors.clear()
        self._needs_reacquisition = set(self.fighter_ids)

    def set_ring_roi(
        self,
        points: Sequence[tuple[float, float]],
        *,
        frame_size: tuple[int, int],
        shot_id: int = 0,
    ) -> None:
        if len(points) != 4 or any(
            not 0 <= coordinate <= 1 for point in points for coordinate in point
        ):
            raise ValueError("Зона ринга требует четыре нормализованные точки")
        width, height = frame_size
        polygon = np.asarray(
            [(x * width, y * height) for x, y in points], dtype=np.float32
        )
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 16:
            raise ValueError(
                "Точки ринга должны задавать выпуклый четырёхугольник по порядку"
            )
        self._ring_rois[int(shot_id)] = tuple((float(x), float(y)) for x, y in polygon)

    def set_identity_overrides(self, overrides: Mapping[str, str]) -> None:
        self._identity_overrides = {
            str(key): IdentityState(value) for key, value in overrides.items()
        }

    def export_identity_profile(self) -> dict[str, object]:
        return {
            **self._gallery.export(),
            "ring_rois": self._ring_rois,
            "identity_overrides": {
                key: str(value) for key, value in self._identity_overrides.items()
            },
        }

    def import_identity_profile(self, profile: Mapping[str, object]) -> None:
        """Hydrate an empty tracker from the first-pass cache's actual enrollment."""
        if self._gallery.core:
            raise RuntimeError(
                "Нельзя перезаписать core gallery; создайте новый tracker"
            )
        gallery = IdentityGallery.from_export(profile)
        polygons = {}
        for shot, points in profile.get("ring_rois", {}).items():
            polygon = np.asarray(points, dtype=np.float32)
            if (
                polygon.shape != (4, 2)
                or not np.isfinite(polygon).all()
                or not cv2.isContourConvex(polygon)
                or abs(cv2.contourArea(polygon)) < 16
            ):
                raise ValueError("Некорректный сохранённый ROI ринга")
            polygons[int(shot)] = tuple((float(x), float(y)) for x, y in polygon)
        overrides = {
            str(key): IdentityState(value)
            for key, value in profile.get("identity_overrides", {}).items()
        }
        self._gallery = gallery
        self._ring_rois = polygons
        self._identity_overrides = overrides
        self.reset(keep_anchors=False)

    def score_candidate(
        self, candidate: RawPose, *, shot_id: int | None = None
    ) -> IdentityMatch:
        shot = self._shot_id if shot_id is None else int(shot_id)
        override = self._identity_overrides.get(
            f"shot-{shot}-track-{candidate.source_track_id}"
        )
        if override is not None:
            return IdentityMatch(
                override,
                0.0,
                1.0,
                1.0,
                override in {IdentityState.FIGHTER_A, IdentityState.FIGHTER_B},
                "user_override",
            )
        polygon = self._ring_rois.get(shot)
        if polygon is not None:
            feet = [
                candidate.keypoints.get(key) for key in ("left_ankle", "right_ankle")
            ]
            visible = [
                point for point in feet if point is not None and point.score >= 0.4
            ]
            if visible and all(
                cv2.pointPolygonTest(
                    np.asarray(polygon, dtype=np.float32), (point.x, point.y), False
                )
                < 0
                for point in visible
            ):
                return IdentityMatch(
                    IdentityState.OTHER, 1.0, 0.0, 0.0, False, "outside_ring"
                )
        return self._gallery.match(candidate.appearance, candidate.appearance_parts)

    def reset(self, *, keep_anchors: bool = True, keep_appearance: bool = True) -> None:
        self._tracks.clear()
        self._legacy_vote_tracks.clear()
        self._legacy_vote_candidates.clear()
        self._needs_reacquisition.clear()
        for history in self._reacquisition_history.values():
            history.clear()
        for samples in self._auto_enrollment_samples.values():
            samples.clear()
        if not keep_anchors:
            self.anchors.clear()
        if not keep_appearance:
            self._gallery = IdentityGallery(
                max_distance=self.gallery_max_distance,
                min_margin=self.gallery_min_margin,
                adaptive_confidence_min=self.adaptive_identity_confidence_min,
                adaptive_margin_min=self.adaptive_identity_margin_min,
            )
        self._last_identity_states = {
            fighter_id: IdentityState.UNKNOWN for fighter_id in self.fighter_ids
        }

    def set_anchors(
        self, fighter_a: tuple[float, float], fighter_b: tuple[float, float]
    ) -> None:
        self.anchors = {self.fighter_ids[0]: fighter_a, self.fighter_ids[1]: fighter_b}
        self._tracks.clear()

    def clear_anchors(self) -> None:
        """Keep active tracks but stop reusing first-shot screen positions."""

        self.anchors.clear()

    @property
    def shot_id(self) -> int:
        """The local shot epoch; motion tracks never cross this boundary."""

        return self._shot_id

    @property
    def core_gallery(self) -> Mapping[IdentityState, tuple[tuple[float, ...], ...]]:
        return self._gallery.core

    @property
    def negative_gallery(self) -> tuple[tuple[float, ...], ...]:
        return self._gallery.negative

    @property
    def adaptive_gallery(self) -> Mapping[IdentityState, tuple[tuple[float, ...], ...]]:
        return {
            state: self._gallery.adaptive_samples(state)
            for state in (IdentityState.FIGHTER_A, IdentityState.FIGHTER_B)
        }

    @property
    def last_identity_states(self) -> Mapping[str, IdentityState]:
        return dict(self._last_identity_states)

    def add_negative_appearance(self, descriptor: Sequence[float]) -> None:
        self._gallery.add_negative(descriptor)

    def _identity_state(self, fighter_id: str) -> IdentityState:
        if fighter_id == self.fighter_ids[0]:
            return IdentityState.FIGHTER_A
        if fighter_id == self.fighter_ids[1]:
            return IdentityState.FIGHTER_B
        raise KeyError(fighter_id)

    def _fighter_id(self, state: IdentityState) -> str | None:
        if state == IdentityState.FIGHTER_A:
            return self.fighter_ids[0]
        if state == IdentityState.FIGHTER_B:
            return self.fighter_ids[1]
        return None

    @staticmethod
    def _quality(pose: RawPose) -> float:
        visible = sum(point.score >= 0.25 for point in pose.keypoints.values())
        return pose.bbox.area * max(0.05, pose.confidence) * min(1.0, visible / 10.0)

    @staticmethod
    def _assignment_cost(state: _TrackState, candidate: RawPose) -> float:
        predicted = (
            state.center[0] + state.velocity[0],
            state.center[1] + state.velocity[1],
        )
        center = candidate.bbox.center
        reference_scale = max(25.0, state.bbox.height, candidate.bbox.height)
        distance = math.dist(predicted, center) / reference_scale
        area_ratio = (candidate.bbox.area + 1.0) / (state.bbox.area + 1.0)
        scale_penalty = abs(math.log(area_ratio)) * 0.28
        confidence_penalty = (1.0 - candidate.confidence) * 0.25
        appearance_penalty = appearance_distance(
            state.appearance,
            candidate.appearance,
        )
        return distance + scale_penalty + confidence_penalty + 0.72 * appearance_penalty

    @staticmethod
    def _appearance_distance(
        first: tuple[float, ...] | None,
        second: tuple[float, ...] | None,
    ) -> float:
        # Kept as a compatibility hook for notebooks that used the old helper.
        return appearance_distance(first, second)

    @staticmethod
    def _bbox_iou(first: BBox, second: BBox) -> float:
        x1 = max(first.x1, second.x1)
        y1 = max(first.y1, second.y1)
        x2 = min(first.x2, second.x2)
        y2 = min(first.y2, second.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = first.area + second.area - intersection
        return intersection / union if union > 1e-9 else 0.0

    def _profile_assignment(
        self,
        candidates: list[RawPose],
        candidate_indices: list[int],
    ) -> dict[str, int] | None:
        proposals: list[tuple[float, str, int]] = []
        for index in candidate_indices:
            match = self.score_candidate(candidates[index])
            fighter_id = self._fighter_id(match.state)
            if match.accepted and fighter_id is not None:
                proposals.append((match.distance, fighter_id, index))
        result: dict[str, int] = {}
        used: set[int] = set()
        for _, fighter_id, index in sorted(proposals):
            if fighter_id in result or index in used:
                continue
            result[fighter_id] = index
            used.add(index)
        return result or None

    def _anchor_assignment(self, candidates: list[RawPose]) -> dict[str, int]:
        """Resolve only candidates close enough to an explicit user anchor.

        Missing is intentionally cheaper than stretching an anchor to a large
        foreground referee.  This is the opposite of the old forced-fill rule.
        """

        options: dict[str, list[tuple[float, int]]] = {}
        for fighter_id in self.fighter_ids:
            anchor = self.anchors.get(fighter_id)
            if anchor is None:
                options[fighter_id] = []
                continue
            scored: list[tuple[float, int]] = []
            for index, candidate in enumerate(candidates):
                # A huge foreground referee must not become artificially close
                # merely because the old denominator used bbox height.  The
                # click must be inside the detection and near its centre when
                # measured against the narrower person dimension.
                if not (
                    candidate.bbox.x1 <= anchor[0] <= candidate.bbox.x2
                    and candidate.bbox.y1 <= anchor[1] <= candidate.bbox.y2
                ):
                    continue
                reference_scale = max(
                    25.0,
                    min(candidate.bbox.width, candidate.bbox.height),
                )
                distance = math.dist(anchor, candidate.bbox.center) / reference_scale
                if distance <= 0.58:
                    scored.append((distance, index))
            options[fighter_id] = scored

        unmatched_cost = 0.52
        best: tuple[float, dict[str, int]] | None = None
        for first in [None, *options[self.fighter_ids[0]]]:
            for second in [None, *options[self.fighter_ids[1]]]:
                if first is not None and second is not None and first[1] == second[1]:
                    continue
                mapping: dict[str, int] = {}
                total = 0.0
                for fighter_id, selection in zip(self.fighter_ids, (first, second)):
                    if selection is None:
                        total += unmatched_cost
                    else:
                        total += selection[0]
                        mapping[fighter_id] = selection[1]
                if best is None or total < best[0]:
                    best = (total, mapping)
        return dict(best[1]) if best is not None else {}

    @staticmethod
    def _corner_colour_scores(
        descriptor: Sequence[float] | None,
    ) -> tuple[float, float]:
        """Return saturated red/blue mass from the RTMLib HSV descriptor."""

        if descriptor is None or len(descriptor) != 48:
            return 0.0, 0.0
        histogram = np.asarray(descriptor, dtype=np.float32).reshape(12, 4)
        red = float(histogram[[0, 11], 2:].sum())
        blue = float(histogram[6:9, 2:].sum())
        return red, blue

    def _clear_auto_enrollment(self) -> None:
        for samples in self._auto_enrollment_samples.values():
            samples.clear()

    def _auto_colour_enrollment(
        self,
        candidates: list[RawPose],
    ) -> dict[str, int]:
        """Bootstrap A/B only from repeated, separated red/blue evidence.

        This is the safe ``auto_confirm`` proposal path.  It can identify the
        two colour corners even with a referee in frame, but refuses neutral,
        ambiguous or same-colour pairs and therefore preserves UNKNOWN.
        """

        evidence = [
            (index, *self._corner_colour_scores(candidate.appearance))
            for index, candidate in enumerate(candidates)
            if candidate.appearance is not None and candidate.confidence >= 0.45
        ]
        if len(evidence) < 2:
            self._clear_auto_enrollment()
            return {}

        red_order = sorted(evidence, key=lambda item: item[1], reverse=True)
        blue_order = sorted(evidence, key=lambda item: item[2], reverse=True)
        red_index, red_score, red_blue_score = red_order[0]
        blue_index, blue_red_score, blue_score = blue_order[0]
        red_runner_up = red_order[1][1]
        blue_runner_up = blue_order[1][2]
        valid = (
            red_index != blue_index
            and red_score >= 0.035
            and blue_score >= 0.035
            and red_score - red_runner_up >= 0.025
            and blue_score - blue_runner_up >= 0.025
            and red_score - red_blue_score >= 0.02
            and blue_score - blue_red_score >= 0.02
            and appearance_distance(
                candidates[red_index].appearance,
                candidates[blue_index].appearance,
            )
            >= 0.12
        )
        if not valid:
            self._clear_auto_enrollment()
            return {}

        mapping = {
            self.fighter_ids[0]: red_index,
            self.fighter_ids[1]: blue_index,
        }
        for fighter_id, candidate_index in mapping.items():
            descriptor = candidates[candidate_index].appearance
            if descriptor is not None:
                self._auto_enrollment_samples[self._identity_state(fighter_id)].append(
                    tuple(descriptor)
                )

        if any(
            len(self._auto_enrollment_samples[state]) < self._auto_enrollment_required
            for state in (IdentityState.FIGHTER_A, IdentityState.FIGHTER_B)
        ):
            return {}

        if not self._gallery.core:
            self._gallery.set_core_once(
                IdentityState.FIGHTER_A,
                tuple(self._auto_enrollment_samples[IdentityState.FIGHTER_A]),
            )
            self._gallery.set_core_once(
                IdentityState.FIGHTER_B,
                tuple(self._auto_enrollment_samples[IdentityState.FIGHTER_B]),
            )
            assigned = set(mapping.values())
            for index, candidate in enumerate(candidates):
                if index not in assigned and candidate.appearance is not None:
                    self._gallery.add_negative(candidate.appearance)
        return mapping

    def _initial_assignment(self, candidates: list[RawPose]) -> dict[str, int]:
        quality_order = sorted(
            range(len(candidates)),
            key=lambda index: self._quality(candidates[index]),
            reverse=True,
        )
        selected_indices = quality_order[: self.candidate_limit]
        if not selected_indices:
            return {}
        if self.anchors and self._shot_id == 0:
            return self._anchor_assignment(candidates)
        profile_candidates = [
            index
            for index in selected_indices
            if candidates[index].appearance is not None
        ]
        profile_result = self._profile_assignment(candidates, profile_candidates)
        if profile_result:
            return self._apply_reacquisition_gate(profile_result, candidates)
        if not self._gallery.core and self._shot_id == 0:
            return self._auto_colour_enrollment(candidates)
        return {}

    def _blocked_by_gallery(self, fighter_id: str, candidate: RawPose) -> bool:
        if not self._gallery.core:
            return False
        if candidate.appearance is None:
            return True
        match = self.score_candidate(candidate)
        expected = self._identity_state(fighter_id)
        return not (match.accepted and match.state == expected)

    def _tracked_assignment(self, candidates: list[RawPose]) -> dict[str, int]:
        active_ids = [
            fighter_id for fighter_id in self.fighter_ids if fighter_id in self._tracks
        ]
        if not active_ids:
            return self._initial_assignment(candidates)
        best: tuple[float, dict[str, int]] | None = None
        unmatched_cost = min(1.05, self.max_assignment_cost * 0.36)
        choices: list[int | None] = [None, *range(len(candidates))]
        for selected in itertools.product(choices, repeat=len(active_ids)):
            used = [item for item in selected if item is not None]
            if len(used) != len(set(used)):
                continue
            total = 0.0
            mapping: dict[str, int] = {}
            valid = True
            for fighter_id, candidate_index in zip(active_ids, selected):
                if candidate_index is None:
                    total += unmatched_cost
                    continue
                candidate = candidates[candidate_index]
                if self._blocked_by_gallery(fighter_id, candidate):
                    valid = False
                    break
                cost = self._assignment_cost(self._tracks[fighter_id], candidate)
                if cost > self.max_assignment_cost:
                    valid = False
                    break
                total += cost
                mapping[fighter_id] = candidate_index
            if valid and (best is None or total < best[0]):
                best = (total, mapping)
        assignments = dict(best[1]) if best is not None else {}

        available = [
            index
            for index in range(len(candidates))
            if index not in assignments.values()
        ]
        inactive = [
            fighter_id
            for fighter_id in self.fighter_ids
            if fighter_id not in self._tracks
        ]
        if self.anchors and self._shot_id == 0:
            anchor_candidates = [candidates[index] for index in available]
            local_assignment = self._anchor_assignment(anchor_candidates)
            profile_assignment = {
                fighter_id: available[local_index]
                for fighter_id, local_index in local_assignment.items()
                if fighter_id in inactive
            }
        else:
            profile_assignment = self._profile_assignment(candidates, available) or {}
        profile_assignment = {
            fighter_id: index
            for fighter_id, index in profile_assignment.items()
            if fighter_id in inactive
        }
        assignments.update(
            self._apply_reacquisition_gate(profile_assignment, candidates)
        )
        return assignments

    def _reacquisition_token(
        self, fighter_id: str, candidate: RawPose
    ) -> str | int | None:
        if candidate.source_track_id is not None:
            return candidate.source_track_id
        # A role is not a track ID. Votes from different people must never combine.
        token = self._legacy_vote_candidates.get(id(candidate))
        return f"legacy-vote-{token}" if token is not None else None

    def _prepare_legacy_vote_tokens(
        self, candidates: Sequence[RawPose], timestamp_ms: int
    ) -> None:
        self._legacy_vote_candidates = {}
        self._legacy_vote_tracks = {
            token: value
            for token, value in self._legacy_vote_tracks.items()
            if 0 <= timestamp_ms - value[1] <= 250
        }
        options = []
        for index, candidate in enumerate(candidates):
            if candidate.source_track_id is not None:
                continue
            for token, (previous, _) in self._legacy_vote_tracks.items():
                overlap = self._bbox_iou(previous.bbox, candidate.bbox)
                distance = appearance_distance(
                    previous.appearance, candidate.appearance
                )
                if overlap >= 0.55 and distance <= 0.25:
                    options.append((1.0 - overlap + distance, index, token))
        used_indices, used_tokens = set(), set()
        for _, index, token in sorted(options):
            if index not in used_indices and token not in used_tokens:
                self._legacy_vote_candidates[id(candidates[index])] = token
                self._legacy_vote_tracks[token] = (candidates[index], timestamp_ms)
                used_indices.add(index)
                used_tokens.add(token)
        for index, candidate in enumerate(candidates):
            if candidate.source_track_id is None and index not in used_indices:
                self._legacy_vote_serial += 1
                token = self._legacy_vote_serial
                self._legacy_vote_candidates[id(candidate)] = token
                self._legacy_vote_tracks[token] = (candidate, timestamp_ms)

    def _apply_reacquisition_gate(
        self,
        proposals: Mapping[str, int],
        candidates: Sequence[RawPose],
    ) -> dict[str, int]:
        confirmed: dict[str, int] = {}
        for fighter_id in self.fighter_ids:
            if fighter_id not in self._needs_reacquisition:
                if fighter_id in proposals:
                    confirmed[fighter_id] = proposals[fighter_id]
                continue
            candidate_index = proposals.get(fighter_id)
            token: str | int | None = None
            if candidate_index is not None:
                token = self._reacquisition_token(
                    fighter_id,
                    candidates[candidate_index],
                )
            history = self._reacquisition_history[fighter_id]
            history.append(token)
            if token is None:
                continue
            votes = Counter(item for item in history if item is not None)
            if votes[token] >= self.reacquisition_votes:
                confirmed[fighter_id] = candidate_index
                self._needs_reacquisition.discard(fighter_id)
                history.clear()
        return confirmed

    def _begin_new_shot(self) -> None:
        self._shot_id += 1
        self._tracks.clear()
        self._legacy_vote_tracks.clear()
        self._legacy_vote_candidates.clear()
        self._needs_reacquisition = set(self.fighter_ids)
        for history in self._reacquisition_history.values():
            history.clear()
        self._clear_auto_enrollment()
        self._last_identity_states = {
            fighter_id: IdentityState.UNKNOWN for fighter_id in self.fighter_ids
        }

    def _enroll_calibration_evidence(
        self,
        assignments: Mapping[str, int],
        candidates: Sequence[RawPose],
    ) -> None:
        # Enrollment must describe a clean simultaneous pair over multiple
        # frames.  A partial frame can no longer freeze the referee into A's
        # immutable gallery when the real red fighter is temporarily missing.
        if set(assignments) != set(self.fighter_ids):
            self._clear_auto_enrollment()
            return
        for fighter_id, candidate_index in assignments.items():
            appearance = candidates[candidate_index].appearance
            if appearance is None:
                self._clear_auto_enrollment()
                return
            self._auto_enrollment_samples[self._identity_state(fighter_id)].append(
                tuple(appearance)
            )
        if any(
            len(self._auto_enrollment_samples[state]) < self._auto_enrollment_required
            for state in (IdentityState.FIGHTER_A, IdentityState.FIGHTER_B)
        ):
            return
        if not self._gallery.core:
            self._gallery.set_core_once(
                IdentityState.FIGHTER_A,
                tuple(self._auto_enrollment_samples[IdentityState.FIGHTER_A]),
            )
            self._gallery.set_core_once(
                IdentityState.FIGHTER_B,
                tuple(self._auto_enrollment_samples[IdentityState.FIGHTER_B]),
            )
        assigned_indices = set(assignments.values())
        for index, candidate in enumerate(candidates):
            if index not in assigned_indices and candidate.appearance is not None:
                self._gallery.add_negative(candidate.appearance)

    def process(
        self,
        frame_index: int,
        timestamp_ms: int,
        poses: Sequence[RawPose | PoseObservation | Mapping[str, object]],
        scene_cut: bool = False,
        *,
        active_fight: bool = True,
        shot_id: int | None = None,
        decoded_source_roles: Mapping[str, str] | None = None,
        identity_votes_confirmed: bool = False,
        decoded_matches: Mapping[str, IdentityMatch] | None = None,
    ) -> list[PoseObservation]:
        if scene_cut:
            self._begin_new_shot()
            if shot_id is not None:
                self._shot_id = int(shot_id)
        elif shot_id is not None and shot_id != self._shot_id:
            self._begin_new_shot()
            self._shot_id = int(shot_id)
        candidates = [coerce_raw_pose(pose) for pose in poses]
        candidates = [candidate for candidate in candidates if candidate.bbox.area > 0]
        candidates.sort(key=self._quality, reverse=True)

        def candidate_match(candidate: RawPose) -> IdentityMatch:
            raw_match = self.score_candidate(candidate)
            pooled = (decoded_matches or {}).get(str(candidate.source_track_id))
            return (
                pooled
                if pooled is not None
                and (
                    raw_match.state != IdentityState.OTHER
                    or pooled.reason == "segment_user_override"
                    or (
                        raw_match.reason == "negative_gallery_closer"
                        and pooled.reason
                        in {
                            "observed_continuity",
                            "source_reacquired",
                            "negative_gallery_pending",
                        }
                    )
                )
                else raw_match
            )

        if decoded_source_roles is None:
            candidates = candidates[: self.candidate_limit]
            self._prepare_legacy_vote_tokens(candidates, timestamp_ms)
        assignments = (
            (
                self._tracked_assignment(candidates)
                if self._tracks
                else self._initial_assignment(candidates)
            )
            if active_fight and decoded_source_roles is None
            else {}
        )
        if decoded_source_roles is not None:
            assignments = {}
            if active_fight:
                for index, candidate in enumerate(candidates):
                    role = decoded_source_roles.get(str(candidate.source_track_id))
                    if role in self.fighter_ids and role not in assignments:
                        match = candidate_match(candidate)
                        if match.accepted and self._fighter_id(match.state) == role:
                            assignments[role] = index
                if not identity_votes_confirmed:
                    assignments = self._apply_reacquisition_gate(
                        assignments, candidates
                    )

        if self.anchors and self._shot_id == 0:
            self._enroll_calibration_evidence(assignments, candidates)

        gallery_ready = set(self._gallery.core) >= {
            IdentityState.FIGHTER_A,
            IdentityState.FIGHTER_B,
        }

        overlap_or_clinch = False
        assigned_indices = set(assignments.values())
        for first_index, second_index in itertools.combinations(
            range(len(candidates)), 2
        ):
            if not {first_index, second_index} & assigned_indices:
                continue
            if (
                self._bbox_iou(
                    candidates[first_index].bbox,
                    candidates[second_index].bbox,
                )
                > 0.35
            ):
                overlap_or_clinch = True
                break

        observations: list[PoseObservation] = []
        assigned_ids = set(assignments)
        self._last_identity_states = {
            fighter_id: IdentityState.UNKNOWN for fighter_id in self.fighter_ids
        }
        for fighter_id, candidate_index in assignments.items():
            candidate = candidates[candidate_index]
            previous = self._tracks.get(fighter_id)
            center = candidate.bbox.center
            if previous is None:
                velocity = (0.0, 0.0)
                assignment_confidence = 0.92 if self.anchors else 0.90
                stable_frames = 1
            else:
                displacement = (
                    center[0] - previous.center[0],
                    center[1] - previous.center[1],
                )
                velocity = (
                    previous.velocity[0] * 0.55 + displacement[0] * 0.45,
                    previous.velocity[1] * 0.55 + displacement[1] * 0.45,
                )
                cost = self._assignment_cost(previous, candidate)
                assignment_confidence = math.exp(-0.55 * cost)
                same_source = (
                    previous.source_track_id is None
                    or candidate.source_track_id is None
                    or previous.source_track_id == candidate.source_track_id
                )
                stable_frames = previous.stable_frames + 1 if same_source else 1
            track_confidence = min(
                1.0, max(0.0, candidate.confidence * assignment_confidence)
            )
            self._tracks[fighter_id] = _TrackState(
                candidate.bbox,
                center,
                velocity,
                0,
                candidate.appearance,
                candidate.source_track_id,
                stable_frames,
            )
            match = candidate_match(candidate)
            expected_identity = self._identity_state(fighter_id)
            identity_confirmed = bool(
                gallery_ready and match.accepted and match.state == expected_identity
            )
            identity_confidence = (
                max(0.0, min(1.0, 1.0 - match.distance))
                if identity_confirmed
                else 0.0
            )
            identity_margin = match.margin if identity_confirmed else 0.0
            poor_visibility = (
                sum(point.score >= 0.25 for point in candidate.keypoints.values()) < 8
            )
            if identity_confirmed and match.reason == "gallery_match":
                self._gallery.update_adaptive(
                    expected_identity,
                    candidate.appearance,
                    confidence=identity_confidence,
                    margin=identity_margin,
                    stable_frames=stable_frames,
                    overlap_or_clinch=overlap_or_clinch or poor_visibility,
                    active_fight=active_fight,
                    parts=candidate.appearance_parts or None,
                    evidence_origin=match.reason,
                )
            identity_state = (
                expected_identity if identity_confirmed else IdentityState.UNKNOWN
            )
            review_status = (
                (
                    ReviewStatus.USER_CONFIRMED
                    if match.reason in {"user_override", "segment_user_override"}
                    else ReviewStatus.AUTO_CONFIRMED
                )
                if identity_confirmed
                else ReviewStatus.NEEDS_REVIEW
            )
            self._last_identity_states[fighter_id] = identity_state
            observations.append(
                PoseObservation(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    fighter_id=fighter_id,
                    bbox=candidate.bbox,
                    keypoints=candidate.keypoints,
                    track_confidence=track_confidence,
                    is_scene_cut=scene_cut,
                    source_track_id=candidate.source_track_id,
                    detector_bbox=candidate.bbox,
                    shot_id=self._shot_id,
                    identity_state=identity_state,
                    identity_confidence=identity_confidence,
                    identity_margin=identity_margin,
                    scene_state=(
                        SceneState.ACTIVE_FIGHT
                        if active_fight
                        else SceneState.UNCERTAIN
                    ),
                    review_status=review_status,
                    detector_confidence=candidate.detector_confidence,
                    pose_confidence=candidate.pose_confidence,
                    identity_rejection_reason=None
                    if identity_confirmed
                    else match.reason,
                )
            )

        for fighter_id in list(self._tracks):
            if fighter_id in assigned_ids:
                continue
            state = self._tracks[fighter_id]
            state.missed_frames += 1
            state.center = (
                state.center[0] + state.velocity[0],
                state.center[1] + state.velocity[1],
            )
            state.velocity = (state.velocity[0] * 0.85, state.velocity[1] * 0.85)
            if state.missed_frames > self.max_missing_frames:
                del self._tracks[fighter_id]
                self._needs_reacquisition.add(fighter_id)
                self._reacquisition_history[fighter_id].clear()

        observations.sort(
            key=lambda observation: self.fighter_ids.index(observation.fighter_id)
        )
        emitted_by_role = {
            observation.fighter_id: observation for observation in observations
        }
        self.last_diagnostics = []
        for index, candidate in enumerate(candidates):
            match = candidate_match(candidate)
            distances, negative_distance = self._gallery.distances(
                candidate.appearance, candidate.appearance_parts
            )
            selected = next(
                (
                    role
                    for role, selected_index in assignments.items()
                    if selected_index == index
                ),
                None,
            )
            emitted = emitted_by_role.get(selected)
            self.last_diagnostics.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "shot_id": self._shot_id,
                    "source_track_id": candidate.source_track_id,
                    "candidate_index": index,
                    "detector_bbox": candidate.bbox.to_dict(),
                    "detector_confidence": candidate.detector_confidence,
                    "pose_confidence": candidate.pose_confidence,
                    "a_distance": distances[IdentityState.FIGHTER_A],
                    "b_distance": distances[IdentityState.FIGHTER_B],
                    "other_distance": negative_distance,
                    "identity_state": str(emitted.identity_state)
                    if emitted is not None
                    else str(
                        IdentityState.OTHER
                        if match.state == IdentityState.OTHER
                        else IdentityState.UNKNOWN
                    ),
                    "proposed_state": str(match.state),
                    "selected_fighter_id": selected,
                    "confidence": float(emitted.identity_confidence or 0.0)
                    if emitted is not None
                    else 0.0,
                    "gallery_similarity": max(0.0, 1.0 - min(distances.values())),
                    "margin": match.margin,
                    "reason": match.reason
                    if not match.accepted or selected
                    else "temporal_unmatched",
                    "scene_state": str(
                        SceneState.ACTIVE_FIGHT
                        if active_fight
                        else SceneState.UNCERTAIN
                    ),
                }
            )
        return observations


# Friendly aliases used by integration code and notebooks.
StableTwoFighterTracker = TwoFighterTracker
PoseEstimator = RTMLibPoseBackend
