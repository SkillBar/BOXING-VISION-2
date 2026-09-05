"""Three-view enrollment proposals. Proposals never equal user confirmation."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from .contracts import BBox
from .pose import RawPose, create_pose_backend


@lru_cache(maxsize=1)
def calibration_backend():
    return create_pose_backend("auto", strict=True, mode="lightweight", device="cpu",
                               detector_score_threshold=0.1, detector_nms_threshold=0.6)


def box_iou(a: BBox, b: BBox) -> float:
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return intersection / max(1, a.area + b.area - intersection)


def normalized_box(box: BBox, width: int, height: int) -> list[float]:
    return [round(float(np.clip(v / size, 0, 1)), 6) for v, size in
            zip((box.x1, box.y1, box.x2, box.y2), (width, height, width, height))]


def enrollment_support_keypoints(poses: list[RawPose], width: int, height: int) -> list[dict[str, dict[str, float]]]:
    """Keep existing ankle evidence for ROI checks; do not run a model again."""
    supports = []
    for pose in poses:
        points = {}
        for name in ("left_ankle", "right_ankle"):
            point = pose.keypoints.get(name)
            if point is not None and np.isfinite((point.x, point.y, point.score)).all():
                # Do not clamp off-image predictions into apparently visible
                # feet at the image border.
                points[name] = {"x": float(point.x) / width, "y": float(point.y) / height,
                                "score": float(point.score)}
        supports.append(points)
    return supports


def validate_ring_floor_roi(state: dict[str, Any]) -> dict[str, Any]:
    """Reject a rope-strip ROI only when both visible supports contradict it.

    The ROI belongs to the first calibration view only. Ankles are imperfect
    estimates of the floor contact, so use a body-scale tolerance and accept
    either foot inside/on/near the polygon. Missing, clipped or low-confidence
    ankles do not become bbox-bottom pseudo-measurements.
    """
    views = state.get("views", [])
    points = state.get("ring_points", [])
    if not views or len(points) != 4:
        return {"status": "insufficient_evidence", "checked_roles": [], "outside_roles": []}
    view = views[0]
    h, w = np.asarray(view["image"]).shape[:2]
    polygon = np.asarray([(float(x) * w, float(y) * h) for x, y in points], np.float32)
    if not np.isfinite(polygon).all():
        raise ValueError("Область ринга содержит некорректные точки; отметьте четыре точки пола заново")
    supports = view.get("support_keypoints", [])
    checked, outside = [], []
    for role in ("fighter_a", "fighter_b"):
        index = view.get("selection", {}).get(role)
        if not isinstance(index, int) or index < 0 or index >= len(supports) or index >= len(view["boxes"]):
            continue
        x1, y1, x2, y2 = (float(value) for value in view["boxes"][index])
        box_height = y2 - y1
        if box_height <= 0 or x2 <= x1:
            continue
        # Covers pose uncertainty and ankle-to-sole offset, and scales equally
        # for a 4K source, a small preview and a fighter at the far ring edge.
        tolerance = box_height * .06
        border_margin = box_height * .015
        visible = []
        evidence = supports[index]
        if not isinstance(evidence, dict):
            continue
        for name in ("left_ankle", "right_ankle"):
            point = evidence.get(name)
            if not isinstance(point, dict):
                continue
            try:
                x, y, score = float(point["x"]) * w, float(point["y"]) * h, float(point["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite((x, y, score)).all() or not .65 <= score <= 1:
                continue
            if not (border_margin < x < w - border_margin and border_margin < y < h - border_margin):
                continue
            if not (x1 - tolerance <= x <= x2 + tolerance and y1 - tolerance <= y <= y2 + tolerance):
                continue
            visible.append((x, y))
        if not visible:
            continue
        checked.append(role)
        if all(cv2.pointPolygonTest(polygon, point, True) < -tolerance for point in visible):
            outside.append(role)
    if len(outside) == 2:
        raise ValueError(
            "Область ринга не включает видимые стопы обоих бойцов. Похоже, отмечена полоса канатов. "
            "На первом кадре выберите «Область ринга», очистите точки и отметьте по периметру пол "
            "внутри ринга, включая места, где стоят оба бойца. Если стопы перекрыты, выберите другой кадр."
        )
    return {"status": "checked" if len(checked) == 2 else "insufficient_evidence",
            "checked_roles": checked, "outside_roles": outside}


def match_enrollment_box(poses: list[RawPose], normalized: list[float], width: int, height: int) -> RawPose:
    reference = BBox(*(v * s for v, s in zip(normalized, (width, height, width, height))))
    candidate = max(poses, key=lambda pose: box_iou(reference, pose.bbox), default=None)
    if candidate is None or box_iou(reference, candidate.bbox) < 0.5:
        raise ValueError("Подтверждённый боксёр не найден на калибровочном кадре; выберите другой кадр")
    return candidate


def propose_roles(frame: np.ndarray, poses: list[RawPose]) -> dict[str, int | None]:
    """Equipment colors propose roles only. Never sort people left/right."""
    h, w = frame.shape[:2]
    scored = []
    for index, pose in enumerate(poses):
        b = pose.bbox
        # Weak detections support existing motion tracks, never initial roles.
        detector_score = pose.detector_confidence if pose.detector_confidence is not None else b.score
        if detector_score < .65 or b.area < w * h * .02:
            continue
        crop = frame[max(0, int(b.y1 + b.height * .2)):min(h, int(b.y1 + b.height * .72)),
                     max(0, int(b.x1 + b.width * .25)):min(w, int(b.x2 - b.width * .25))]
        if not crop.size:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturated = (hsv[..., 1] > 95) & (hsv[..., 2] > 45)
        red = float(np.mean(saturated & ((hsv[..., 0] < 12) | (hsv[..., 0] > 170))))
        blue = float(np.mean(saturated & (hsv[..., 0] > 95) & (hsv[..., 0] < 132)))
        scored.append((index, red, blue))
    a = max(scored, key=lambda item: item[1] - item[2], default=None)
    b = max(scored, key=lambda item: item[2] - item[1], default=None)
    return {
        "fighter_a": a[0] if a and a[1] > .08 and a[1] - a[2] > .05 else None,
        "fighter_b": b[0] if b and b[2] > .08 and b[2] - b[1] > .05 and (not a or b[0] != a[0]) else None,
    }


def render_enrollment_view(state: dict[str, Any], index: int) -> tuple[np.ndarray, str]:
    sample = state["views"][index]
    image = np.asarray(sample["image"]).copy()
    selected = sample["selection"]
    for candidate_id, box in enumerate(sample["boxes"]):
        role = next((role for role in ("fighter_a", "fighter_b") if selected.get(role) == candidate_id), None)
        color = (255, 81, 74) if role == "fighter_a" else (86, 130, 255) if role == "fighter_b" else (145, 154, 166)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = "A" if role == "fighter_a" else "B" if role else str(candidate_id + 1)
        cv2.putText(image, label, (x1 + 4, max(24, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2, cv2.LINE_AA)
    points = state.get("ring_points", [])
    if index == 0 and points:
        h, w = image.shape[:2]
        pixels = np.asarray([(round(x*w), round(y*h)) for x, y in points], np.int32)
        for x, y in pixels:
            cv2.circle(image, (int(x), int(y)), 5, (244, 193, 93), -1)
        if len(pixels) > 1:
            cv2.polylines(image, [pixels], len(pixels) == 4, (244, 193, 93), 2)
    ready = sum(all(view["selection"].get(role) is not None for role in ("fighter_a", "fighter_b")) for view in state["views"])
    message = f"Кадр {index + 1}/3 · {sample['time_s']:.2f} с · заполнено {ready}/3. Проверьте A и B; клик по человеку выбирает его настоящую рамку."
    if state.get("confirmed"):
        message = "Три кадра подтверждены. Перед полным анализом будет выполнена десятисекундная проверка."
    return image, message


def confirm_enrollment(state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(state.get("views", [])) != 3:
        raise ValueError("Сначала предложите три калибровочных кадра")
    samples = []
    for view in state["views"]:
        a, b = (view["selection"].get(role) for role in ("fighter_a", "fighter_b"))
        if a is None or b is None or a == b:
            raise ValueError("На каждом из трёх кадров выберите двух разных бойцов")
        h, w = np.asarray(view["image"]).shape[:2]
        samples.append({"time_s": float(view["time_s"]) - float(state.get("start_s", 0)),
                        "fighter_a": normalized_box(BBox(*view["boxes"][a]), w, h),
                        "fighter_b": normalized_box(BBox(*view["boxes"][b]), w, h)})
    validate_ring_floor_roi(state)
    state = dict(state, confirmed=True, enrollment_samples=samples)
    return state, samples
