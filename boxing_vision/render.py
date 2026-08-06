from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contracts import Keypoint, PoseObservation, PunchEvent, RoundScore

Color = tuple[int, int, int]  # OpenCV BGR

FIGHTER_COLORS: tuple[Color, ...] = (
    (52, 70, 235),  # red corner
    (235, 142, 48),  # blue corner
    (68, 202, 137),
    (188, 95, 219),
)

ARM_CONNECTIONS = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
)

TECHNIQUE_LABELS = {
    "jab": "джеб",
    "cross": "кросс",
    "straight": "прямой",
    "hook": "хук",
    "uppercut": "апперкот",
    "unknown": "тип неясен",
}

OUTCOME_LABELS = {
    "landed": "вероятное попадание",
    "likely_landed": "вероятное попадание",
    "blocked": "вероятный блок",
    "missed": "вероятный промах",
    "miss": "вероятный промах",
    "unclear": "результат неясен",
    "unknown": "результат неясен",
}

TARGET_LABELS = {
    "head": "голова",
    "body": "корпус",
    "unknown": "цель неясна",
}


def _clip_point(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x = round(np.clip(point[0], 0, max(0, width - 1)))
    y = round(np.clip(point[1], 0, max(0, height - 1)))
    return x, y


def _keypoint_value(value: Keypoint | Mapping[str, Any]) -> tuple[float, float, float]:
    if isinstance(value, Mapping):
        return float(value["x"]), float(value["y"]), float(value.get("score", 1.0))
    return float(value.x), float(value.y), float(value.score)


def _visible_point(
    observation: PoseObservation,
    name: str,
    threshold: float,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    value = observation.keypoints.get(name)
    if value is None:
        return None
    x, y, score = _keypoint_value(value)
    if score < threshold or not np.isfinite((x, y, score)).all():
        return None
    return _clip_point((x, y), width, height)


def _event_value(
    event: PunchEvent | Mapping[str, Any], name: str, default: Any = None
) -> Any:
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def _score_value(
    score: RoundScore | Mapping[str, Any], name: str, default: Any = None
) -> Any:
    if isinstance(score, Mapping):
        return score.get(name, default)
    return getattr(score, name, default)


def _first_numeric(
    mapping: Mapping[str, Any], names: Sequence[str], default: float = 0.0
) -> float:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return default


@lru_cache(maxsize=32)
def _font(
    size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


class FrameRenderer:
    """Stateful frame renderer with short wrist histories for motion trails."""

    def __init__(
        self,
        fighter_names: Mapping[str, str] | None = None,
        *,
        pose_threshold: float = 0.25,
        trail_length: int = 12,
        hud_width_ratio: float = 0.30,
        max_trail_gap_ms: int = 500,
    ) -> None:
        if not 0.0 <= pose_threshold <= 1.0:
            raise ValueError("pose_threshold must be between 0 and 1")
        if trail_length < 2:
            raise ValueError("trail_length must be at least 2")
        if not 0.15 <= hud_width_ratio <= 0.5:
            raise ValueError("hud_width_ratio must be between 0.15 and 0.5")
        if max_trail_gap_ms < 1:
            raise ValueError("max_trail_gap_ms must be positive")

        self.fighter_names = dict(fighter_names or {})
        self.pose_threshold = pose_threshold
        self.trail_length = trail_length
        self.hud_width_ratio = hud_width_ratio
        self.max_trail_gap_ms = max_trail_gap_ms
        self._trails: dict[tuple[str, str], deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )
        self._trail_timestamps: dict[tuple[str, str], int] = {}
        self._fighter_order: list[str] = []

    def reset(self) -> None:
        """Clear scene-specific tracking state after a camera cut."""

        self._trails.clear()
        self._trail_timestamps.clear()

    def _fighter_index(self, fighter_id: str) -> int:
        if fighter_id not in self._fighter_order:
            self._fighter_order.append(fighter_id)
        return self._fighter_order.index(fighter_id)

    def _color(self, fighter_id: str) -> Color:
        return FIGHTER_COLORS[self._fighter_index(fighter_id) % len(FIGHTER_COLORS)]

    def _name(self, fighter_id: str, summary: Mapping[str, Any] | None = None) -> str:
        if fighter_id in self.fighter_names:
            return self.fighter_names[fighter_id]
        if summary:
            fighters = summary.get("fighters", {})
            if isinstance(fighters, Mapping):
                stats = fighters.get(fighter_id, {})
                if isinstance(stats, Mapping) and stats.get("name"):
                    return str(stats["name"])
        return fighter_id

    def draw(
        self,
        frame: np.ndarray,
        observations: Sequence[PoseObservation],
        active_events: Sequence[PunchEvent | Mapping[str, Any]]
        | PunchEvent
        | Mapping[str, Any] = (),
        summary: Mapping[str, Any] | None = None,
        round_scores: Sequence[RoundScore | Mapping[str, Any]] | None = None,
        timestamp_ms: int | None = None,
    ) -> np.ndarray:
        """Return a rendered BGR frame without modifying the caller's array."""

        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR numpy array")
        if frame.dtype != np.uint8:
            raise ValueError("frame dtype must be uint8")

        image = frame.copy()
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("frame cannot be empty")

        if isinstance(active_events, (PunchEvent, Mapping)):
            events: list[PunchEvent | Mapping[str, Any]] = [active_events]
        else:
            events = list(active_events)
        if timestamp_ms is not None:
            events = [
                event
                for event in events
                if int(_event_value(event, "start_ms", 0)) - 150
                <= timestamp_ms
                <= int(_event_value(event, "end_ms", 0)) + 450
            ]

        text_ops: list[tuple[tuple[int, int], str, int, Color, bool]] = []
        if any(observation.is_scene_cut for observation in observations):
            self.reset()

        for observation in observations:
            self._draw_observation(image, observation, events, summary, text_ops)

        self._draw_event_banner(image, events, text_ops)
        self._draw_hud(
            image,
            observations,
            summary or {},
            round_scores,
            timestamp_ms,
            text_ops,
        )
        return self._apply_text(image, text_ops)

    def _draw_observation(
        self,
        image: np.ndarray,
        observation: PoseObservation,
        events: Sequence[PunchEvent | Mapping[str, Any]],
        summary: Mapping[str, Any] | None,
        text_ops: list[tuple[tuple[int, int], str, int, Color, bool]],
    ) -> None:
        height, width = image.shape[:2]
        color = self._color(observation.fighter_id)
        bbox = observation.bbox
        x1, y1 = _clip_point((bbox.x1, bbox.y1), width, height)
        x2, y2 = _clip_point((bbox.x2, bbox.y2), width, height)
        if x2 <= x1 or y2 <= y1:
            return

        thickness = max(1, round(min(width, height) / 280))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        name = self._name(observation.fighter_id, summary)
        confidence = max(0.0, min(1.0, observation.track_confidence))
        label = f"{name}  ID {confidence:.0%}"
        font_size = max(11, round(height / 48))
        label_height = font_size + 9
        label_width = min(width - x1, max(86, int(len(label) * font_size * 0.58) + 12))
        label_top = max(0, y1 - label_height)
        cv2.rectangle(image, (x1, label_top), (x1 + label_width, y1), color, -1)
        text_ops.append(
            ((x1 + 5, label_top + 2), label, font_size, (255, 255, 255), True)
        )

        self._draw_zones(image, observation, color)
        self._draw_arms_and_trails(image, observation, color)

        fighter_events = [
            event
            for event in events
            if str(_event_value(event, "attacker_id", "")) == observation.fighter_id
        ]
        if fighter_events:
            event = max(
                fighter_events,
                key=lambda item: float(_event_value(item, "confidence", 0.0)),
            )
            confidence = float(_event_value(event, "confidence", 0.0))
            technique = TECHNIQUE_LABELS.get(
                str(_event_value(event, "technique", "unknown")),
                str(_event_value(event, "technique", "unknown")),
            )
            event_label = f"{technique} · {confidence:.0%}"
            y = min(height - font_size - 4, y2 + 5)
            cv2.rectangle(
                image,
                (x1, y),
                (
                    min(
                        width - 1,
                        x1 + max(110, int(len(event_label) * font_size * 0.6)),
                    ),
                    y + font_size + 8,
                ),
                (24, 24, 24),
                -1,
            )
            text_ops.append(((x1 + 5, y + 2), event_label, font_size, color, True))

    def _draw_arms_and_trails(
        self,
        image: np.ndarray,
        observation: PoseObservation,
        color: Color,
    ) -> None:
        height, width = image.shape[:2]
        points = {
            name: _visible_point(
                observation,
                name,
                self.pose_threshold,
                width,
                height,
            )
            for connection in ARM_CONNECTIONS
            for name in connection
        }
        thickness = max(2, round(min(width, height) / 200))
        for start_name, end_name in ARM_CONNECTIONS:
            start = points[start_name]
            end = points[end_name]
            if start is not None and end is not None:
                cv2.line(image, start, end, color, thickness, cv2.LINE_AA)

        for wrist_name in ("left_wrist", "right_wrist"):
            point = points[wrist_name]
            if point is not None:
                trail_key = (observation.fighter_id, wrist_name)
                previous_timestamp = self._trail_timestamps.get(trail_key)
                if previous_timestamp is not None:
                    gap_ms = observation.timestamp_ms - previous_timestamp
                    if gap_ms <= 0 or gap_ms > self.max_trail_gap_ms:
                        self._trails[trail_key].clear()
                self._trails[trail_key].append(point)
                self._trail_timestamps[trail_key] = observation.timestamp_ms
                cv2.circle(
                    image, point, thickness + 2, (255, 255, 255), -1, cv2.LINE_AA
                )
                cv2.circle(image, point, thickness + 3, color, 1, cv2.LINE_AA)

            trail = self._trails[(observation.fighter_id, wrist_name)]
            for index in range(1, len(trail)):
                ratio = index / max(1, len(trail) - 1)
                faded = tuple(int(channel * (0.25 + 0.75 * ratio)) for channel in color)
                trail_thickness = max(1, round(thickness * ratio))
                cv2.line(
                    image,
                    trail[index - 1],
                    trail[index],
                    faded,
                    trail_thickness,
                    cv2.LINE_AA,
                )

    def _draw_zones(
        self,
        image: np.ndarray,
        observation: PoseObservation,
        color: Color,
    ) -> None:
        height, width = image.shape[:2]
        bbox = observation.bbox
        overlay = image.copy()

        head_candidates = [
            _visible_point(observation, name, self.pose_threshold, width, height)
            for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        ]
        head_points = [point for point in head_candidates if point is not None]
        if head_points:
            center = (
                round(sum(point[0] for point in head_points) / len(head_points)),
                round(sum(point[1] for point in head_points) / len(head_points)),
            )
        else:
            center = _clip_point(
                ((bbox.x1 + bbox.x2) / 2.0, bbox.y1 + bbox.height * 0.15),
                width,
                height,
            )
        radius_x = max(5, round(max(1.0, bbox.width) * 0.13))
        radius_y = max(6, round(max(1.0, bbox.height) * 0.10))
        cv2.ellipse(
            overlay, center, (radius_x, radius_y), 0, 0, 360, color, -1, cv2.LINE_AA
        )
        cv2.ellipse(
            image, center, (radius_x, radius_y), 0, 0, 360, color, 1, cv2.LINE_AA
        )

        shoulders = [
            _visible_point(observation, name, self.pose_threshold, width, height)
            for name in ("left_shoulder", "right_shoulder")
        ]
        hips = [
            _visible_point(observation, name, self.pose_threshold, width, height)
            for name in ("left_hip", "right_hip")
        ]
        torso_points = [point for point in (*shoulders, *hips) if point is not None]
        if len(torso_points) >= 3:
            hull = cv2.convexHull(np.asarray(torso_points, dtype=np.int32))
            cv2.fillConvexPoly(overlay, hull, color, cv2.LINE_AA)
            cv2.polylines(image, [hull], True, color, 1, cv2.LINE_AA)
        else:
            body_x1, body_y1 = _clip_point(
                (bbox.x1 + bbox.width * 0.2, bbox.y1 + bbox.height * 0.23),
                width,
                height,
            )
            body_x2, body_y2 = _clip_point(
                (bbox.x2 - bbox.width * 0.2, bbox.y1 + bbox.height * 0.60),
                width,
                height,
            )
            cv2.rectangle(overlay, (body_x1, body_y1), (body_x2, body_y2), color, -1)
            cv2.rectangle(image, (body_x1, body_y1), (body_x2, body_y2), color, 1)

        cv2.addWeighted(overlay, 0.13, image, 0.87, 0, image)

    def _draw_event_banner(
        self,
        image: np.ndarray,
        events: Sequence[PunchEvent | Mapping[str, Any]],
        text_ops: list[tuple[tuple[int, int], str, int, Color, bool]],
    ) -> None:
        if not events:
            return
        height, width = image.shape[:2]
        event = max(
            events, key=lambda item: float(_event_value(item, "confidence", 0.0))
        )
        technique = TECHNIQUE_LABELS.get(
            str(_event_value(event, "technique", "unknown")),
            str(_event_value(event, "technique", "unknown")),
        )
        target = TARGET_LABELS.get(
            str(_event_value(event, "target", "unknown")),
            str(_event_value(event, "target", "unknown")),
        )
        outcome = OUTCOME_LABELS.get(
            str(_event_value(event, "outcome", "unclear")),
            str(_event_value(event, "outcome", "unclear")),
        )
        confidence = float(_event_value(event, "confidence", 0.0))
        impact = int(_event_value(event, "impact_proxy_0_100", 0))
        replay = (
            " · ВОЗМОЖНЫЙ ПОВТОР"
            if bool(_event_value(event, "is_replay", False))
            else ""
        )
        evidence = _event_value(event, "evidence", {})
        possible_fall = (
            " · ВОЗМОЖНОЕ ПАДЕНИЕ"
            if isinstance(evidence, Mapping) and bool(evidence.get("possible_knockdown"))
            else ""
        )
        label = (
            f"{technique} → {target} · {outcome} · уверенность {confidence:.0%} · "
            f"impact {impact}{replay}{possible_fall}"
        )

        font_size = max(12, round(height / 42))
        banner_width = min(max(180, round(width * 0.69)), max(180, width - 8))
        banner_height = font_size + 14
        x1 = max(4, (width - banner_width) // 2)
        x2 = min(width - 4, x1 + banner_width)
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, 6), (x2, 6 + banner_height), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
        color = self._color(str(_event_value(event, "attacker_id", "fighter")))
        cv2.rectangle(image, (x1, 6), (x1 + 5, 6 + banner_height), color, -1)
        text_ops.append(((x1 + 11, 10), label, font_size, (245, 245, 245), True))

    def _draw_hud(
        self,
        image: np.ndarray,
        observations: Sequence[PoseObservation],
        summary: Mapping[str, Any],
        round_scores: Sequence[RoundScore | Mapping[str, Any]] | None,
        timestamp_ms: int | None,
        text_ops: list[tuple[tuple[int, int], str, int, Color, bool]],
    ) -> None:
        height, width = image.shape[:2]
        panel_width = max(130, min(370, round(width * self.hud_width_ratio)))
        panel_width = min(width, panel_width)
        x = width - panel_width
        overlay = image.copy()
        cv2.rectangle(overlay, (x, 0), (width, height), (10, 13, 19), -1)
        cv2.addWeighted(overlay, 0.76, image, 0.24, 0, image)
        cv2.line(image, (x, 0), (x, height), (68, 75, 88), 1, cv2.LINE_AA)

        small = max(10, round(height / 55))
        medium = max(11, round(height / 45))
        title = max(12, round(height / 36))
        cursor_y = 10
        text_ops.append(
            ((x + 10, cursor_y), "BOXING VISION", title, (245, 245, 245), True)
        )
        cursor_y += title + 6
        text_ops.append(
            ((x + 10, cursor_y), "НЕОФИЦИАЛЬНАЯ AI-ОЦЕНКА", small, (74, 204, 255), True)
        )
        cursor_y += small + 10
        if timestamp_ms is not None:
            minutes, remainder = divmod(max(0, timestamp_ms), 60_000)
            seconds = remainder / 1000.0
            text_ops.append(
                (
                    (x + 10, cursor_y),
                    f"Время {minutes:02d}:{seconds:04.1f}",
                    medium,
                    (220, 224, 230),
                    False,
                )
            )
            cursor_y += medium + 8

        fighters = summary.get("fighters", {})
        if not isinstance(fighters, Mapping):
            fighters = {}
        fighter_ids: list[str] = []
        for observation in observations:
            if observation.fighter_id not in fighter_ids:
                fighter_ids.append(observation.fighter_id)
        for fighter_id in fighters:
            fighter_id = str(fighter_id)
            if fighter_id not in fighter_ids:
                fighter_ids.append(fighter_id)
        for fighter_id in self._fighter_order:
            if fighter_id not in fighter_ids:
                fighter_ids.append(fighter_id)

        available_height = max(50, height - cursor_y - 75)
        block_height = max(66, available_height // max(1, min(2, len(fighter_ids))))
        for fighter_id in fighter_ids[:2]:
            if cursor_y + 55 >= height:
                break
            color = self._color(fighter_id)
            stats = fighters.get(fighter_id, {})
            if not isinstance(stats, Mapping):
                stats = {}
            name = self._name(fighter_id, summary)
            cv2.rectangle(
                image, (x + 8, cursor_y), (x + 12, cursor_y + medium + 2), color, -1
            )
            text_ops.append(((x + 18, cursor_y - 1), name, medium, color, True))
            row_y = cursor_y + medium + 6

            attempts = _first_numeric(
                stats, ("attempts", "thrown", "punches", "candidates")
            )
            landed = _first_numeric(stats, ("likely_landed", "landed"))
            blocked = _first_numeric(stats, ("blocked", "likely_blocked"))
            missed = _first_numeric(stats, ("missed", "miss", "likely_missed"))
            unclear = _first_numeric(stats, ("unclear", "unknown"))
            accuracy = _first_numeric(stats, ("accuracy", "accuracy_pct"), -1.0)
            if accuracy < 0:
                accuracy = landed / attempts * 100.0 if attempts else 0.0
            elif accuracy <= 1.0:
                accuracy *= 100.0
            impact = _first_numeric(
                stats,
                (
                    "average_impact_proxy",
                    "impact_proxy_avg",
                    "average_impact",
                    "impact_proxy_0_100",
                    "impact",
                ),
            )

            rows = (
                f"Кандидаты: {attempts:.0f}  Попад.: {landed:.0f}",
                f"Блок: {blocked:.0f}  Промах: {missed:.0f}",
                f"Неясно: {unclear:.0f}  Точность: {accuracy:.0f}%",
                f"Impact proxy: {impact:.0f}/100",
            )
            for row in rows:
                if row_y + small >= height:
                    break
                text_ops.append(((x + 12, row_y), row, small, (222, 225, 230), False))
                row_y += small + 4
            cursor_y += block_height

        score_values: Sequence[RoundScore | Mapping[str, Any]] | None = round_scores
        if score_values is None:
            candidate = summary.get("round_scores")
            if isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes)
            ):
                score_values = candidate
        if score_values and cursor_y + 35 < height:
            score = score_values[-1]
            round_number = int(_score_value(score, "round", 1))
            points_a = int(_score_value(score, "fighter_a_points", 10))
            points_b = int(_score_value(score, "fighter_b_points", 10))
            confidence = float(_score_value(score, "confidence", 0.0))
            cursor_y = max(cursor_y, height - medium * 3 - 24)
            cv2.line(
                image,
                (x + 10, cursor_y - 7),
                (width - 10, cursor_y - 7),
                (68, 75, 88),
                1,
            )
            text_ops.append(
                (
                    (x + 10, cursor_y),
                    f"Раунд {round_number}:  {points_a} — {points_b}",
                    medium,
                    (248, 248, 248),
                    True,
                )
            )
            text_ops.append(
                (
                    (x + 10, cursor_y + medium + 5),
                    f"Уверенность счёта: {confidence:.0%}",
                    small,
                    (74, 204, 255),
                    False,
                )
            )

    @staticmethod
    def _apply_text(
        image: np.ndarray,
        operations: Sequence[tuple[tuple[int, int], str, int, Color, bool]],
    ) -> np.ndarray:
        if not operations:
            return image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas)
        for (x, y), value, size, color, bold in operations:
            rgb_color = (color[2], color[1], color[0])
            draw.text((x, y), value, font=_font(size, bold), fill=rgb_color)
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def render_frame(
    frame: np.ndarray,
    observations: Sequence[PoseObservation],
    active_events: Sequence[PunchEvent | Mapping[str, Any]]
    | PunchEvent
    | Mapping[str, Any] = (),
    summary: Mapping[str, Any] | None = None,
    round_scores: Sequence[RoundScore | Mapping[str, Any]] | None = None,
    timestamp_ms: int | None = None,
    *,
    renderer: FrameRenderer | None = None,
) -> np.ndarray:
    """Functional wrapper; pass a renderer to retain wrist trails across frames."""

    target = renderer or FrameRenderer()
    return target.draw(
        frame,
        observations,
        active_events,
        summary,
        round_scores,
        timestamp_ms,
    )


render_annotated_frame = render_frame


__all__ = ["FIGHTER_COLORS", "FrameRenderer", "render_annotated_frame", "render_frame"]
