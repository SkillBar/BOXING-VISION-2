from __future__ import annotations

import os
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any
from zlib import crc32

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import HUD_MODES
from .contracts import (
    BBox,
    DisplayTrack,
    Keypoint,
    PoseObservation,
    PunchEvent,
    RenderFrameContext,
    RoundScore,
    SceneState,
)
from .identity import is_confirmed_identity
from .quality import result_eligibility

Color = tuple[int, int, int]  # OpenCV BGR

FIGHTER_COLORS: tuple[Color, ...] = (
    (67, 75, 255),  # fighter_a / red corner / RGB #FF4B43
    (255, 127, 77),  # fighter_b / blue corner / RGB #4D7FFF
    (68, 202, 137),
    (188, 95, 219),
)

FIGHTER_COLOR_BY_ID: dict[str, Color] = {
    "fighter_a": FIGHTER_COLORS[0],
    "fighter_b": FIGHTER_COLORS[1],
}

ARM_CONNECTIONS = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
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
    configured_fonts = os.environ.get("BOXING_VISION_FONT_DIR")
    supplied = (
        [str(Path(configured_fonts) / f"{prefix}{style}.{ext}")
         for style in (("Semibold", "Bold", "Regular") if bold else ("Regular",))
         for prefix in ("SF-Pro-Text-", "SFProText-") for ext in ("otf", "ttf")]
        if configured_fonts else []
    )
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = (*supplied,
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        str(windows_fonts / ("segoeuib.ttf" if bold else "segoeui.ttf")),
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                loaded = ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
            try:
                names = loaded.get_variation_names()
                desired = b"Semibold" if bold else b"Regular"
                if desired in names:
                    loaded.set_variation_by_name(desired)
            except (OSError, AttributeError):
                pass
            return loaded
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _fit_text(value: str, width: int, size: int, bold: bool = False) -> str:
    font = _font(size, bold)
    if font.getlength(value) <= width:
        return value
    clipped = value
    while clipped and font.getlength(clipped + "…") > width:
        clipped = clipped[:-1]
    return clipped.rstrip() + "…"


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
        hud_mode: str = "technical",
        tracking_overlay_style: str = "full",
    ) -> None:
        if not 0.0 <= pose_threshold <= 1.0:
            raise ValueError("pose_threshold must be between 0 and 1")
        if trail_length < 2:
            raise ValueError("trail_length must be at least 2")
        if not 0.15 <= hud_width_ratio <= 0.5:
            raise ValueError("hud_width_ratio must be between 0.15 and 0.5")
        if max_trail_gap_ms < 1:
            raise ValueError("max_trail_gap_ms must be positive")
        if hud_mode not in HUD_MODES:
            raise ValueError("hud_mode must be compact, technical or none")
        if tracking_overlay_style not in {"full", "corners"}:
            raise ValueError("tracking_overlay_style must be full or corners")

        self.fighter_names = dict(fighter_names or {})
        self.pose_threshold = pose_threshold
        self.trail_length = trail_length
        self.hud_width_ratio = hud_width_ratio
        self.max_trail_gap_ms = max_trail_gap_ms
        self.hud_mode = hud_mode
        self.tracking_overlay_style = tracking_overlay_style
        self._trails: dict[tuple[str, str], deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )
        self._trail_timestamps: dict[tuple[str, str], int] = {}
        self._fighter_order: list[str] = []
        self._leader_points: dict[str, tuple[float, float]] = {}
        self._leader_seen_ms: dict[str, int] = {}
        self._held_events: dict[str, tuple[int, PunchEvent | Mapping[str, Any]]] = {}
        self._last_cut_timestamp: int | None = None
        self._requires_confirmation = False
        self._confirmation_timestamps: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=3)
        )
        self._smoothed_boxes: dict[str, tuple[BBox, int, str | int | None]] = {}
        self._last_shot_id: int | None = None
        self._last_render_timestamp: int | None = None
        self._display_joints: dict[tuple[str, str], tuple[Keypoint, int]] = {}
        self._display_boxes: dict[str, tuple[BBox, int]] = {}

    def reset(self) -> None:
        """Clear scene-specific tracking state after a camera cut."""

        self._trails.clear()
        self._trail_timestamps.clear()
        self._leader_points.clear()
        self._leader_seen_ms.clear()
        self._held_events.clear()
        self._confirmation_timestamps.clear()
        self._smoothed_boxes.clear()
        self._display_joints.clear()
        self._display_boxes.clear()
        self._requires_confirmation = True

    def _forget_fighter(self, fighter_id: str) -> None:
        """Explicit identity rejection is not a short detector dropout."""
        for key in tuple(self._trails):
            if key[0] == fighter_id:
                self._trails.pop(key, None)
                self._trail_timestamps.pop(key, None)
        self._leader_points.pop(fighter_id, None)
        self._leader_seen_ms.pop(fighter_id, None)
        self._confirmation_timestamps.pop(fighter_id, None)
        self._smoothed_boxes.pop(fighter_id, None)
        self._held_events.pop(fighter_id, None)

    def _smooth_observation(
        self, observation: PoseObservation, timestamp_ms: int
    ) -> PoseObservation:
        bbox = observation.detector_bbox or observation.bbox
        previous = self._smoothed_boxes.get(observation.fighter_id)
        if previous is not None:
            prior, before_ms, source = previous
            delta = timestamp_ms - before_ms
            if source != observation.source_track_id or delta > 250 or delta < 0:
                self._forget_fighter(observation.fighter_id)
            elif delta > 0:
                alpha = 1.0 - 0.65 ** (delta / (1000 / 30))
                values = [
                    a + (b - a) * alpha
                    for a, b in zip(
                        (prior.x1, prior.y1, prior.x2, prior.y2),
                        (bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                    )
                ]
                bbox = BBox(*values, score=bbox.score)
        self._smoothed_boxes[observation.fighter_id] = (
            bbox,
            timestamp_ms,
            observation.source_track_id,
        )
        return replace(observation, bbox=bbox, detector_bbox=bbox)

    def _fighter_index(self, fighter_id: str) -> int:
        if fighter_id == "fighter_a":
            return 0
        if fighter_id == "fighter_b":
            return 1
        if fighter_id not in self._fighter_order:
            self._fighter_order.append(fighter_id)
        return 2 + self._fighter_order.index(fighter_id)

    def _color(self, fighter_id: str) -> Color:
        known = FIGHTER_COLOR_BY_ID.get(fighter_id)
        if known is not None:
            return known
        # Unknown/debug identities must also be stable regardless of which one
        # happens to appear in the first rendered frame.
        return FIGHTER_COLORS[2 + crc32(fighter_id.encode("utf-8")) % 2]

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
        *,
        frame_context: RenderFrameContext | None = None,
        include_hud: bool = True,
        tracking_frame_callback: Callable[[np.ndarray], None] | None = None,
        display_tracks: Sequence[DisplayTrack] | None = None,
    ) -> np.ndarray:
        """Render an export frame, optionally snapshotting clean compact tracking.

        The callback runs before timer, stat cards, event text and connectors.
        It receives its own BGR array, so a writer cannot mutate the export.
        ``include_hud=False`` is for an internal compact workspace renderer;
        it does not add or change any public HUD mode.
        """

        if tracking_frame_callback is not None and self.hud_mode != "compact":
            raise ValueError("tracking_frame_callback requires the compact renderer")

        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR numpy array")
        if frame.dtype != np.uint8:
            raise ValueError("frame dtype must be uint8")

        image = frame.copy()
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("frame cannot be empty")

        if frame_context is not None:
            timestamp_ms = frame_context.timestamp_ms
        now = int(
            timestamp_ms
            if timestamp_ms is not None
            else max(
                (
                    item.timestamp_ms
                    for item in (*observations, *(display_tracks or ()))
                ),
                default=0,
            )
        )
        if (
            self._last_render_timestamp is not None
            and now < self._last_render_timestamp
        ) or (
            frame_context is not None
            and self._last_shot_id is not None
            and frame_context.shot_id != self._last_shot_id
        ):
            self.reset()
        self._last_render_timestamp = now
        if frame_context is not None:
            self._last_shot_id = frame_context.shot_id
        if isinstance(active_events, (PunchEvent, Mapping)):
            events: list[PunchEvent | Mapping[str, Any]] = [active_events]
        else:
            events = list(active_events)
        events = [
            event
            for event in events
            if not bool(_event_value(event, "is_replay", False))
            and str(_event_value(event, "review_status", "")).lower()
            not in {"rejected", "deleted", "needs_review"}
        ]
        if timestamp_ms is not None:
            events = [
                event
                for event in events
                if int(_event_value(event, "start_ms", 0))
                <= timestamp_ms
                <= int(_event_value(event, "end_ms", 0)) + 450
            ]
        scene_active = not (
            frame_context is not None
            and str(frame_context.scene_state) != SceneState.ACTIVE_FIGHT.value
        )
        if not scene_active:
            # A short post-event hold is useful during live action, but it must
            # never leak a punch label or contact pulse into a break/replay.
            events = []
            self.reset()

        text_ops: list[tuple[tuple[int, int], str, int, Color, bool]] = []
        cut_timestamps = [
            observation.timestamp_ms
            for observation in observations
            if observation.is_scene_cut
        ]
        if frame_context is not None and frame_context.is_scene_cut:
            cut_timestamps.append(frame_context.timestamp_ms)
        if cut_timestamps:
            cut_timestamp = min(cut_timestamps)
            if cut_timestamp != self._last_cut_timestamp:
                self.reset()
                self._last_cut_timestamp = cut_timestamp

        invalid_ids = {
            item.fighter_id
            for item in observations
            if not self._confirmed_identity(item)
            or (frame_context is not None and item.shot_id != frame_context.shot_id)
        }
        for fighter_id in invalid_ids:
            self._forget_fighter(fighter_id)
        observations = [
            item
            for item in observations
            if scene_active and item.fighter_id not in invalid_ids
        ]
        if self.hud_mode == "compact":
            observations = [
                self._smooth_observation(item, now) for item in observations
            ]
        confirmed_ids = {
            item.fighter_id for item in observations if self._confirmed_identity(item)
        }
        if frame_context is not None and confirmed_ids != {"fighter_a", "fighter_b"}:
            events = []
            self._held_events.clear()
        else:
            events = [
                event
                for event in events
                if not invalid_ids.intersection(
                    {
                        str(_event_value(event, "attacker_id", "")),
                        str(_event_value(event, "defender_id", "")),
                    }
                )
            ]

        # The renderer is also a public entry point. A direct round_scores
        # argument must never bypass the shared result-eligibility decision.
        if not result_eligibility(summary or {})[0]:
            round_scores = ()
            if summary:
                summary = {
                    **summary,
                    "round_scores": [],
                    "winner": {},
                    "winner_id": None,
                    "winner_name": None,
                    "score_total": {},
                    "confidence": 0.0,
                }

        if display_tracks is None:
            for observation in observations:
                self._draw_observation(image, observation, events, summary, text_ops)
        else:
            self.draw_tracking_layer(image, display_tracks, text_ops, frame_context)

        if tracking_frame_callback is not None:
            # Prediction/uncertainty labels belong to tracking itself, unlike
            # the export-only timer, event description and statistics cards.
            tracking_frame_callback(self._apply_text(image.copy(), text_ops))

        if include_hud and self.hud_mode == "technical":
            self._draw_event_banner(image, events, text_ops)
            self._draw_technical_hud(
                image,
                observations,
                summary or {},
                round_scores,
                timestamp_ms,
                text_ops,
            )
        elif include_hud and self.hud_mode == "compact":
            self._draw_compact_hud(
                image,
                observations,
                events,
                summary or {},
                timestamp_ms,
                text_ops,
            )
            if (
                frame_context is not None
                and frame_context.scene_state != "ACTIVE_FIGHT"
            ):
                self._draw_paused_state(image, text_ops)
            elif frame_context is not None and confirmed_ids != {
                "fighter_a",
                "fighter_b",
            }:
                self._draw_paused_state(
                    image,
                    text_ops,
                    label="Личности не подтверждены · анализ приостановлен",
                )
        return self._apply_text(image, text_ops)

    def _smooth_display_track(self, track: DisplayTrack) -> DisplayTrack:
        """Independent short joint fading; never alters analytical evidence."""
        token = f"{track.shot_id}:{track.source_track_id}:{track.segment_id}"
        if str(track.display_state) in {"PREDICTED", "LOST"}:
            self._display_boxes.pop(token, None)
            for key in tuple(self._display_joints):
                if key[0] == token:
                    self._display_joints.pop(key, None)
            return replace(track, keypoints={})
        prior = self._display_boxes.get(token)
        box = track.bbox
        if prior is not None and 0 < track.timestamp_ms - prior[1] <= 250:
            alpha = 1 - 0.35 ** ((track.timestamp_ms - prior[1]) / (1000 / 30))
            box = BBox(
                *[
                    a + (b - a) * alpha
                    for a, b in zip(
                        (prior[0].x1, prior[0].y1, prior[0].x2, prior[0].y2),
                        (box.x1, box.y1, box.x2, box.y2),
                    )
                ],
                score=box.score,
            )
        self._display_boxes[token] = (box, track.timestamp_ms)
        points: dict[str, Keypoint] = {}
        names = set(track.keypoints) | {
            name for identity, name in self._display_joints if identity == token
        }
        for name in names:
            current = track.keypoints.get(name)
            previous = self._display_joints.get((token, name))
            if current is not None and current.score >= self.pose_threshold:
                point = current
                if previous is not None and 0 < track.timestamp_ms - previous[1] <= 200:
                    alpha = 1 - 0.25 ** (
                        (track.timestamp_ms - previous[1]) / (1000 / 30)
                    )
                    point = Keypoint(
                        previous[0].x + (current.x - previous[0].x) * alpha,
                        previous[0].y + (current.y - previous[0].y) * alpha,
                        current.score,
                    )
                self._display_joints[(token, name)] = (point, track.timestamp_ms)
                points[name] = point
            elif previous is not None:
                age = track.timestamp_ms - previous[1]
                if 0 <= age < 200:
                    # Retain only the last measured location with fading score;
                    # do not infer a moving glove from a missing observation.
                    points[name] = replace(
                        previous[0], score=previous[0].score * (1 - age / 200)
                    )
                else:
                    self._display_joints.pop((token, name), None)
        return replace(track, bbox=box, keypoints=points)

    @staticmethod
    def _dashed_box(
        image: np.ndarray,
        first: tuple[int, int],
        last: tuple[int, int],
        color: Color,
        thickness: int,
    ) -> None:
        x1, y1 = first
        x2, y2 = last
        dash = max(5, thickness * 4)
        for start, end in (
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ):
            vector = np.asarray(end, float) - start
            length = float(np.linalg.norm(vector))
            if length <= 0:
                continue
            for offset in range(0, round(length), dash * 2):
                p1 = np.asarray(start) + vector * (offset / length)
                p2 = np.asarray(start) + vector * (min(offset + dash, length) / length)
                cv2.line(
                    image,
                    tuple(np.rint(p1).astype(int)),
                    tuple(np.rint(p2).astype(int)),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

    def draw_tracking_layer(
        self,
        image: np.ndarray,
        tracks: Sequence[DisplayTrack],
        text_ops: list[tuple[tuple[int, int], str, int, Color, bool]],
        frame_context: RenderFrameContext | None = None,
    ) -> None:
        """The common preview/export layer; deliberately bypasses event gates."""
        if (
            frame_context is not None
            and str(frame_context.scene_state) != "ACTIVE_FIGHT"
        ):
            return
        height, width = image.shape[:2]
        thickness = max(1, round(height / 360))
        for raw_track in tracks:
            if (
                str(raw_track.scene_state) != "ACTIVE_FIGHT"
                or str(raw_track.identity_state) == "OTHER"
            ):
                continue
            if frame_context is not None and raw_track.shot_id != frame_context.shot_id:
                continue
            if str(raw_track.display_state) == "LOST":
                self._smooth_display_track(raw_track)
                continue
            predicted = str(raw_track.display_state) == "PREDICTED"
            if (
                predicted
                and not 0
                <= raw_track.timestamp_ms - raw_track.evidence_timestamp_ms
                <= 1000
            ):
                continue
            known = raw_track.fighter_id in {"fighter_a", "fighter_b"} and str(
                raw_track.identity_state
            ) == {"fighter_a": "FIGHTER_A", "fighter_b": "FIGHTER_B"}.get(
                raw_track.fighter_id
            )
            if predicted and not known:
                continue
            track = self._smooth_display_track(raw_track)
            color = self._color(track.fighter_id) if known else (194, 200, 208)
            first = _clip_point((track.bbox.x1, track.bbox.y1), width, height)
            last = _clip_point((track.bbox.x2, track.bbox.y2), width, height)
            if last[0] <= first[0] or last[1] <= first[1]:
                continue
            if predicted or not known:
                self._dashed_box(image, first, last, color, thickness)
            elif self.tracking_overlay_style == "corners":
                length = max(12, min(42, round(track.bbox.width * 0.18)))
                for x, y, dx, dy in (
                    (first[0], first[1], 1, 1),
                    (last[0], first[1], -1, 1),
                    (first[0], last[1], 1, -1),
                    (last[0], last[1], -1, -1),
                ):
                    cv2.line(
                        image,
                        (x, y),
                        (x + dx * length, y),
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
                    cv2.line(
                        image,
                        (x, y),
                        (x, y + dy * length),
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
            else:
                cv2.rectangle(image, first, last, color, thickness, cv2.LINE_AA)
            label = (
                "Прогноз"
                if predicted
                else "Личность уточняется"
                if not known
                else "A"
                if track.fighter_id == "fighter_a"
                else "B"
            )
            if predicted:
                label = f"{'A' if track.fighter_id == 'fighter_a' else 'B'} · {label}"
            size = max(12, round(height / 48))
            label_width = min(
                width - first[0], round(_font(size).getlength(label)) + 14
            )
            top = max(0, first[1] - size - 10)
            cv2.rectangle(
                image,
                (first[0], top),
                (first[0] + label_width, top + size + 8),
                (13, 16, 20),
                -1,
            )
            text_ops.append(((first[0] + 7, top + 2), label, size, color, False))
            if not predicted:
                self._draw_zones(image, track, color, allow_inferred=False)
                visual_pose = replace(
                    track,
                    fighter_id=f"display:{track.shot_id}:{track.source_track_id}:{track.segment_id}",
                )
                self._draw_arms_and_trails(
                    image, visual_pose, color, neutral_skeleton=True
                )

    @staticmethod
    def _confirmed_identity(observation: PoseObservation) -> bool:
        return is_confirmed_identity(observation)

    @staticmethod
    def _draw_paused_state(
        image: np.ndarray,
        text_ops: list[tuple[tuple[int, int], str, int, Color, bool]],
        *,
        label: str = "Анализ приостановлен",
    ) -> None:
        height, width = image.shape[:2]
        size = max(14, round(height / 38))
        panel_width = min(
            width - 24,
            max(220, round(len(label) * size * 0.56) + 28),
        )
        x1 = (width - panel_width) // 2
        y1 = max(48, round(height * 0.2))
        overlay = image.copy()
        cv2.rectangle(
            overlay, (x1, y1), (x1 + panel_width, y1 + size + 18), (7, 9, 12), -1
        )
        cv2.addWeighted(overlay, 0.86, image, 0.14, 0, image)
        cv2.rectangle(
            image, (x1, y1), (x1 + panel_width, y1 + size + 18), (59, 66, 76), 1
        )
        text_ops.append(((x1 + 14, y1 + 8), label, size, (245, 247, 250), True))

    def _draw_observation(
        self,
        image: np.ndarray,
        observation: PoseObservation,
        events: Sequence[PunchEvent | Mapping[str, Any]],
        summary: Mapping[str, Any] | None,
        text_ops: list[tuple[tuple[int, int], str, int, Color, bool]],
    ) -> None:
        if not self._confirmed_identity(observation):
            return

        height, width = image.shape[:2]
        color = self._color(observation.fighter_id)
        bbox = observation.detector_bbox or observation.bbox
        x1, y1 = _clip_point((bbox.x1, bbox.y1), width, height)
        x2, y2 = _clip_point((bbox.x2, bbox.y2), width, height)
        if x2 <= x1 or y2 <= y1:
            return

        thickness = max(1, round(min(width, height) / 420))
        if self.tracking_overlay_style == "corners":
            # Four quiet corner brackets preserve the fighter lock without
            # drawing a heavy detector rectangle across the action.
            layer = image.copy()
            bracket = max(12, min(42, round(min(x2 - x1, y2 - y1) * 0.18)))
            segments = (
                ((x1, y1 + bracket), (x1, y1), (x1 + bracket, y1)),
                ((x2 - bracket, y1), (x2, y1), (x2, y1 + bracket)),
                ((x1, y2 - bracket), (x1, y2), (x1 + bracket, y2)),
                ((x2 - bracket, y2), (x2, y2), (x2, y2 - bracket)),
            )
            for points in segments:
                cv2.polylines(
                    layer,
                    [np.asarray(points, dtype=np.int32)],
                    False,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            cv2.addWeighted(layer, 0.72, image, 0.28, 0, image)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        name = self._name(observation.fighter_id, summary)
        confidence = max(0.0, min(1.0, float(observation.identity_confidence or 0.0)))
        font_size = max(12, round(height / 48))
        if self.hud_mode != "compact":
            label = f"{name}  ID {confidence:.0%}"
            label_height = font_size + 9
            label_width = min(
                width - x1, max(86, int(len(label) * font_size * 0.58) + 12)
            )
            label_top = max(0, y1 - label_height)
            cv2.rectangle(image, (x1, label_top), (x1 + label_width, y1), color, -1)
            text_ops.append(
                ((x1 + 5, label_top + 2), label, font_size, (255, 255, 255), True)
            )

        self._draw_zones(
            image, observation, color, allow_inferred=self.hud_mode == "technical"
        )
        self._draw_arms_and_trails(
            image,
            observation,
            color,
            neutral_skeleton=self.hud_mode == "compact",
        )

        fighter_events = [
            event
            for event in events
            if str(_event_value(event, "attacker_id", "")) == observation.fighter_id
        ]
        if fighter_events and self.hud_mode == "technical":
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

    def _draw_fighter_card(
        self,
        image: np.ndarray,
        fighter_id: str,
        observation: PoseObservation | None,
        other_observation: PoseObservation | None,
        events: Sequence[PunchEvent | Mapping[str, Any]],
        summary: Mapping[str, Any],
        timestamp_ms: int,
        text_ops: list[tuple[tuple[int, int], str, int, Color, bool]],
    ) -> None:
        """Draw one fixed broadcast card and a smoothed line to its fighter."""

        height, width = image.shape[:2]
        color = self._color(fighter_id)
        name = self._name(fighter_id, summary)
        fighters = summary.get("fighters", {})
        fighter = fighters.get(fighter_id, {}) if isinstance(fighters, Mapping) else {}
        if not isinstance(fighter, Mapping):
            fighter = {}
        stats = fighter.get("stats", fighter)
        if not isinstance(stats, Mapping):
            stats = fighter
        attempts = int(_first_numeric(stats, ("attempts",), 0))
        landed = int(_first_numeric(stats, ("likely_landed", "landed"), 0))
        accuracy = _first_numeric(stats, ("accuracy", "accuracy_pct"), 0.0)
        if accuracy <= 1:
            accuracy *= 100

        fighter_events = [
            event
            for event in events
            if str(_event_value(event, "attacker_id", "")) == fighter_id
        ]
        active = max(
            fighter_events,
            key=lambda item: float(_event_value(item, "confidence", 0.0)),
            default=None,
        )
        if active is not None:
            self._held_events[fighter_id] = (timestamp_ms + 900, active)
        else:
            held = self._held_events.get(fighter_id)
            if held and timestamp_ms <= held[0]:
                active = held[1]
            elif held:
                self._held_events.pop(fighter_id, None)

        card_width = min(max(190, round(width * 0.22)), max(190, round(width * 0.34)))
        card_width = min(card_width, max(1, width // 2 - 20))
        card_height = min(max(104, round(height * 0.17)), 126)
        inset = max(10, round(min(width, height) * 0.018))
        card_left = inset if fighter_id == "fighter_a" else width - inset - card_width
        card_top = min(max(48, round(height * 0.14)), max(4, height - card_height - 8))
        card_right = card_left + card_width
        card_bottom = card_top + card_height

        if observation is not None:
            raw_anchor = self._outer_shoulder_anchor(
                observation,
                width,
                height,
                prefer_left=fighter_id == "fighter_a",
            )
            previous = self._leader_points.get(fighter_id)
            smoothed = (
                (previous[0] * 0.72 + raw_anchor[0] * 0.28)
                if previous
                else float(raw_anchor[0]),
                (previous[1] * 0.72 + raw_anchor[1] * 0.28)
                if previous
                else float(raw_anchor[1]),
            )
            self._leader_points[fighter_id] = smoothed
            self._leader_seen_ms[fighter_id] = min(
                timestamp_ms, observation.timestamp_ms
            )
            timestamps = self._confirmation_timestamps[fighter_id]
            if not timestamps or timestamps[-1] != observation.timestamp_ms:
                timestamps.append(observation.timestamp_ms)

        anchor = self._leader_points.get(fighter_id)
        age_ms = timestamp_ms - self._leader_seen_ms.get(fighter_id, -10_000)
        confirmed = (
            not self._requires_confirmation
            or len(self._confirmation_timestamps[fighter_id]) >= 3
        )
        if anchor is not None and age_ms <= 400 and confirmed:
            fade = 1.0 if age_ms <= 200 else max(0.0, 1.0 - (age_ms - 200) / 200.0)
            connector_start = (
                card_right if fighter_id == "fighter_a" else card_left,
                card_top + card_height // 2,
            )
            connector_end = _clip_point(anchor, width, height)
            elbow_x = (
                min(connector_end[0] - 8, connector_start[0] + max(12, width // 45))
                if fighter_id == "fighter_a"
                else max(
                    connector_end[0] + 8, connector_start[0] - max(12, width // 45)
                )
            )
            points = [
                connector_start,
                (elbow_x, connector_start[1]),
                (elbow_x, connector_end[1]),
                connector_end,
            ]
            if not self._observations_overlap(
                observation, other_observation
            ) and not self._polyline_intersects_observation(
                points,
                other_observation,
                width,
                height,
            ):
                line_layer = image.copy()
                cv2.polylines(
                    line_layer,
                    [np.asarray(points, dtype=np.int32)],
                    False,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(line_layer, connector_end, 3, color, -1, cv2.LINE_AA)
                cv2.addWeighted(line_layer, fade, image, 1.0 - fade, 0, image)

        overlay = image.copy()
        cv2.rectangle(
            overlay, (card_left, card_top), (card_right, card_bottom), (7, 9, 12), -1
        )
        cv2.addWeighted(overlay, 0.84, image, 0.16, 0, image)
        accent_x = card_left if fighter_id == "fighter_a" else card_right - 3
        cv2.rectangle(
            image, (accent_x, card_top), (accent_x + 3, card_bottom), color, -1
        )
        cv2.rectangle(
            image,
            (card_left, card_top),
            (card_right, card_bottom),
            (59, 66, 76),
            1,
            cv2.LINE_AA,
        )

        title_size = max(15, round(height / 35))
        small = max(12, round(height / 50))
        label_x = card_left + 12
        text_ops.append(
            (
                (label_x, card_top + 8),
                _fit_text(name, card_width - 24, title_size, True),
                title_size,
                (248, 249, 251),
                True,
            )
        )
        text_ops.append(
            (
                (label_x, card_top + title_size + 13),
                f"Попадания {landed}/{attempts} · {accuracy:.0f}%",
                small,
                (195, 202, 212),
                False,
            )
        )

        if active is not None:
            event_y = card_bottom - small * 2 - 13
            event_title = self._compact_event_title(active)
            outcome = OUTCOME_LABELS.get(
                str(_event_value(active, "outcome", "unclear")), "исход не определён"
            )
            text_ops.append(
                (
                    (label_x, event_y),
                    _fit_text(event_title, card_width - 24, small, True),
                    small,
                    (244, 246, 249),
                    True,
                )
            )
            outcome_color = (
                (69, 255, 140)
                if str(_event_value(active, "outcome", ""))
                in {"landed", "likely_landed"}
                else (195, 202, 212)
            )
            text_ops.append(
                (
                    (label_x, event_y + small + 4),
                    _fit_text(outcome, card_width - 24, small),
                    small,
                    outcome_color,
                    False,
                )
            )

    def _torso_anchor(
        self, observation: PoseObservation, width: int, height: int
    ) -> tuple[int, int]:
        points = [
            _visible_point(observation, name, self.pose_threshold, width, height)
            for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        ]
        visible = [point for point in points if point is not None]
        if visible:
            return (
                round(sum(point[0] for point in visible) / len(visible)),
                round(sum(point[1] for point in visible) / len(visible)),
            )
        bbox = observation.bbox
        return _clip_point(
            ((bbox.x1 + bbox.x2) / 2.0, bbox.y1 + bbox.height * 0.42),
            width,
            height,
        )

    def _outer_shoulder_anchor(
        self,
        observation: PoseObservation,
        width: int,
        height: int,
        *,
        prefer_left: bool,
    ) -> tuple[int, int]:
        shoulders = [
            point
            for point in (
                _visible_point(
                    observation,
                    "left_shoulder",
                    self.pose_threshold,
                    width,
                    height,
                ),
                _visible_point(
                    observation,
                    "right_shoulder",
                    self.pose_threshold,
                    width,
                    height,
                ),
            )
            if point is not None
        ]
        if shoulders:
            return (
                min(shoulders, key=lambda point: point[0])
                if prefer_left
                else max(shoulders, key=lambda point: point[0])
            )
        return self._torso_anchor(observation, width, height)

    @staticmethod
    def _observations_overlap(
        first: PoseObservation | None, second: PoseObservation | None
    ) -> bool:
        if first is None or second is None:
            return False
        a, b = first.detector_bbox or first.bbox, second.detector_bbox or second.bbox
        intersection = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
            0.0, min(a.y2, b.y2) - max(a.y1, b.y1)
        )
        return intersection / max(1.0, a.area + b.area - intersection) >= 0.35

    @staticmethod
    def _polyline_intersects_observation(
        points: Sequence[tuple[int, int]],
        observation: PoseObservation | None,
        width: int,
        height: int,
    ) -> bool:
        if observation is None:
            return False
        bbox = observation.detector_bbox or observation.bbox
        x1, y1 = _clip_point((bbox.x1, bbox.y1), width, height)
        x2, y2 = _clip_point((bbox.x2, bbox.y2), width, height)
        if x2 <= x1 or y2 <= y1:
            return False
        rect = (x1, y1, x2 - x1, y2 - y1)
        return any(
            cv2.clipLine(rect, points[index - 1], points[index])[0]
            for index in range(1, len(points))
        )

    @staticmethod
    def _compact_event_title(event: PunchEvent | Mapping[str, Any]) -> str:
        hand = str(_event_value(event, "hand", "unknown"))
        technique_key = str(_event_value(event, "technique", "unknown"))
        target_key = str(_event_value(event, "target", "unknown"))
        target = (
            "в голову"
            if target_key == "head"
            else "в корпус"
            if target_key == "body"
            else ""
        )
        if technique_key == "unknown":
            side = "левой" if hand == "left" else "правой" if hand == "right" else ""
            return f"Удар {side} {target}".strip()
        side = "Левый" if hand == "left" else "Правый" if hand == "right" else ""
        technique = TECHNIQUE_LABELS.get(technique_key, technique_key)
        return f"{side} {technique} {target}".strip()

    def _draw_arms_and_trails(
        self,
        image: np.ndarray,
        observation: PoseObservation,
        color: Color,
        *,
        neutral_skeleton: bool = False,
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
        thickness = (
            max(1, min(2, round(min(width, height) / 420)))
            if neutral_skeleton
            else max(1, round(min(width, height) / 280))
        )
        bone_color = (232, 235, 239) if neutral_skeleton else color

        def visibility(name: str) -> float:
            value = observation.keypoints.get(name)
            if value is None:
                return 0.0
            score = _keypoint_value(value)[2]
            return float(
                np.clip(
                    (score - self.pose_threshold)
                    / max(0.01, 0.85 - self.pose_threshold),
                    0,
                    1,
                )
            )

        for start_name, end_name in ARM_CONNECTIONS:
            start = points[start_name]
            end = points[end_name]
            if start is not None and end is not None:
                layer = image.copy()
                cv2.line(layer, start, end, bone_color, thickness, cv2.LINE_AA)
                if neutral_skeleton:
                    alpha = 0.72 * min(visibility(start_name), visibility(end_name))
                    cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, image)
                else:
                    image[:] = layer

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
                layer = image.copy()
                cv2.circle(
                    layer, point, thickness + 2, (255, 255, 255), -1, cv2.LINE_AA
                )
                cv2.circle(layer, point, thickness + 3, color, 1, cv2.LINE_AA)
                alpha = 0.85 * visibility(wrist_name) if neutral_skeleton else 1.0
                cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, image)

            trail = self._trails[(observation.fighter_id, wrist_name)]
            trail_age = observation.timestamp_ms - self._trail_timestamps.get(
                (observation.fighter_id, wrist_name), -10_000
            )
            trail_alpha = (
                max(0.0, 1.0 - trail_age / 200.0)
                if point is None
                else visibility(wrist_name)
            )
            if trail_age > 200:
                trail.clear()
            for index in range(1, len(trail)):
                ratio = index / max(1, len(trail) - 1)
                faded = tuple(int(channel * (0.25 + 0.75 * ratio)) for channel in color)
                trail_thickness = max(1, round(thickness * ratio))
                layer = image.copy()
                cv2.line(
                    layer,
                    trail[index - 1],
                    trail[index],
                    faded,
                    trail_thickness,
                    cv2.LINE_AA,
                )
                alpha = 0.72 * trail_alpha if neutral_skeleton else trail_alpha
                cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, image)

    def _draw_zones(
        self,
        image: np.ndarray,
        observation: PoseObservation | DisplayTrack,
        color: Color,
        *,
        allow_inferred: bool = True,
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
        elif allow_inferred:
            center = _clip_point(
                ((bbox.x1 + bbox.x2) / 2.0, bbox.y1 + bbox.height * 0.15),
                width,
                height,
            )
        if head_points or allow_inferred:
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
        elif allow_inferred:
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
            " · Возможный повтор"
            if bool(_event_value(event, "is_replay", False))
            else ""
        )
        evidence = _event_value(event, "evidence", {})
        possible_fall = (
            " · Возможное падение"
            if isinstance(evidence, Mapping)
            and bool(evidence.get("possible_knockdown"))
            else ""
        )
        label = (
            f"{technique} → {target} · {outcome} · уверенность {confidence:.0%} · "
            f"интенсивность {impact}{replay}{possible_fall}"
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

    def _draw_compact_hud(
        self,
        image: np.ndarray,
        observations: Sequence[PoseObservation],
        events: Sequence[PunchEvent | Mapping[str, Any]],
        summary: Mapping[str, Any],
        timestamp_ms: int | None,
        text_ops: list[tuple[tuple[int, int], str, int, Color, bool]],
    ) -> None:
        """Draw fixed A/B cards, leader lines and the round clock."""

        height, width = image.shape[:2]
        medium = max(12, round(height / 46))
        observation_by_id = {
            observation.fighter_id: observation
            for observation in observations
            if self._confirmed_identity(observation)
        }
        fighters = summary.get("fighters", {})
        fighter_ids = {
            fighter_id
            for fighter_id in ("fighter_a", "fighter_b")
            if fighter_id in observation_by_id
            or (isinstance(fighters, Mapping) and fighter_id in fighters)
        }
        now = max(0, int(timestamp_ms or 0))
        for fighter_id in ("fighter_a", "fighter_b"):
            if fighter_id in fighter_ids:
                self._draw_fighter_card(
                    image,
                    fighter_id,
                    observation_by_id.get(fighter_id),
                    observation_by_id.get(
                        "fighter_b" if fighter_id == "fighter_a" else "fighter_a"
                    ),
                    events,
                    summary,
                    now,
                    text_ops,
                )

        self._draw_selected_contact(
            image,
            observation_by_id,
            events,
            now,
        )

        clock = self._compact_clock(summary, timestamp_ms)
        panel_height = medium + 14
        panel_width = min(width - 16, max(122, round(len(clock) * medium * 0.63) + 26))
        x1, y1 = max(8, (width - panel_width) // 2), 6
        x2 = min(width - 8, x1 + panel_width)
        y2 = min(height - 1, y1 + panel_height)
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (8, 10, 13), -1)
        cv2.addWeighted(overlay, 0.84, image, 0.16, 0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (52, 58, 66), 1, cv2.LINE_AA)
        text_ops.append(((x1 + 13, y1 + 5), clock, medium, (241, 243, 247), True))

    def _draw_selected_contact(
        self,
        image: np.ndarray,
        observations: Mapping[str, PoseObservation],
        events: Sequence[PunchEvent | Mapping[str, Any]],
        timestamp_ms: int,
    ) -> None:
        """Draw the selected-reference concentric impact pulse conservatively."""

        eligible = [
            event
            for event in events
            if str(_event_value(event, "outcome", ""))
            in {"landed", "likely_landed", "blocked", "block"}
            and int(_event_value(event, "peak_ms", timestamp_ms))
            <= timestamp_ms
            <= int(_event_value(event, "peak_ms", timestamp_ms)) + 900
        ]
        if not eligible:
            return
        event = max(
            eligible,
            key=lambda item: float(_event_value(item, "confidence", 0.0)),
        )
        defender_id = str(_event_value(event, "defender_id", ""))
        observation = observations.get(defender_id)
        if observation is None:
            return
        height, width = image.shape[:2]
        target = str(_event_value(event, "target", "unknown"))
        if target == "head":
            names = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        elif target == "body":
            names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        else:
            return
        points = [
            point
            for point in (
                _visible_point(
                    observation,
                    name,
                    self.pose_threshold,
                    width,
                    height,
                )
                for name in names
            )
            if point is not None
        ]
        if not points:
            return
        center = (
            round(sum(point[0] for point in points) / len(points)),
            round(sum(point[1] for point in points) / len(points)),
        )
        peak_ms = int(_event_value(event, "peak_ms", timestamp_ms))
        phase = min(1.0, max(0.0, timestamp_ms - peak_ms) / 900.0)
        base = max(8, round(min(width, height) * 0.012))
        accent = self._color(str(_event_value(event, "attacker_id", "fighter_a")))
        outcome = str(_event_value(event, "outcome", ""))
        core = (
            (72, 255, 152) if outcome in {"landed", "likely_landed"} else (93, 193, 244)
        )
        layer = image.copy()
        evidence = _event_value(event, "evidence", {})
        evidence = evidence if isinstance(evidence, Mapping) else {}
        # Canonical silhouette coordinates are not camera image coordinates.
        contact = evidence.get("contact_point_image_norm")
        geometry_confidence = evidence.get("contact_point_image_confidence", 0.0)
        precise = (
            isinstance(contact, Mapping)
            and all(
                isinstance(contact.get(axis), (float, int))
                and np.isfinite(contact[axis])
                and 0 <= contact[axis] <= 1
                for axis in ("x", "y")
            )
            and isinstance(geometry_confidence, (float, int))
            and geometry_confidence >= 0.8
        )
        if not precise:
            # A target class permits a zone outline, never a fabricated contact dot.
            if target == "body" and len(points) >= 3:
                hull = cv2.convexHull(np.asarray(points, dtype=np.int32))
                cv2.polylines(layer, [hull], True, accent, 2, cv2.LINE_AA)
            elif target == "head":
                cv2.ellipse(
                    layer,
                    center,
                    (
                        max(8, round(observation.bbox.width * 0.15)),
                        max(10, round(observation.bbox.height * 0.11)),
                    ),
                    0,
                    0,
                    360,
                    accent,
                    2,
                    cv2.LINE_AA,
                )
            alpha = 0.6 * (1.0 - phase)
            cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0, image)
            return
        center = _clip_point(
            (float(contact["x"]) * width, float(contact["y"]) * height), width, height
        )
        cv2.circle(
            layer, center, base + round(base * phase * 1.8), accent, 2, cv2.LINE_AA
        )
        cv2.circle(layer, center, base + round(base * 0.55), accent, 1, cv2.LINE_AA)
        cv2.circle(layer, center, max(3, base // 3), core, -1, cv2.LINE_AA)
        cv2.circle(layer, center, max(5, base // 2), (245, 247, 250), 1, cv2.LINE_AA)
        alpha = 0.92 - 0.42 * phase
        cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0, image)

    @staticmethod
    def _compact_clock(summary: Mapping[str, Any], timestamp_ms: int | None) -> str:
        timestamp_ms = max(0, int(timestamp_ms or 0))
        metadata = summary.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        round_length_s = max(1, int(metadata.get("round_length_s", 180) or 180))
        rest_length_s = max(0, int(metadata.get("rest_length_s", 60) or 60))
        scheduled_rounds = max(1, int(metadata.get("scheduled_rounds", 1) or 1))
        fight_start_s = max(0.0, float(metadata.get("fight_start_s", 0) or 0))
        elapsed_s = max(0.0, timestamp_ms / 1000.0 - fight_start_s)
        cycle_s = max(1, round_length_s + rest_length_s)
        round_number = min(scheduled_rounds, int(elapsed_s // cycle_s) + 1)
        phase_s = elapsed_s % cycle_s
        if timestamp_ms / 1000.0 < fight_start_s:
            remaining_s = fight_start_s - timestamp_ms / 1000.0
            phase = "До начала"
        elif phase_s < round_length_s:
            remaining_s = max(0.0, round_length_s - phase_s)
            phase = f"Раунд {round_number}"
        else:
            remaining_s = max(0.0, cycle_s - phase_s)
            phase = "Перерыв"
        minutes, seconds = divmod(remaining_s, 60.0)
        return f"{phase} · {int(minutes):01d}:{seconds:04.1f}"

    def _draw_technical_hud(
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

        small = max(12, round(height / 55))
        medium = max(12, round(height / 45))
        title = max(12, round(height / 36))
        cursor_y = 10
        text_ops.append(
            ((x + 10, cursor_y), "Boxing Vision", title, (245, 245, 245), True)
        )
        cursor_y += title + 6
        text_ops.append(
            ((x + 10, cursor_y), "Аналитика модели", small, (74, 204, 255), True)
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
        fighter_ids: list[str] = [
            fighter_id
            for fighter_id in ("fighter_a", "fighter_b")
            if fighter_id in fighters
            or any(observation.fighter_id == fighter_id for observation in observations)
        ]
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
                f"Интенсивность: {impact:.0f}/100",
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
    frame_context: RenderFrameContext | None = None,
    display_tracks: Sequence[DisplayTrack] | None = None,
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
        frame_context=frame_context,
        display_tracks=display_tracks,
    )


render_annotated_frame = render_frame


__all__ = [
    "FIGHTER_COLORS",
    "FIGHTER_COLOR_BY_ID",
    "FrameRenderer",
    "render_annotated_frame",
    "render_frame",
]
