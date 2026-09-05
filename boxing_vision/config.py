from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

HUD_MODES = frozenset({"compact", "technical", "none"})
ENROLLMENT_MODES = frozenset({"auto_confirm", "manual", "legacy_anchor"})
REGION_MODES = frozenset({"auto", "none", "manual"})
TIMING_MODES = frozenset({"scheduled", "continuous"})


def validate_working_region(points: object) -> None:
    """Validate an explicitly selected, normalized four-corner working area."""
    try:
        if len(points) != 4 or any(len(point) != 2 for point in points):
            raise ValueError
        polygon = [tuple(float(value) for value in point) for point in points]
    except (TypeError, ValueError) as exc:
        raise ValueError("Рабочая область задаётся четырьмя точками") from exc
    if not all(math.isfinite(value) and 0 <= value <= 1 for point in polygon for value in point):
        raise ValueError("Точки рабочей области должны находиться внутри кадра")
    turns = []
    for index in range(4):
        a, b, c = (polygon[(index + offset) % 4] for offset in range(3))
        turns.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(turn > 1e-9 for turn in turns) or all(turn < -1e-9 for turn in turns)):
        raise ValueError("Точки рабочей области должны образовывать выпуклый четырёхугольник без пересечений")


@dataclass(slots=True)
class AnalysisConfig:
    """User-visible fight setup and conservative runtime defaults."""

    fighter_a_name: str = "Красный угол"
    fighter_b_name: str = "Синий угол"
    fighter_a_record: str | None = None
    fighter_b_record: str | None = None
    fighter_a_portrait_path: str | Path | None = None
    fighter_b_portrait_path: str | Path | None = None
    fighter_a_stance: str = "unknown"
    fighter_b_stance: str = "unknown"
    fighter_a_anchor: tuple[float, float] | None = None
    fighter_b_anchor: tuple[float, float] | None = None
    enrollment_mode: str = "auto_confirm"
    enrollment_frames: tuple[float, ...] = ()
    enrollment_samples: tuple[dict[str, object], ...] = ()
    enrollment_confirmed: bool = False
    region_mode: str = "auto"
    ring_rois: tuple[tuple[float, float], ...] = ()
    identity_overrides: dict[str, str] = field(default_factory=dict)
    segment_identity_overrides: dict[str, str] = field(default_factory=dict)
    scene_overrides: dict[str, str] = field(default_factory=dict)
    model_bundle_id: str = "cooperative-v3"
    punch_model: str = "auto"
    detector_score_threshold: float = 0.1
    detector_nms_threshold: float = 0.6
    identity_gallery_distance_max: float = 0.35
    identity_margin_min: float = 0.12
    adaptive_identity_confidence_min: float = 0.90
    adaptive_identity_margin_min: float = 0.20
    scheduled_rounds: int = 12
    timing_mode: str = "scheduled"
    round_length_s: int = 180
    rest_length_s: int = 60
    fight_start_s: float = 0.0
    fight_end_s: float | None = None
    analysis_fps: float = 15.0
    output_fps: int = 30
    output_height: int = 720
    max_duration_s: int = 3600
    confidence_threshold: float = 0.55
    pose_score_threshold: float = 0.25
    detector_frequency: int = 3
    backend: str = "auto"
    hud_mode: str = "compact"
    tracking_overlay_style: str = "full"
    display_prediction_ms: int = 1000
    keep_debug: bool = False
    confirmed_knockdowns_a_rounds: tuple[int, ...] = ()
    confirmed_knockdowns_b_rounds: tuple[int, ...] = ()

    @property
    def effective_region_mode(self) -> str:
        """Old cache/CLI configs infer their existing polygon, never discard it."""
        return ("manual" if self.ring_rois else "none") if self.region_mode == "auto" else self.region_mode

    @property
    def effective_ring_rois(self) -> tuple[tuple[float, float], ...]:
        return self.ring_rois if self.effective_region_mode == "manual" else ()

    def validate(self) -> None:
        if self.region_mode not in REGION_MODES:
            raise ValueError("Неизвестный режим рабочей области")
        if self.timing_mode not in TIMING_MODES:
            raise ValueError("Режим времени должен быть continuous или scheduled")
        if self.tracking_overlay_style not in {"full", "corners"}:
            raise ValueError("Стиль трекинга должен быть full или corners")
        if isinstance(self.display_prediction_ms, bool) or not isinstance(self.display_prediction_ms, int) or not 0 <= self.display_prediction_ms <= 1000:
            raise ValueError("Прогноз визуального трекинга должен быть от 0 до 1000 мс")
        if self.effective_region_mode == "manual":
            validate_working_region(self.ring_rois)
        if not isinstance(self.segment_identity_overrides, dict) or any(
            not isinstance(key, str) or not key or str(value) not in {"FIGHTER_A", "FIGHTER_B", "OTHER", "UNKNOWN"}
            for key, value in self.segment_identity_overrides.items()
        ):
            raise ValueError("Некорректное исправление сегмента личности")
        if self.hud_mode not in HUD_MODES:
            raise ValueError("HUD должен быть compact, technical или none")
        if self.enrollment_mode not in ENROLLMENT_MODES:
            raise ValueError(
                "Режим калибровки должен быть auto_confirm, manual или legacy_anchor"
            )
        if not 1 <= self.scheduled_rounds <= 24:
            raise ValueError("Количество раундов должно быть от 1 до 24")
        if not 60 <= self.round_length_s <= 300:
            raise ValueError("Длина раунда должна быть от 60 до 300 секунд")
        if not 0 <= self.rest_length_s <= 180:
            raise ValueError("Перерыв между раундами должен быть от 0 до 180 секунд")
        if self.fight_start_s < 0:
            raise ValueError("Начало боя не может быть отрицательным")
        if self.fight_end_s is not None and self.fight_end_s <= self.fight_start_s:
            raise ValueError("Конец боя должен быть позже начала")
        if self.analysis_fps <= 0 or self.output_fps <= 0:
            raise ValueError("FPS должен быть положительным")
        anchors = (self.fighter_a_anchor, self.fighter_b_anchor)
        if (anchors[0] is None) != (anchors[1] is None):
            raise ValueError("Для подтверждения нужны координаты обоих бойцов")
        if anchors[0] is not None and anchors[1] is not None:
            for anchor in anchors:
                assert anchor is not None
                if len(anchor) != 2 or not all(0.0 <= float(value) <= 1.0 for value in anchor):
                    raise ValueError("Координаты бойцов должны быть нормализованы от 0 до 1")
            distance = (
                (anchors[0][0] - anchors[1][0]) ** 2
                + (anchors[0][1] - anchors[1][1]) ** 2
            ) ** 0.5
            if distance < 0.03:
                raise ValueError("Выберите двух разных бойцов на кадре")
        if self.enrollment_frames and len(self.enrollment_frames) != 3:
            raise ValueError("Для калибровки нужны ровно три кадра")
        if self.enrollment_samples:
            if len(self.enrollment_samples) != 3:
                raise ValueError("Подтвердите три калибровочных кадра")
            for sample in self.enrollment_samples:
                if not isinstance(sample, dict) or float(sample.get("time_s", -1)) < 0:
                    raise ValueError("Некорректное время калибровки")
                for role in ("fighter_a", "fighter_b"):
                    box = sample.get(role)
                    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(0 <= float(v) <= 1 for v in box):
                        raise ValueError("Выберите реальные рамки обоих бойцов на каждом кадре")
                if sample["fighter_a"] == sample["fighter_b"]:
                    raise ValueError("Один человек не может быть обоими бойцами")
        if self.punch_model not in {"auto", "baseline", "acm40960"}:
            raise ValueError("Неизвестный классификатор ударов")
        if not 0 < self.detector_score_threshold < 1 or not 0 < self.detector_nms_threshold < 1:
            raise ValueError("Пороги детектора должны быть между 0 и 1")
        if any(float(value) < 0 for value in self.enrollment_frames):
            raise ValueError("Время калибровочного кадра не может быть отрицательным")
        if self.ring_rois and len(self.ring_rois) != 4:
            raise ValueError("Внутренняя зона ринга задаётся четырьмя точками")
        for point in self.ring_rois:
            if len(point) != 2 or not all(0.0 <= float(value) <= 1.0 for value in point):
                raise ValueError("Точки ринга должны быть нормализованы от 0 до 1")
        for value, label in (
            (self.identity_gallery_distance_max, "gallery distance"),
            (self.identity_margin_min, "identity margin"),
            (self.adaptive_identity_confidence_min, "adaptive confidence"),
            (self.adaptive_identity_margin_min, "adaptive margin"),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{label} должен быть от 0 до 1")
        scene_states = {
            "ACTIVE_FIGHT",
            "BREAK",
            "REPLAY",
            "NON_FIGHT",
            "UNCERTAIN",
        }
        if any(str(value).upper() not in scene_states for value in self.scene_overrides.values()):
            raise ValueError("scene_overrides содержит неизвестное состояние сцены")
        for round_number in (
            *self.confirmed_knockdowns_a_rounds,
            *self.confirmed_knockdowns_b_rounds,
        ):
            if not 1 <= int(round_number) <= self.scheduled_rounds:
                raise ValueError("Раунд подтверждённого нокдауна вне расписания боя")

    def to_dict(self) -> dict[str, object]:
        """Return the persistable configuration without local upload paths.

        Portrait paths are runtime-only inputs.  The processed, run-relative
        filenames are written to ``summary.json`` by the pipeline instead, so
        serialising a config can never disclose the user's original path.
        """

        payload = asdict(self)
        for key in ("fighter_a_portrait_path", "fighter_b_portrait_path"):
            payload.pop(key, None)
        return payload


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = Path(os.environ["BOXING_VISION_DATA_DIR"]).expanduser().resolve() / "runs" if os.environ.get("BOXING_VISION_DATA_DIR") else PROJECT_ROOT / "runs"
