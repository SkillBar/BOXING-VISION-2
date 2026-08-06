from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class AnalysisConfig:
    """User-visible fight setup and conservative runtime defaults."""

    fighter_a_name: str = "Красный угол"
    fighter_b_name: str = "Синий угол"
    fighter_a_stance: str = "unknown"
    fighter_b_stance: str = "unknown"
    fighter_a_anchor: tuple[float, float] | None = None
    fighter_b_anchor: tuple[float, float] | None = None
    scheduled_rounds: int = 12
    round_length_s: int = 180
    rest_length_s: int = 60
    fight_start_s: float = 0.0
    fight_end_s: float | None = None
    analysis_fps: float = 10.0
    output_fps: int = 30
    output_height: int = 720
    max_duration_s: int = 3600
    confidence_threshold: float = 0.55
    pose_score_threshold: float = 0.25
    detector_frequency: int = 3
    backend: str = "auto"
    keep_debug: bool = False
    confirmed_knockdowns_a_rounds: tuple[int, ...] = ()
    confirmed_knockdowns_b_rounds: tuple[int, ...] = ()

    def validate(self) -> None:
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
        for round_number in (
            *self.confirmed_knockdowns_a_rounds,
            *self.confirmed_knockdowns_b_rounds,
        ):
            if not 1 <= int(round_number) <= self.scheduled_rounds:
                raise ValueError("Раунд подтверждённого нокдауна вне расписания боя")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
