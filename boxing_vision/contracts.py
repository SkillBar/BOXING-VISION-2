from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class Keypoint:
    x: float
    y: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class PoseObservation:
    frame_index: int
    timestamp_ms: int
    fighter_id: str
    bbox: BBox
    keypoints: dict[str, Keypoint]
    track_confidence: float = 1.0
    is_scene_cut: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(slots=True)
class PunchEvent:
    event_id: str
    round: int
    start_ms: int
    peak_ms: int
    end_ms: int
    attacker_id: str
    defender_id: str
    hand: str
    technique: str
    target: str
    outcome: str
    confidence: float
    impact_proxy_0_100: int
    is_replay: bool = False
    review_status: str = "unreviewed"
    clip_path: str | None = None
    evidence: dict[str, float] = field(default_factory=dict)
    is_counter: bool = False
    combo_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RoundScore:
    round: int
    fighter_a_points: int
    fighter_b_points: int
    fighter_a_index: float
    fighter_b_index: float
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisResult:
    job_id: str
    run_dir: Path
    annotated_video: Path
    events_path: Path
    summary_path: Path
    log_path: Path
    clips_dir: Path
    events: list[PunchEvent]
    summary: dict[str, Any]

