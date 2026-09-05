from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
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


class IdentityState(StrEnum):
    """Global identity assigned after shot-local motion tracking."""

    FIGHTER_A = "FIGHTER_A"
    FIGHTER_B = "FIGHTER_B"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class SceneState(StrEnum):
    """Broadcast state used to gate tracking updates and punch events."""

    ACTIVE_FIGHT = "ACTIVE_FIGHT"
    BREAK = "BREAK"
    REPLAY = "REPLAY"
    NON_FIGHT = "NON_FIGHT"
    UNCERTAIN = "UNCERTAIN"


class ReviewStatus(StrEnum):
    AUTO_CONFIRMED = "AUTO_CONFIRMED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    USER_CONFIRMED = "USER_CONFIRMED"
    REJECTED = "REJECTED"


class DisplayState(StrEnum):
    OBSERVED = "OBSERVED"
    INTERPOLATED = "INTERPOLATED"
    PREDICTED = "PREDICTED"
    LOST = "LOST"


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
    source_track_id: str | int | None = None
    detector_bbox: BBox | None = None
    shot_id: int = 0
    identity_state: IdentityState | str | None = None
    identity_confidence: float | None = None
    identity_margin: float | None = None
    scene_state: SceneState | str = SceneState.ACTIVE_FIGHT
    review_status: ReviewStatus | str = ReviewStatus.AUTO_CONFIRMED
    detector_confidence: float | None = None
    pose_confidence: float | None = None
    identity_rejection_reason: str | None = None
    segment_id: str | None = None
    physical_track_id: str | None = None
    identity_origin: str | None = None

    def __post_init__(self) -> None:
        # Old render caches predate explicit global identity.  Infer only the
        # two canonical roles; every other label remains safely UNKNOWN.
        if self.identity_state is None:
            self.identity_state = {
                "fighter_a": IdentityState.FIGHTER_A,
                "fighter_b": IdentityState.FIGHTER_B,
            }.get(self.fighter_id, IdentityState.UNKNOWN)
        if self.identity_confidence is None:
            self.identity_confidence = float(self.track_confidence)
        if self.detector_bbox is None:
            self.detector_bbox = self.bbox

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(slots=True)
class DisplayTrack:
    """Presentation evidence, deliberately not a PoseObservation or punch input.

    Unlike legacy analytical observations, an absent identity is never inferred
    from a label. Predictions must remain distinguishable from measurements.
    All geometry is in normalized-video pixels, not mannequin coordinates.
    """

    timestamp_ms: int
    evidence_timestamp_ms: int
    bbox: BBox
    keypoints: dict[str, Keypoint] = field(default_factory=dict)
    shot_id: int = 0
    source_track_id: str | int | None = None
    segment_id: str | None = None
    physical_track_id: str | None = None
    fighter_id: str | None = None
    identity_state: IdentityState | str = IdentityState.UNKNOWN
    identity_origin: str | None = None
    display_state: DisplayState | str = DisplayState.OBSERVED
    detector_confidence: float | None = None
    identity_confidence: float | None = None
    identity_margin: float | None = None
    scene_state: SceneState | str = SceneState.ACTIVE_FIGHT
    is_scene_cut: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RawPunchProposal:
    """Measured motion retained for review; not a scored PunchEvent."""

    proposal_id: str
    shot_id: int
    source_track_id: str | int
    segment_id: str
    start_ms: int
    peak_ms: int
    end_ms: int
    hand: str
    confidence: float
    physical_track_id: str | None = None
    fighter_id: str | None = None
    status: str = "needs_review"
    reason: str = "identity_unresolved"
    resolved_event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RenderFrameContext:
    """Scene-level render state kept separate from fighter observations."""

    timestamp_ms: int
    shot_id: int = 0
    is_scene_cut: bool = False
    scene_state: str = "ACTIVE_FIGHT"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    evidence: dict[str, Any] = field(default_factory=dict)
    is_counter: bool = False
    combo_id: str | None = None
    proposal_confidence: float | None = None
    classification_confidence: float | None = None
    outcome_confidence: float | None = None
    target_point_norm: dict[str, float] | None = None
    target_point_confidence: float | None = None
    target_uncertainty_radius: float | None = None
    target_point_space: str = "unspecified"
    target_point_source: str | None = None
    model_version: str | None = None

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
