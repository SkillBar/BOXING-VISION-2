from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

import cv2

from .artifacts import (
    JobArtifacts,
    atomic_write_json,
    create_job_artifacts,
    extract_event_clips,
    finalize_h264_video,
    persist_fighter_portrait,
)
from .config import DEFAULT_RUNS_DIR, AnalysisConfig
from .contracts import (
    AnalysisResult,
    BBox,
    DisplayTrack,
    IdentityState,
    Keypoint,
    PoseObservation,
    PunchEvent,
    RenderFrameContext,
    ReviewStatus,
    SceneState,
)
from .events import detect_punch_events
from .identity import is_confirmed_identity
from .pose import TwoFighterTracker, create_pose_backend
from .previews import generate_hover_previews
from .quality import apply_result_gate, result_eligibility
from .render import FrameRenderer
from .scoring import build_fight_summary, score_rounds
from .video import (
    ReplayDetector,
    SceneCutDetector,
    VideoValidationError,
    iter_video_frames,
    normalize_video,
    probe_video,
    validate_video,
)

ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]
_P = ParamSpec("_P")
_R = TypeVar("_R")
_ACTIVE_ARTIFACTS: ContextVar[JobArtifacts | None] = ContextVar(
    "boxing_vision_active_artifacts",
    default=None,
)
_RENDER_CACHE_DIRNAME = ".render_cache"
_RENDER_CACHE_VIDEO = "normalized.mp4"
_RENDER_CACHE_OBSERVATIONS = "observations.jsonl.gz"
_RENDER_CACHE_CONFIG = "config.json"
_RENDER_CACHE_VERSION = 3
_WORKSPACE_PREVIEW_FILENAME = "workspace-preview.mp4"
_BODY_MAP_ASSET = (
    Path(__file__).with_name("static")
    / "assets"
    / "body-map-v3-base.png"
)


class AnalysisCancelledError(RuntimeError):
    """Raised when the local user cancels the only active analysis job."""


class RenderCacheUnavailableError(RuntimeError):
    """Raised when a run predates, or has lost, its render-only cache."""


def _check_cancel(callback: CancelCallback | None) -> None:
    if callback is not None and callback():
        raise AnalysisCancelledError("Обработка отменена пользователем")


def _cleanup_failed_job(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Remove large working media on failure while retaining the audit log."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        token = _ACTIVE_ARTIFACTS.set(None)
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            artifacts = _ACTIVE_ARTIFACTS.get()
            if artifacts is not None:
                try:
                    with artifacts.log_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            f"{datetime.now(UTC).isoformat()} failed={type(exc).__name__}: {exc}\n"
                        )
                except OSError:
                    pass
                shutil.rmtree(artifacts.work_dir, ignore_errors=True)
                artifacts.annotated_video.unlink(missing_ok=True)
                (artifacts.run_dir / _WORKSPACE_PREVIEW_FILENAME).unlink(missing_ok=True)
                artifacts.events_path.unlink(missing_ok=True)
                artifacts.summary_path.unlink(missing_ok=True)
                (artifacts.run_dir / "boxing-vision-result.zip").unlink(missing_ok=True)
                shutil.rmtree(artifacts.clips_dir, ignore_errors=True)
                shutil.rmtree(artifacts.run_dir / "profiles", ignore_errors=True)
                shutil.rmtree(artifacts.run_dir / "previews", ignore_errors=True)
                (artifacts.run_dir / "preview_manifest.json").unlink(missing_ok=True)
                shutil.rmtree(artifacts.run_dir / "assets", ignore_errors=True)
                for filename in (
                    "body_map_manifest.json",
                    "model_manifest.json",
                    "tracklets.jsonl",
                    "tracking_diagnostics.jsonl",
                    "scenes.json",
                    "identity_profile.json",
                    "review.json",
                ):
                    (artifacts.run_dir / filename).unlink(missing_ok=True)
                # Failed calibration is diagnostic evidence, not a finished run.
                # Retain its bounded 10s detection cache to explain the rejection.
                if type(exc).__name__ != "EnrollmentRequiredError":
                    shutil.rmtree(artifacts.run_dir / _RENDER_CACHE_DIRNAME, ignore_errors=True)
                for temporary in artifacts.run_dir.rglob("*.part*"):
                    temporary.unlink(missing_ok=True)
                for temporary in artifacts.run_dir.rglob("*.tmp.*"):
                    temporary.unlink(missing_ok=True)
            raise
        finally:
            _ACTIVE_ARTIFACTS.reset(token)

    return wrapped


def _confirmed_knockdowns(
    config: AnalysisConfig,
) -> dict[int, dict[str, int]]:
    confirmed: dict[int, dict[str, int]] = {}
    for fighter_id, rounds in (
        ("fighter_a", config.confirmed_knockdowns_a_rounds),
        ("fighter_b", config.confirmed_knockdowns_b_rounds),
    ):
        for round_number in rounds:
            round_counts = confirmed.setdefault(int(round_number), {})
            round_counts[fighter_id] = round_counts.get(fighter_id, 0) + 1
    return confirmed


def _emit(callback: ProgressCallback | None, fraction: float, description: str) -> None:
    if callback is None:
        return
    callback(min(1.0, max(0.0, float(fraction))), description)


def _important_events(
    events: Iterable[PunchEvent], limit: int = 24
) -> list[PunchEvent]:
    candidates = [event for event in events if not event.is_replay]
    candidates.sort(
        key=lambda event: (
            event.outcome == "likely_landed",
            event.review_status == "confirmed",
            event.confidence,
            event.impact_proxy_0_100,
        ),
        reverse=True,
    )
    return sorted(candidates[:limit], key=lambda event: event.peak_ms)


def _renderable_events(events: Iterable[PunchEvent]) -> list[PunchEvent]:
    """Exclude manually rejected candidates from regenerated overlays."""

    return [
        event
        for event in events
        if event.review_status.lower() not in {"rejected", "deleted"}
    ]


def _interpolate_pose_observation(
    before: PoseObservation,
    after: PoseObservation,
    timestamp_ms: int,
) -> PoseObservation:
    """Interpolate one confirmed fighter track without crossing a scene cut."""

    duration = max(1, after.timestamp_ms - before.timestamp_ms)
    ratio = min(1.0, max(0.0, (timestamp_ms - before.timestamp_ms) / duration))

    def blend(left: float, right: float) -> float:
        return left + (right - left) * ratio

    keys = before.keypoints.keys() & after.keypoints.keys()
    keypoints = {
        name: Keypoint(
            blend(before.keypoints[name].x, after.keypoints[name].x),
            blend(before.keypoints[name].y, after.keypoints[name].y),
            blend(before.keypoints[name].score, after.keypoints[name].score),
        )
        for name in keys
    }
    before_detector = before.detector_bbox or before.bbox
    after_detector = after.detector_bbox or after.bbox
    same_identity = before.identity_state == after.identity_state
    return PoseObservation(
        frame_index=before.frame_index,
        timestamp_ms=timestamp_ms,
        fighter_id=before.fighter_id,
        bbox=BBox(
            blend(before.bbox.x1, after.bbox.x1),
            blend(before.bbox.y1, after.bbox.y1),
            blend(before.bbox.x2, after.bbox.x2),
            blend(before.bbox.y2, after.bbox.y2),
            blend(before.bbox.score, after.bbox.score),
        ),
        keypoints=keypoints,
        track_confidence=blend(before.track_confidence, after.track_confidence),
        is_scene_cut=before.is_scene_cut and timestamp_ms == before.timestamp_ms,
        source_track_id=(
            before.source_track_id
            if before.source_track_id == after.source_track_id
            else None
        ),
        detector_bbox=BBox(
            blend(before_detector.x1, after_detector.x1),
            blend(before_detector.y1, after_detector.y1),
            blend(before_detector.x2, after_detector.x2),
            blend(before_detector.y2, after_detector.y2),
            blend(before_detector.score, after_detector.score),
        ),
        shot_id=before.shot_id,
        segment_id=before.segment_id if before.segment_id == after.segment_id else None,
        physical_track_id=before.physical_track_id if before.physical_track_id == after.physical_track_id else None,
        identity_origin=before.identity_origin,
        identity_state=(
            before.identity_state if same_identity else IdentityState.UNKNOWN
        ),
        identity_confidence=blend(
            float(before.identity_confidence or 0.0),
            float(after.identity_confidence or 0.0),
        ),
        identity_margin=min(
            float(before.identity_margin or 0.0),
            float(after.identity_margin or 0.0),
        ),
        scene_state=before.scene_state,
        review_status=(
            before.review_status
            if before.review_status == after.review_status
            else ReviewStatus.NEEDS_REVIEW
        ),
    )


def _sample_pose_tracks(
    tracks: dict[str, list[PoseObservation]],
    indices: dict[str, int],
    timestamp_ms: int,
    *,
    max_interpolation_gap_ms: int = 250,
    max_hold_ms: int = 200,
) -> list[PoseObservation]:
    """Sample stable A/B observations at output FPS for the render pass."""

    sampled: list[PoseObservation] = []
    for fighter_id in ("fighter_a", "fighter_b"):
        track = tracks.get(fighter_id, [])
        if not track:
            continue
        index = indices.get(fighter_id, 0)
        while index + 1 < len(track) and track[index + 1].timestamp_ms <= timestamp_ms:
            index += 1
        indices[fighter_id] = index
        before = track[index] if track[index].timestamp_ms <= timestamp_ms else None
        after = track[index + 1] if index + 1 < len(track) else None
        if (
            before is not None
            and after is not None
            and after.timestamp_ms - before.timestamp_ms <= max_interpolation_gap_ms
            and not after.is_scene_cut
            and before.shot_id == after.shot_id
            and before.source_track_id == after.source_track_id
            and before.segment_id == after.segment_id
            and before.identity_state == after.identity_state
            and before.scene_state == after.scene_state
        ):
            sampled.append(_interpolate_pose_observation(before, after, timestamp_ms))
        elif before is not None and timestamp_ms - before.timestamp_ms <= max_hold_ms:
            sampled.append(
                PoseObservation(
                    frame_index=before.frame_index,
                    timestamp_ms=before.timestamp_ms,
                    fighter_id=before.fighter_id,
                    bbox=before.bbox,
                    keypoints=before.keypoints,
                    track_confidence=before.track_confidence,
                    is_scene_cut=before.is_scene_cut
                    and timestamp_ms == before.timestamp_ms,
                    source_track_id=before.source_track_id,
                    detector_bbox=before.detector_bbox,
                    shot_id=before.shot_id,
                    identity_state=before.identity_state,
                    identity_confidence=before.identity_confidence,
                    identity_margin=before.identity_margin,
                    scene_state=before.scene_state,
                    review_status=before.review_status,
                    detector_confidence=before.detector_confidence,
                    pose_confidence=before.pose_confidence,
                    segment_id=before.segment_id,
                    physical_track_id=before.physical_track_id,
                    identity_origin=before.identity_origin,
                )
            )
    return sampled


def _write_observation_cache(
    destination: Path,
    observations: Iterable[PoseObservation],
) -> Path:
    """Stream observations to a compact, atomic JSONL cache."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            for observation in observations:
                json.dump(
                    observation.to_dict(),
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
        # gzip.open closes and flushes the compressed stream before the atomic
        # rename. fsync makes the cache robust against an interrupted rebuild.
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(destination: Path, records: Iterable[dict[str, object]]) -> Path:
    """Atomically write one compact JSON object per line."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                json.dump(
                    record,
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def _write_analysis_manifests(
    run_dir: Path,
    *,
    config: AnalysisConfig,
    backend_name: str,
    observations: list[PoseObservation],
    duration_ms: int,
) -> dict[str, str]:
    """Persist inspectable identity/scene/model artifacts for review and export."""

    assets_dir = run_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    body_asset_relative: str | None = None
    if _BODY_MAP_ASSET.is_file():
        body_destination = assets_dir / _BODY_MAP_ASSET.name
        shutil.copy2(_BODY_MAP_ASSET, body_destination)
        body_asset_relative = body_destination.relative_to(run_dir).as_posix()
    for asset_name in ("body-map-v3-head.png", "body-map-v3-body.png", "body-map-v3-manifest.json"):
        source = _BODY_MAP_ASSET.parent / asset_name
        if source.is_file():
            shutil.copy2(source, assets_dir / asset_name)

    body_manifest = run_dir / "body_map_manifest.json"
    atomic_write_json(
        body_manifest,
        {
            "version": 3,
            "base": body_asset_relative,
            "head_mask": "assets/body-map-v3-head.png",
            "body_mask": "assets/body-map-v3-body.png",
            "viewport_crop": {"x": .18, "y": 0, "width": .64, "height": .72},
            "coordinate_space": "defender_front_canonical_v1",
            "canvas_size": [1024, 1536],
            "zones": {
                "head": {"default_point": {"x": 0.5, "y": 0.105}},
                "body": {"default_point": {"x": 0.5, "y": 0.315}},
            },
            "production_colors_embedded": False,
        },
    )

    model_manifest = run_dir / "model_manifest.json"
    atomic_write_json(
        model_manifest,
        {
            "version": 1,
            "bundle_id": config.model_bundle_id,
            "runtime_backend": backend_name,
            "commercial_status": "research_baseline_pending_weight_provenance",
            "components": [
                {
                    "name": backend_name,
                    "code_license": "see THIRD_PARTY_NOTICES.md",
                    "weight_license": "pending_review",
                    "commercial_approval": False,
                    "training_datasets": "not_independently_verified",
                },
                {
                    "name": "boxing_vision_temporal_fsm_v2",
                    "code_license": "project",
                    "weight_license": None,
                    "commercial_approval": True,
                },
            ],
        },
    )
    from .pose import _LIGHTWEIGHT_MODEL_HASHES
    model_record = json.loads(model_manifest.read_text(encoding="utf-8"))
    model_record["components"].extend({
        "name": filename, "source_url": url, "sha256": digest,
        "sha256_checked_on_load": backend_name != "unavailable",
        "code_license": "Apache-2.0", "weight_license": "pending_review",
        "training_datasets": "not_independently_verified", "commercial_approval": False,
    } for url, (filename, digest) in _LIGHTWEIGHT_MODEL_HASHES.items())
    model_record["components"].extend([
        {"name": "Roboflow Trackers", "version": "2.6.0", "source_url": "https://github.com/roboflow/trackers",
         "code_license": "Apache-2.0", "weights": None},
        {"name": "PySceneDetect", "version": "0.7.1", "source_url": "https://github.com/Breakthrough/PySceneDetect",
         "code_license": "BSD-3-Clause", "weights": None},
    ])
    atomic_write_json(model_manifest, model_record)
    notices = Path(__file__).resolve().parent.parent / "THIRD_PARTY_NOTICES.md"
    if notices.is_file():
        shutil.copy2(notices, assets_dir / "THIRD_PARTY_NOTICES.md")

    by_tracklet: dict[tuple[int, str], list[PoseObservation]] = defaultdict(list)
    for observation in observations:
        track_id = str(
            observation.source_track_id
            if observation.source_track_id is not None
            else observation.fighter_id
        )
        by_tracklet[(int(observation.shot_id), track_id)].append(observation)

    tracklet_records: list[dict[str, object]] = []
    review_items: list[dict[str, object]] = []
    for (shot_id, source_track_id), track in sorted(by_tracklet.items()):
        track.sort(key=lambda item: item.timestamp_ms)
        identity_states = [str(item.identity_state) for item in track]
        identity_state = max(set(identity_states), key=identity_states.count)
        confidences = [float(item.identity_confidence or 0.0) for item in track]
        margins = [
            float(item.identity_margin)
            for item in track
            if item.identity_margin is not None
        ]
        record: dict[str, object] = {
            "tracklet_id": f"shot-{shot_id}-track-{source_track_id}",
            "source_track_id": source_track_id,
            "shot_id": shot_id,
            "start_ms": track[0].timestamp_ms,
            "end_ms": track[-1].timestamp_ms,
            "frames": len(track),
            "identity_state": identity_state,
            "identity_confidence": round(sum(confidences) / len(confidences), 4),
            "identity_margin": round(min(margins), 4) if margins else None,
            "review_status": str(track[-1].review_status),
        }
        tracklet_records.append(record)
        if (
            identity_state == str(IdentityState.UNKNOWN)
            or float(record["identity_confidence"]) < 0.55
            or (
                record["identity_margin"] is not None
                and float(record["identity_margin"])
                < config.identity_margin_min
            )
        ):
            review_items.append(
                {
                    **record,
                    "reason": "identity_uncertain",
                    "required": True,
                }
            )

    if not observations:
        review_items.append(
            {
                "review_id": "identity-enrollment",
                "shot_id": 0,
                "start_ms": 0,
                "end_ms": max(0, int(duration_ms)),
                "identity_state": str(IdentityState.UNKNOWN),
                "identity_confidence": 0.0,
                "identity_margin": None,
                "review_status": str(ReviewStatus.NEEDS_REVIEW),
                "reason": "fighters_not_confirmed",
                "required": True,
            }
        )

    tracklets_path = _write_jsonl(run_dir / "tracklets.jsonl", tracklet_records)
    diagnostics_path = _write_jsonl(
        run_dir / "tracking_diagnostics.jsonl",
        (
            {
                "timestamp_ms": item.timestamp_ms,
                "shot_id": item.shot_id,
                "source_track_id": item.source_track_id,
                "fighter_id": item.fighter_id,
                "identity_state": str(item.identity_state),
                "identity_confidence": round(float(item.identity_confidence or 0.0), 4),
                "identity_margin": (
                    round(float(item.identity_margin), 4)
                    if item.identity_margin is not None
                    else None
                ),
                "scene_state": str(item.scene_state),
                "review_status": str(item.review_status),
            }
            for item in observations
        ),
    )

    scenes: list[dict[str, object]] = []
    by_shot: dict[int, list[PoseObservation]] = defaultdict(list)
    for observation in observations:
        by_shot[int(observation.shot_id)].append(observation)
    for shot_id, shot in sorted(by_shot.items()):
        shot.sort(key=lambda item: item.timestamp_ms)
        states = [str(item.scene_state) for item in shot]
        scenes.append(
            {
                "shot_id": shot_id,
                "start_ms": shot[0].timestamp_ms,
                "end_ms": shot[-1].timestamp_ms,
                "scene_state": max(set(states), key=states.count),
                "review_status": (
                    str(ReviewStatus.NEEDS_REVIEW)
                    if any(item in {str(SceneState.UNCERTAIN)} for item in states)
                    else str(ReviewStatus.AUTO_CONFIRMED)
                ),
            }
        )
    if not scenes:
        scenes.append(
            {
                "shot_id": 0,
                "start_ms": 0,
                "end_ms": max(0, int(duration_ms)),
                "scene_state": str(SceneState.UNCERTAIN),
                "review_status": str(ReviewStatus.NEEDS_REVIEW),
            }
        )
    scenes_path = run_dir / "scenes.json"
    atomic_write_json(scenes_path, scenes)

    identity_profile = run_dir / "identity_profile.json"
    atomic_write_json(
        identity_profile,
        {
            "version": 2,
            "enrollment_mode": config.enrollment_mode,
            "enrollment_frames": list(config.enrollment_frames),
            "ring_roi": [list(point) for point in config.ring_rois],
            "thresholds": {
                "gallery_distance_max": config.identity_gallery_distance_max,
                "identity_margin_min": config.identity_margin_min,
                "adaptive_confidence_min": config.adaptive_identity_confidence_min,
                "adaptive_margin_min": config.adaptive_identity_margin_min,
            },
            "fighters": {
                "fighter_a": {"role": str(IdentityState.FIGHTER_A), "color": "#FF514A"},
                "fighter_b": {"role": str(IdentityState.FIGHTER_B), "color": "#5682FF"},
            },
        },
    )
    review_path = run_dir / "review.json"
    atomic_write_json(
        review_path,
        {
            "version": 1,
            "required_count": len(review_items),
            "items": review_items,
        },
    )
    return {
        "body_map_manifest": body_manifest.name,
        "model_manifest": model_manifest.name,
        "tracklets": tracklets_path.name,
        "tracking_diagnostics": diagnostics_path.name,
        "scenes": scenes_path.name,
        "identity_profile": identity_profile.name,
        "review": review_path.name,
    }


def _read_observation_cache(source: Path) -> list[PoseObservation]:
    observations: list[PoseObservation] = []
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    bbox_raw = raw["bbox"]
                    keypoints_raw = raw["keypoints"]
                    observation = PoseObservation(
                        frame_index=int(raw["frame_index"]),
                        timestamp_ms=int(raw["timestamp_ms"]),
                        fighter_id=str(raw["fighter_id"]),
                        bbox=BBox(
                            x1=float(bbox_raw["x1"]),
                            y1=float(bbox_raw["y1"]),
                            x2=float(bbox_raw["x2"]),
                            y2=float(bbox_raw["y2"]),
                            score=float(bbox_raw.get("score", 1.0)),
                        ),
                        keypoints={
                            str(name): Keypoint(
                                x=float(point["x"]),
                                y=float(point["y"]),
                                score=float(point.get("score", 1.0)),
                            )
                            for name, point in keypoints_raw.items()
                        },
                        track_confidence=float(raw.get("track_confidence", 1.0)),
                        is_scene_cut=bool(raw.get("is_scene_cut", False)),
                        source_track_id=raw.get("source_track_id"),
                        detector_bbox=(
                            BBox(
                                x1=float(raw["detector_bbox"]["x1"]),
                                y1=float(raw["detector_bbox"]["y1"]),
                                x2=float(raw["detector_bbox"]["x2"]),
                                y2=float(raw["detector_bbox"]["y2"]),
                                score=float(raw["detector_bbox"].get("score", 1.0)),
                            )
                            if isinstance(raw.get("detector_bbox"), dict)
                            else None
                        ),
                        shot_id=int(raw.get("shot_id", 0)),
                        identity_state=str(
                            raw.get("identity_state") or IdentityState.UNKNOWN
                        ),
                        identity_confidence=float(
                            raw.get(
                                "identity_confidence",
                                raw.get("track_confidence", 1.0),
                            )
                        ),
                        identity_margin=(
                            float(raw["identity_margin"])
                            if raw.get("identity_margin") is not None
                            else None
                        ),
                        scene_state=str(
                            raw.get("scene_state") or SceneState.ACTIVE_FIGHT
                        ),
                        review_status=str(
                            raw.get("review_status")
                            or ReviewStatus.AUTO_CONFIRMED
                        ),
                        detector_confidence=raw.get("detector_confidence"),
                        pose_confidence=raw.get("pose_confidence"),
                        identity_rejection_reason=raw.get("identity_rejection_reason"),
                        segment_id=raw.get("segment_id"),
                        physical_track_id=raw.get("physical_track_id"),
                        identity_origin=raw.get("identity_origin"),
                    )
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    raise RenderCacheUnavailableError(
                        f"Кэш поз повреждён в строке {line_number}; запустите анализ заново"
                    ) from exc
                observations.append(observation)
    except (OSError, EOFError) as exc:
        raise RenderCacheUnavailableError(
            "Кэш поз повреждён; запустите анализ заново"
        ) from exc
    return observations


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RenderCacheUnavailableError(
            f"{label} недоступен; запустите анализ заново"
        ) from exc
    if not isinstance(payload, dict):
        raise RenderCacheUnavailableError(f"{label} повреждён; запустите анализ заново")
    return payload


def _config_from_render_cache(payload: dict[str, object]) -> tuple[AnalysisConfig, int]:
    try:
        version = int(payload.get("version", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise RenderCacheUnavailableError(
            "Версия render-cache повреждена; запустите анализ заново"
        ) from exc
    if version not in {1, 2, _RENDER_CACHE_VERSION}:
        raise RenderCacheUnavailableError(
            "Версия render-cache не поддерживается; запустите анализ заново"
        )
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise RenderCacheUnavailableError(
            "Конфигурация render-cache повреждена; запустите анализ заново"
        )
    allowed = set(AnalysisConfig.__dataclass_fields__)
    config_values = {
        key: value
        for key, value in raw_config.items()
        if key in allowed
        and key not in {"fighter_a_portrait_path", "fighter_b_portrait_path"}
    }
    for key in (
        "fighter_a_anchor",
        "fighter_b_anchor",
        "enrollment_frames",
        "confirmed_knockdowns_a_rounds",
        "confirmed_knockdowns_b_rounds",
    ):
        value = config_values.get(key)
        if isinstance(value, list):
            config_values[key] = tuple(value)
    ring_rois = config_values.get("ring_rois")
    if isinstance(ring_rois, list):
        config_values["ring_rois"] = tuple(
            tuple(float(coordinate) for coordinate in point)
            for point in ring_rois
            if isinstance(point, (list, tuple))
        )
    try:
        config = AnalysisConfig(**config_values)
        config.validate()
        rounds_to_score = min(
            config.scheduled_rounds,
            max(1, int(payload.get("rounds_to_score", 1) or 1)),
        )
    except (TypeError, ValueError) as exc:
        raise RenderCacheUnavailableError(
            "Конфигурация render-cache повреждена; запустите анализ заново"
        ) from exc
    return config, rounds_to_score


def _events_from_json(path: Path) -> list[PunchEvent]:
    try:
        raw_events = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RenderCacheUnavailableError("events.json недоступен") from exc
    if not isinstance(raw_events, list):
        raise RenderCacheUnavailableError("events.json повреждён")
    allowed = set(PunchEvent.__dataclass_fields__)
    events: list[PunchEvent] = []
    try:
        for raw in raw_events:
            if not isinstance(raw, dict):
                raise TypeError("event must be an object")
            events.append(
                PunchEvent(
                    **{key: value for key, value in raw.items() if key in allowed}
                )
            )
    except (TypeError, ValueError) as exc:
        raise RenderCacheUnavailableError("events.json повреждён") from exc
    return events


def _replace_hover_previews(
    annotated_video: Path,
    run_dir: Path,
    cache_dir: Path,
) -> Path:
    """Build previews off to the side, then replace the complete set."""

    build_root = cache_dir / f"preview-build-{uuid.uuid4().hex}"
    backup_dir = cache_dir / f"preview-backup-{uuid.uuid4().hex}"
    live_dir = run_dir / "previews"
    installed_new_dir = False
    moved_old_dir = False
    try:
        generated_manifest = generate_hover_previews(annotated_video, build_root)
        generated_dir = build_root / "previews"
        if live_dir.exists():
            os.replace(live_dir, backup_dir)
            moved_old_dir = True
        os.replace(generated_dir, live_dir)
        installed_new_dir = True
        os.replace(generated_manifest, run_dir / "preview_manifest.json")
        shutil.rmtree(backup_dir, ignore_errors=True)
        return run_dir / "preview_manifest.json"
    except BaseException:
        if installed_new_dir:
            shutil.rmtree(live_dir, ignore_errors=True)
        if moved_old_dir and backup_dir.exists():
            os.replace(backup_dir, live_dir)
        raise
    finally:
        shutil.rmtree(build_root, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)


def _enrich_summary(
    summary: dict[str, object],
    *,
    config: AnalysisConfig,
    duration_s: float,
    processing_s: float,
    backend_name: str,
    observations: list[PoseObservation],
    events: list[PunchEvent],
    portrait_filenames: dict[str, str] | None = None,
    frame_states: list[dict[str, object]] | None = None,
    tracking_diagnostics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    portrait_filenames = portrait_filenames or {}
    fighters = summary.get("fighters")
    if isinstance(fighters, dict):
        profiles = {
            "fighter_a": (config.fighter_a_record, portrait_filenames.get("fighter_a")),
            "fighter_b": (config.fighter_b_record, portrait_filenames.get("fighter_b")),
        }
        for fighter_id, corner in (("fighter_a", "red"), ("fighter_b", "blue")):
            fighter = fighters.get(fighter_id)
            if isinstance(fighter, dict):
                fighter["id"] = fighter_id
                fighter["corner"] = corner
                record, portrait_filename = profiles[fighter_id]
                if record and record.strip():
                    fighter["record"] = record.strip()
                if portrait_filename:
                    fighter["portrait_filename"] = portrait_filename
                # The UI accepts a nested stats object while the renderer uses
                # the same values directly. Keeping both makes the JSON useful
                # outside Gradio without changing the render contract.
                fighter["stats"] = {
                    key: value
                    for key, value in fighter.items()
                    if key
                    not in {
                        "id",
                        "name",
                        "corner",
                        "record",
                        "portrait_filename",
                        "stats",
                    }
                }

    winner_id = summary.get("winner_id")
    winner_name = str(summary.get("winner_name") or "Ничья")
    confidence = float(summary.get("confidence") or 0.0)
    summary["winner"] = {
        "fighter_id": winner_id,
        "name": winner_name,
        "confidence": round(confidence, 4),
        "label": "Прогноз модели" if winner_id else "Равный результат по модели",
    }
    tracking_confidences = [observation.track_confidence for observation in observations]
    identities_by_timestamp: dict[int, list[str]] = {}
    identity_confidences: list[float] = []
    low_margin_frames = 0
    identity_state_mismatches = 0
    expected_states = {
        "fighter_a": IdentityState.FIGHTER_A,
        "fighter_b": IdentityState.FIGHTER_B,
    }
    for observation in observations:
        expected = expected_states.get(observation.fighter_id)
        confidence = float(observation.identity_confidence or 0.0)
        margin = observation.identity_margin
        identity_confidences.append(confidence)
        if expected is not None and observation.identity_state != expected:
            identity_state_mismatches += 1
        if margin is not None and float(margin) < config.identity_margin_min:
            low_margin_frames += 1
        confirmed = is_confirmed_identity(
            observation, minimum_margin=config.identity_margin_min
        )
        if confirmed:
            identities_by_timestamp.setdefault(observation.timestamp_ms, []).append(
                observation.fighter_id
            )
    expected_analysis_frames = max(1, (
        sum(record.get("scene_state") == "ACTIVE_FIGHT" for record in frame_states)
        if frame_states is not None else sum(
            _scheduled_scene_state(round(index * 1000 / config.analysis_fps), config, shot_id=0) == SceneState.ACTIVE_FIGHT
            for index in range(round(duration_s * config.analysis_fps))
        )
    ))
    verified_pair_frames = sum(
        set(identities) >= {"fighter_a", "fighter_b"}
        for identities in identities_by_timestamp.values()
    )
    verified_coverage = min(1.0, verified_pair_frames / expected_analysis_frames)
    duplicate_role_frames = sum(
        len(identities) != len(set(identities))
        for identities in identities_by_timestamp.values()
    )
    # Unknown or low-margin observations are abstentions, not evidence of a swap.
    source_roles: dict[tuple[int, str], set[str]] = defaultdict(set)
    for observation in observations:
        if observation.source_track_id is not None and observation.identity_state in expected_states.values():
            source_roles[(observation.shot_id, str(observation.segment_id or observation.source_track_id))].add(str(observation.identity_state))
    identity_swap_suspected = bool(duplicate_role_frames or any(len(roles) > 1 for roles in source_roles.values()))
    unresolved_conflicts = sum(
        bool(row.get("identity_conflict")) or row.get("reason") in {
            "source_identity_conflict", "segment_identity_conflict", "identity_transition_conflict"
        }
        or (row.get("segment_reason") == "identity_conflict_boundary"
            and row.get("identity_origin") != "user_confirmed")
        for row in (tracking_diagnostics or [])
        if str(row.get("review_status", "")) != "USER_CONFIRMED"
    )
    identity_swap_suspected |= bool(unresolved_conflicts)
    event_confidences = [event.confidence for event in events if not event.is_replay]
    summary["metadata"] = {
        "duration_s": round(duration_s, 3),
        "processing_s": round(processing_s, 3),
        "backend": backend_name,
        "analysis_fps": config.analysis_fps,
        "output_fps": config.output_fps,
        "scheduled_rounds": config.scheduled_rounds,
        "round_length_s": config.round_length_s,
        "rest_length_s": config.rest_length_s,
        "fight_start_s": round(config.fight_start_s, 3),
        "fight_end_s": round(
            config.fight_end_s
            if config.fight_end_s is not None
            else config.fight_start_s + duration_s,
            3,
        ),
        "hud_mode": config.hud_mode,
        "timing_mode": getattr(config, "timing_mode", "scheduled"),
        "tracking_overlay_style": getattr(config, "tracking_overlay_style", "full"),
        "enrollment_mode": config.enrollment_mode,
        "model_bundle_id": config.model_bundle_id,
        "body_map_manifest": "body_map_manifest.json",
        "model_manifest": "model_manifest.json",
        "render_cache_available": True,
        "export_stale": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "confirmed_knockdowns_suffered": _confirmed_knockdowns(config),
        "disclaimer": "Оценка модели с указанием уверенности.",
    }
    summary["quality"] = {
        "tracking_confidence": round(
            sum(tracking_confidences) / len(tracking_confidences), 4
        )
        if tracking_confidences
        else 0.0,
        "event_confidence": round(sum(event_confidences) / len(event_confidences), 4)
        if event_confidences
        else 0.0,
        "pose_observations": len(observations),
        "event_candidates": len(events),
        "event_rate_per_second": round(len(events) / max(0.001, duration_s), 4),
        "identity_verified_coverage": round(verified_coverage, 4),
        "coverage_basis": "active_fight_samples_not_ground_truth_visibility",
        "active_fight_samples": expected_analysis_frames,
        "identity_precision_measured": False,
        "identity_confidence": round(
            sum(identity_confidences) / len(identity_confidences), 4
        )
        if identity_confidences
        else 0.0,
        "identity_low_margin_frames": low_margin_frames,
        "identity_state_mismatches": identity_state_mismatches,
        "identity_swap_suspected": identity_swap_suspected,
        "unresolved_identity_conflict_samples": unresolved_conflicts,
        "winner_visible": bool(
            verified_coverage >= 0.90 and not identity_swap_suspected
        ),
        "event_wording": "кандидаты ударов",
        "possible_knockdowns": sum(
            bool(event.evidence.get("possible_knockdown")) for event in events
        ),
    }
    return apply_result_gate(summary)


def _scheduled_scene_state(
    timestamp_ms: int,
    config: AnalysisConfig,
    *,
    shot_id: int,
    is_replay: bool = False,
) -> SceneState:
    """Resolve a conservative scene gate for normalized fight time."""

    if is_replay:
        return SceneState.REPLAY
    override = config.scene_overrides.get(str(shot_id))
    if override is None:
        override = config.scene_overrides.get(f"shot-{shot_id}")
    if override is not None:
        try:
            return SceneState(str(override).upper())
        except ValueError:
            return SceneState.UNCERTAIN

    if getattr(config, "timing_mode", "scheduled") == "continuous":
        return SceneState.ACTIVE_FIGHT

    elapsed_s = max(0.0, timestamp_ms / 1000.0)
    cycle_s = max(1, config.round_length_s + config.rest_length_s)
    round_index = int(elapsed_s // cycle_s)
    if round_index >= config.scheduled_rounds:
        return SceneState.NON_FIGHT
    phase_s = elapsed_s - round_index * cycle_s
    if phase_s >= config.round_length_s:
        return SceneState.BREAK
    return SceneState.ACTIVE_FIGHT


def _render_video(
    normalized_video: Path,
    silent_output: Path,
    *,
    observations: list[PoseObservation],
    events: list[PunchEvent],
    config: AnalysisConfig,
    final_summary: dict[str, object],
    rounds_to_score: int,
    confirmed_knockdowns: dict[int, dict[str, int]],
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
    workspace_silent_output: Path | None = None,
    display_tracks: list[DisplayTrack] | None = None,
    tracking_only: bool = False,
) -> None:
    metadata = probe_video(normalized_video)
    pose_tracks: dict[str, list[PoseObservation]] = {}
    for observation in observations:
        pose_tracks.setdefault(observation.fighter_id, []).append(observation)
    for track in pose_tracks.values():
        track.sort(key=lambda observation: observation.timestamp_ms)
    pose_indices = {fighter_id: 0 for fighter_id in pose_tracks}
    from .display_tracking import DisplayTrackSampler
    from .tracking_artifacts import read_display_cache
    display_path = normalized_video.parent / "display_tracks.jsonl.gz"
    if display_tracks is None and display_path.is_file():
        display_tracks = read_display_cache(display_path)
    display_sampler = DisplayTrackSampler(display_tracks) if display_tracks is not None else None

    fighter_names = {
        "fighter_a": config.fighter_a_name,
        "fighter_b": config.fighter_b_name,
    }
    renderer = FrameRenderer(
        fighter_names,
        pose_threshold=config.pose_score_threshold,
        trail_length=10,
        hud_width_ratio=0.29,
        hud_mode=config.hud_mode,
        tracking_overlay_style=getattr(config, "tracking_overlay_style", "full"),
    )
    # Compact export can snapshot the already drawn tracking layer. Technical
    # and none retain their historical export appearance while preview uses a
    # separate, inexpensive renderer on the very same decoded frame.
    workspace_renderer = (
        FrameRenderer(fighter_names, pose_threshold=config.pose_score_threshold,
                      trail_length=10, hud_mode="compact",
                      tracking_overlay_style=getattr(config, "tracking_overlay_style", "full"))
        if workspace_silent_output is not None and config.hud_mode != "compact"
        else None
    )

    writer: cv2.VideoWriter | None = None
    workspace_writer: cv2.VideoWriter | None = None
    event_cursor = 0
    scored_cursor = 0
    active_events: deque[PunchEvent] = deque()
    seen_events: list[PunchEvent] = []
    visible_events = _renderable_events(events)
    ordered_events = sorted(visible_events, key=lambda event: event.start_ms)
    events_by_peak = sorted(visible_events, key=lambda event: event.peak_ms)
    running_summary = build_fight_summary(
        [],
        fighter_names=fighter_names,
        scheduled_rounds=rounds_to_score,
        confirmed_knockdowns_suffered=confirmed_knockdowns,
    )
    final_metadata = final_summary.get("metadata")
    if isinstance(final_metadata, dict):
        running_summary["metadata"] = dict(final_metadata)
    running_scores = score_rounds(
        [],
        scheduled_rounds=rounds_to_score,
        confirmed_knockdowns_suffered=confirmed_knockdowns,
    )
    rendered_frames = 0
    shot_id = 0
    result_allowed, _ = result_eligibility(final_summary)
    first_pass_path = normalized_video.parent / "first_pass.json"
    frame_states = []
    if first_pass_path.is_file():
        frame_states = _read_json_object(first_pass_path, label="Кэш сцен").get("frame_states", [])
    context_index = 0

    try:
        for video_frame in iter_video_frames(normalized_video):
            _check_cancel(cancel_callback)
            frame = video_frame.image
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(silent_output),
                    fourcc,
                    float(metadata.fps or config.output_fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("OpenCV не смог создать промежуточное видео")
                if workspace_silent_output is not None:
                    workspace_writer = cv2.VideoWriter(
                        str(workspace_silent_output), fourcc,
                        float(metadata.fps or config.output_fps), (width, height),
                    )
                    if not workspace_writer.isOpened():
                        raise RuntimeError("OpenCV не смог создать чистое видео рабочего места")

            timestamp_ms = video_frame.timestamp_ms
            cached_context = None
            if frame_states:
                while context_index + 1 < len(frame_states) and frame_states[context_index + 1]["timestamp_ms"] <= timestamp_ms:
                    context_index += 1
                cached_context = frame_states[context_index]
            current_observations = _sample_pose_tracks(
                pose_tracks,
                pose_indices,
                timestamp_ms,
            )
            observed_shots = {
                int(observation.shot_id) for observation in current_observations
            }
            observed_shot = max(observed_shots) if observed_shots else shot_id
            is_scene_cut = observed_shot != shot_id or any(
                observation.is_scene_cut for observation in current_observations
            )
            shot_id = observed_shot
            scene_states = {
                SceneState(str(observation.scene_state))
                for observation in current_observations
                if str(observation.scene_state) in {item.value for item in SceneState}
            }
            if len(scene_states) == 1:
                scene_state = next(iter(scene_states))
            elif len(scene_states) > 1:
                scene_state = SceneState.UNCERTAIN
            else:
                scene_state = _scheduled_scene_state(
                    timestamp_ms,
                    config,
                    shot_id=shot_id,
                )
            if cached_context:
                next_shot = int(cached_context["shot_id"])
                is_scene_cut = next_shot != observed_shot or (
                    bool(cached_context.get("is_scene_cut")) and abs(timestamp_ms - int(cached_context["timestamp_ms"])) < 1000 / config.output_fps
                )
                shot_id = next_shot
                scene_state = SceneState(cached_context["scene_state"])
                current_observations = [obs for obs in current_observations if obs.shot_id == shot_id]

            while (
                event_cursor < len(ordered_events)
                and ordered_events[event_cursor].start_ms - 150 <= timestamp_ms
            ):
                active_events.append(ordered_events[event_cursor])
                event_cursor += 1
            while active_events and active_events[0].end_ms + 450 < timestamp_ms:
                active_events.popleft()

            changed = False
            while (
                scored_cursor < len(events_by_peak)
                and events_by_peak[scored_cursor].peak_ms <= timestamp_ms
            ):
                seen_events.append(events_by_peak[scored_cursor])
                scored_cursor += 1
                changed = True
            if changed:
                running_summary = build_fight_summary(
                    seen_events,
                    fighter_names=fighter_names,
                    scheduled_rounds=rounds_to_score,
                    confirmed_knockdowns_suffered=confirmed_knockdowns,
                )
                if isinstance(final_metadata, dict):
                    running_summary["metadata"] = dict(final_metadata)
                running_scores = score_rounds(
                    seen_events,
                    scheduled_rounds=rounds_to_score,
                    confirmed_knockdowns_suffered=confirmed_knockdowns,
                )

            running_summary["quality"] = dict(final_summary.get("quality", {}))
            context = RenderFrameContext(
                timestamp_ms=timestamp_ms, shot_id=shot_id,
                is_scene_cut=is_scene_cut, scene_state=scene_state.value,
            )
            rendered = renderer.draw(
                frame,
                current_observations,
                (
                    list(active_events)
                    if scene_state == SceneState.ACTIVE_FIGHT
                    else []
                ),
                running_summary,
                running_scores if result_allowed else [],
                timestamp_ms,
                frame_context=context,
                include_hud=not tracking_only,
                display_tracks=display_sampler.sample(timestamp_ms, context) if display_sampler is not None else None,
                tracking_frame_callback=(
                    workspace_writer.write
                    if workspace_writer is not None and workspace_renderer is None
                    else None
                ),
            )
            writer.write(rendered)
            if workspace_writer is not None and workspace_renderer is not None:
                workspace_writer.write(workspace_renderer.draw(
                    frame, current_observations, (), running_summary, (), timestamp_ms,
                    frame_context=context, include_hud=False,
                    display_tracks=display_sampler.sample(timestamp_ms, context) if display_sampler is not None else None,
                ))
            rendered_frames += 1
            if rendered_frames % max(1, round(metadata.fps)) == 0:
                _emit(
                    progress_callback,
                    0.72
                    + 0.20
                    * min(1.0, timestamp_ms / max(1.0, metadata.duration_s * 1000.0)),
                    "Формируем размеченную трансляцию",
                )
    finally:
        if writer is not None:
            writer.release()
        if workspace_writer is not None:
            workspace_writer.release()
    if rendered_frames == 0 or not silent_output.is_file():
        raise VideoValidationError("В нормализованном видео нет декодируемых кадров")


def _create_identity_tracker(config: AnalysisConfig) -> TwoFighterTracker:
    """Build the tracker from the complete persisted identity policy."""

    return TwoFighterTracker(
        ("fighter_a", "fighter_b"),
        max_missing_frames=12,
        gallery_max_distance=config.identity_gallery_distance_max,
        gallery_min_margin=config.identity_margin_min,
        adaptive_identity_confidence_min=config.adaptive_identity_confidence_min,
        adaptive_identity_margin_min=config.adaptive_identity_margin_min,
    )


@_cleanup_failed_job
def analyze_video(
    input_path: str | Path,
    config: AnalysisConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    *,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    cancel_callback: CancelCallback | None = None,
) -> AnalysisResult:
    """Run the complete local, offline boxing-analysis workflow."""

    started = time.perf_counter()
    config = config or AnalysisConfig()
    config.validate()
    _check_cancel(cancel_callback)
    source = Path(input_path).expanduser().resolve()
    artifacts = create_job_artifacts(runs_dir)
    _ACTIVE_ARTIFACTS.set(artifacts)
    render_cache_dir = artifacts.run_dir / _RENDER_CACHE_DIRNAME
    render_cache_dir.mkdir(exist_ok=False)
    normalized_video = render_cache_dir / _RENDER_CACHE_VIDEO

    def log(message: str) -> None:
        with artifacts.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(UTC).isoformat()} {message}\n")

    reported_fraction = 0.0

    def report(fraction: float, description: str) -> None:
        nonlocal reported_fraction
        _check_cancel(cancel_callback)
        reported_fraction = max(reported_fraction, fraction)
        _emit(progress_callback, reported_fraction, description)

    def cancellation_probe() -> bool:
        _check_cancel(cancel_callback)
        return False

    log(f"job={artifacts.job_id} source_name={source.name}")
    report(0.01, "Проверяем видео")
    source_metadata = validate_video(source, max_duration_s=config.max_duration_s)
    input_metadata = source_metadata.to_dict()
    input_metadata["path"] = source.name
    log(f"input={input_metadata}")

    portrait_filenames: dict[str, str] = {}
    for fighter_id, portrait_path in (
        ("fighter_a", config.fighter_a_portrait_path),
        ("fighter_b", config.fighter_b_portrait_path),
    ):
        portrait_filename = persist_fighter_portrait(
            portrait_path,
            artifacts.run_dir,
            fighter_id,
        )
        if portrait_filename:
            portrait_filenames[fighter_id] = portrait_filename

    report(0.04, "Подготавливаем видео 720p CFR")
    normalize_video(
        source,
        normalized_video,
        output_height=config.output_height,
        output_fps=config.output_fps,
        start_s=config.fight_start_s,
        end_s=config.fight_end_s,
        max_duration_s=config.max_duration_s,
        cancel_callback=cancellation_probe,
        progress_callback=lambda value: report(
            0.04 + value * 0.11,
            "Подготавливаем видео 720p CFR",
        ),
    )
    normalized_metadata = probe_video(normalized_video)

    report(0.16, "Загружаем RTMPose")
    backend = create_pose_backend(
        config.backend,
        strict=config.backend != "none",
        mode="lightweight",
        # RTMLib's YOLOX graph currently fails in ONNX Runtime's CoreML EP on
        # this Mac (static output rank mismatch). CPU is stable and benchmarks
        # at roughly 20 analysed frames/s after warm-up on the test footage.
        device="cpu",
        pose_score_threshold=config.pose_score_threshold,
        detector_score_threshold=config.detector_score_threshold,
        detector_nms_threshold=config.detector_nms_threshold,
    )
    tracker = _create_identity_tracker(config)
    cut_detector = SceneCutDetector()
    replay_detector = ReplayDetector()
    observations: list[PoseObservation] = []
    replay_intervals: list[tuple[int, int]] = []
    replay_start: int | None = None
    analysed_frames = 0
    anchors_initialized = False

    first_pass = None
    if config.backend != "none" and config.enrollment_mode != "legacy_anchor":
        from .first_pass import run_first_pass
        first_pass = run_first_pass(
            normalized_video, render_cache_dir, backend, tracker, config,
            progress=report, cancelled=cancellation_probe,
            check_cancel=lambda: _check_cancel(cancel_callback),
        )
        observations = first_pass.observations
        replay_intervals = first_pass.replay_intervals
        analysed_frames = len(first_pass.frame_states)
        log(f"preflight={first_pass.preflight}")

    for video_frame in (iter_video_frames(
        normalized_video, target_fps=config.analysis_fps
    ) if first_pass is None else ()):
        _check_cancel(cancel_callback)
        if not anchors_initialized:
            if (
                config.fighter_a_anchor is not None
                and config.fighter_b_anchor is not None
            ):
                height, width = video_frame.image.shape[:2]
                tracker.set_anchors(
                    (
                        config.fighter_a_anchor[0] * width,
                        config.fighter_a_anchor[1] * height,
                    ),
                    (
                        config.fighter_b_anchor[0] * width,
                        config.fighter_b_anchor[1] * height,
                    ),
                )
                log("fighter anchors initialized from confirmation frame")
            anchors_initialized = True
        scene_cut, cut_score = cut_detector.update(video_frame.image)
        if scene_cut and replay_start is not None:
            replay_intervals.append(
                (replay_start, max(replay_start, video_frame.timestamp_ms - 1))
            )
            replay_start = None
        possible_replay = replay_detector.update(
            video_frame.image,
            video_frame.timestamp_ms,
            is_scene_cut=scene_cut,
        )
        if possible_replay:
            replay_start = (
                replay_detector.confirmed_start_ms
                if replay_detector.confirmed_start_ms is not None
                else video_frame.timestamp_ms
            )
            replay_shot_id = replay_detector.shot_id
            observations = [
                replace(observation, scene_state=SceneState.REPLAY)
                if observation.shot_id == replay_shot_id
                and observation.timestamp_ms >= replay_start
                else observation
                for observation in observations
            ]
        if scene_cut:
            backend.reset()
            log(
                f"scene_cut timestamp_ms={video_frame.timestamp_ms} score={cut_score:.3f}"
            )

        scene_state = _scheduled_scene_state(
            video_frame.timestamp_ms,
            config,
            shot_id=replay_detector.shot_id,
            is_replay=replay_detector.is_replay,
        )
        poses = backend.infer(video_frame.image)
        frame_observations = tracker.process(
            video_frame.frame_index,
            video_frame.timestamp_ms,
            poses,
            scene_cut=scene_cut,
            active_fight=scene_state == SceneState.ACTIVE_FIGHT,
        )
        frame_observations = [
            replace(
                observation,
                scene_state=scene_state,
                review_status=(
                    ReviewStatus.NEEDS_REVIEW
                    if scene_state == SceneState.UNCERTAIN
                    else ReviewStatus.AUTO_CONFIRMED
                ),
            )
            for observation in frame_observations
        ]
        observations.extend(frame_observations)
        assigned_fighters = {
            observation.fighter_id for observation in frame_observations
        }
        if (
            assigned_fighters == {"fighter_a", "fighter_b"}
            and tracker.anchors
            and set(tracker.core_gallery)
            >= {IdentityState.FIGHTER_A, IdentityState.FIGHTER_B}
        ):
            # The confirmation coordinates are evidence for the first identity
            # assignment only. They are not reused after broadcast camera cuts.
            tracker.clear_anchors()
        analysed_frames += 1
        if analysed_frames % max(1, round(config.analysis_fps)) == 0:
            report(
                0.17
                + 0.45
                * min(
                    1.0,
                    video_frame.timestamp_ms
                    / max(1.0, normalized_metadata.duration_s * 1000.0),
                ),
                "Отслеживаем бойцов и движения рук",
            )

    if replay_start is not None:
        replay_intervals.append(
            (replay_start, round(normalized_metadata.duration_s * 1000))
        )
    log(
        f"backend={backend.name} analysed_frames={analysed_frames} "
        f"observations={len(observations)} replay_intervals={replay_intervals}"
    )

    report(0.64, "Выделяем кандидаты ударов")
    normalized_config = replace(config, fight_start_s=0.0, fight_end_s=None)
    events = detect_punch_events(
        observations,
        normalized_config,
        stances={
            "fighter_a": config.fighter_a_stance,
            "fighter_b": config.fighter_b_stance,
        },
        replay_intervals=replay_intervals,
    )
    punch_manifest = None
    if config.punch_model != "baseline" and config.backend != "none":
        from .model_registry import ACM_BUNDLE_ID
        from .punch_models import AcmPunchClassifier, classify_candidate_events
        from .refinement import dense_candidate_poses
        bundle_dir = Path(__file__).resolve().parent.parent / "models" / ACM_BUNDLE_ID
        if (bundle_dir / "manifest.json").is_file():
            try:
                classifier = AcmPunchClassifier(bundle_dir)
                dense = dense_candidate_poses(normalized_video, observations, events, backend,
                    frame_states=first_pass.frame_states if first_pass is not None else None,
                    source_fps=config.analysis_fps,
                    check_cancel=lambda: _check_cancel(cancel_callback),
                    progress=lambda done, total: report(.64 + .015 * done / max(1, total), "Уточняем движения рук на 30 кадрах/с"))
                _write_observation_cache(render_cache_dir / "dense_poses.jsonl.gz", dense)
                events = classify_candidate_events(events, dense, classifier,
                    stances={"fighter_a": config.fighter_a_stance, "fighter_b": config.fighter_b_stance},
                    cancelled=cancellation_probe)
                punch_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
                log(f"punch_refiner={ACM_BUNDLE_ID} dense_observations={len(dense)}")
            except AnalysisCancelledError:
                raise
            except Exception as exc:
                if config.punch_model == "acm40960":
                    raise
                log(f"punch_refiner_unavailable={type(exc).__name__}: {exc}; baseline retained")
        elif config.punch_model == "acm40960":
            raise RuntimeError("Готовый классификатор не установлен. Выполните tools/prepare_punch_model.py")
    manifest_refs = _write_analysis_manifests(
        artifacts.run_dir,
        config=config,
        backend_name=backend.name,
        observations=observations,
        duration_ms=round(normalized_metadata.duration_s * 1000),
    )
    if first_pass is not None:
        _write_jsonl(artifacts.run_dir / "tracklets.jsonl", first_pass.tracklets)
        _write_jsonl(artifacts.run_dir / "tracking_diagnostics.jsonl", first_pass.diagnostics)
        atomic_write_json(artifacts.run_dir / "scenes.json", first_pass.scenes)
        atomic_write_json(artifacts.run_dir / "identity_profile.json", first_pass.identity_profile)
        review_items = [dict(scene, kind="scene", review_id=f"scene-{scene['shot_id']}", required=True)
                        for scene in first_pass.scenes if scene.get("review_status") == "NEEDS_REVIEW"]
        review_items.extend(dict(track, kind="tracklet", required=True) for track in first_pass.tracklets
                            if str(track.get("review_status")) == "NEEDS_REVIEW"
                            and track.get("eligible_for_review", True)
                            and str(track.get("identity_state")) != "OTHER")
        atomic_write_json(artifacts.run_dir / "review.json", {"version": 2, "required_count": len(review_items),
                          "items": review_items, "preflight": first_pass.preflight})
    if punch_manifest is not None:
        manifest_path = artifacts.run_dir / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["components"].append(punch_manifest)
        atomic_write_json(manifest_path, manifest)
    cycle_s = max(1, config.round_length_s + config.rest_length_s)
    rounds_to_score = max(
        1,
        min(
            config.scheduled_rounds, math.ceil(normalized_metadata.duration_s / cycle_s)
        ),
    )
    if getattr(config, "timing_mode", "scheduled") == "continuous":
        rounds_to_score = 1
    fighter_names = {
        "fighter_a": config.fighter_a_name,
        "fighter_b": config.fighter_b_name,
    }
    confirmed_knockdowns = _confirmed_knockdowns(config)
    report(0.665, "Сохраняем render-cache")
    _write_observation_cache(
        render_cache_dir / _RENDER_CACHE_OBSERVATIONS,
        observations,
    )
    atomic_write_json(
        render_cache_dir / _RENDER_CACHE_CONFIG,
        {
            "version": _RENDER_CACHE_VERSION,
            "config": config.to_dict(),
            "rounds_to_score": rounds_to_score,
        },
    )
    summary = build_fight_summary(
        events,
        fighter_names=fighter_names,
        scheduled_rounds=rounds_to_score,
        confirmed_knockdowns_suffered=confirmed_knockdowns,
    )
    processing_so_far = time.perf_counter() - started
    summary = _enrich_summary(
        summary,
        config=config,
        duration_s=normalized_metadata.duration_s,
        processing_s=processing_so_far,
        backend_name=backend.name,
        observations=observations,
        events=events,
        portrait_filenames=portrait_filenames,
        frame_states=first_pass.frame_states if first_pass is not None else None,
        tracking_diagnostics=first_pass.diagnostics if first_pass is not None else None,
    )
    if first_pass is not None:
        from .tracking_artifacts import read_detection_frames, write_tracking_artifacts
        report(0.67, "Сохраняем непрерывный трекинг и движения для проверки")
        display_manifest = write_tracking_artifacts(
            render_cache_dir, read_detection_frames(render_cache_dir / "detections.jsonl.gz"),
            observations, first_pass.diagnostics, events, normalized_config,
            frame_states=first_pass.frame_states,
        )
        summary["quality"].update(display_manifest)
        summary["quality"]["required_review_count"] = len(review_items)
        summary["quality"]["preflight"] = first_pass.preflight
        summary = apply_result_gate(summary)
    metadata = summary.get("metadata")
    if isinstance(metadata, dict):
        metadata.update(manifest_refs)
    log(f"events={len(events)} rounds={rounds_to_score}")

    workspace_silent = artifacts.work_dir / "workspace-preview.silent.mp4"
    workspace_video = artifacts.run_dir / _WORKSPACE_PREVIEW_FILENAME
    report(0.70, "Формируем размеченную трансляцию")
    _render_video(
        normalized_video,
        artifacts.silent_video,
        observations=observations,
        events=events,
        config=config,
        final_summary=summary,
        rounds_to_score=rounds_to_score,
        confirmed_knockdowns=confirmed_knockdowns,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        workspace_silent_output=workspace_silent,
    )
    report(0.93, "Кодируем итоговый H.264 и возвращаем звук")
    finalize_h264_video(
        artifacts.silent_video,
        normalized_video,
        artifacts.annotated_video,
        cancel_callback=cancellation_probe,
    )
    finalize_h264_video(
        workspace_silent, normalized_video, workspace_video,
        cancel_callback=cancellation_probe,
    )
    summary["metadata"]["workspace_preview"] = _WORKSPACE_PREVIEW_FILENAME
    summary["metadata"]["workspace_preview_mode"] = "tracking_only"

    report(0.945, "Создаём превью для таймлайна")
    try:
        preview_manifest = generate_hover_previews(
            workspace_video,
            artifacts.run_dir,
        )
        metadata = summary.get("metadata")
        if isinstance(metadata, dict):
            metadata["preview_manifest"] = preview_manifest.name
        log(f"preview_manifest={preview_manifest.name}")
    except Exception as exc:  # noqa: BLE001 - preview failure must not discard analysis
        # Hover previews improve navigation but are not part of the analytical
        # result.  A codec/Pillow issue must never discard an otherwise valid
        # annotated video and event set.
        shutil.rmtree(artifacts.run_dir / "previews", ignore_errors=True)
        (artifacts.run_dir / "preview_manifest.json").unlink(missing_ok=True)
        log(f"preview_generation_failed={type(exc).__name__}: {exc}")

    important_events = _important_events(events)
    if important_events:
        report(0.96, "Нарезаем ключевые эпизоды")
        extract_event_clips(
            artifacts.annotated_video,
            important_events,
            artifacts.clips_dir,
            padding_s=2.0,
            cancel_callback=cancellation_probe,
        )

    total_processing_s = time.perf_counter() - started
    metadata = summary.get("metadata")
    if isinstance(metadata, dict):
        metadata["processing_s"] = round(total_processing_s, 3)
    atomic_write_json(artifacts.events_path, [event.to_dict() for event in events])
    atomic_write_json(artifacts.summary_path, summary)
    log(f"completed processing_s={total_processing_s:.3f}")

    if not config.keep_debug:
        shutil.rmtree(artifacts.work_dir, ignore_errors=True)
    report(1.0, "Анализ завершён")
    return AnalysisResult(
        job_id=artifacts.job_id,
        run_dir=artifacts.run_dir,
        annotated_video=artifacts.annotated_video,
        events_path=artifacts.events_path,
        summary_path=artifacts.summary_path,
        log_path=artifacts.log_path,
        clips_dir=artifacts.clips_dir,
        events=events,
        summary=summary,
    )


def redecode_from_cache(run_dir: str | Path, *, identity_overrides: dict[str, str] | None = None,
                      scene_overrides: dict[str, str] | None = None,
                      segment_identity_overrides: dict[str, str] | None = None,
                      timing_mode: str | None = None, region_mode: str | None = None) -> dict:
    from .cache_review import redecode_from_cache as redecode
    return redecode(run_dir, identity_overrides=identity_overrides, scene_overrides=scene_overrides,
                    segment_identity_overrides=segment_identity_overrides, timing_mode=timing_mode, region_mode=region_mode)


def rebuild_tracking_preview_from_cache(
    run_dir: str | Path, progress_callback: ProgressCallback | None = None,
    *, cancel_callback: CancelCallback | None = None,
) -> Path:
    """Refresh the review player, without approving or producing final export."""
    root = Path(run_dir).resolve()
    cache = root / _RENDER_CACHE_DIRNAME
    config, rounds = _config_from_render_cache(_read_json_object(cache / _RENDER_CACHE_CONFIG, label="Render-cache"))
    summary = _read_json_object(root / "summary.json", label="summary.json")
    if summary.get("metadata", {}).get("review_recompute_state") == "pending":
        raise RenderCacheUnavailableError("Пересчёт проверки ещё не завершён")
    video = cache / _RENDER_CACHE_VIDEO
    temporary = cache / f".preview-{uuid.uuid4().hex}.mp4"
    destination = root / _WORKSPACE_PREVIEW_FILENAME
    try:
        _render_video(video, temporary,
            observations=_read_observation_cache(cache / _RENDER_CACHE_OBSERVATIONS),
            events=_events_from_json(root / "events.json"), config=replace(config, hud_mode="compact"),
            final_summary=summary, rounds_to_score=rounds,
            confirmed_knockdowns=_confirmed_knockdowns(config),
            progress_callback=progress_callback, cancel_callback=cancel_callback,
            tracking_only=True)
        finalize_h264_video(temporary, video, destination, cancel_callback=cancel_callback)
        summary.setdefault("metadata", {}).update(workspace_preview=_WORKSPACE_PREVIEW_FILENAME,
            workspace_preview_mode="tracking_only", export_stale=True, tracking_preview_stale=False,
            tracking_preview_rebuilt_at=datetime.now(UTC).isoformat())
        atomic_write_json(root / "summary.json", summary)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def rebuild_from_cache(
    run_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
    *,
    cancel_callback: CancelCallback | None = None,
) -> Path:
    """Re-render ``annotated.mp4`` from saved poses and reviewed events.

    No detector, pose backend or punch detector is invoked. Runs created before
    the render-cache contract fail with a user-facing instruction to analyse the
    source again instead of silently re-running ML.
    """

    started = time.perf_counter()
    target_run = Path(run_dir).expanduser().resolve()
    cache_dir = target_run / _RENDER_CACHE_DIRNAME
    normalized_video = cache_dir / _RENDER_CACHE_VIDEO
    observations_path = cache_dir / _RENDER_CACHE_OBSERVATIONS
    config_path = cache_dir / _RENDER_CACHE_CONFIG
    required = (normalized_video, observations_path, config_path)
    if not target_run.is_dir() or not all(path.is_file() for path in required):
        raise RenderCacheUnavailableError(
            "Для этого анализа нет render-cache. Запустите исходное видео заново."
        )

    _check_cancel(cancel_callback)
    _emit(progress_callback, 0.02, "Проверяем render-cache")
    cache_payload = _read_json_object(config_path, label="Render-cache")
    config, rounds_to_score = _config_from_render_cache(cache_payload)
    observations = _read_observation_cache(observations_path)
    events = _events_from_json(target_run / "events.json")
    summary = _read_json_object(target_run / "summary.json", label="summary.json")
    if summary.get("metadata", {}).get("review_recompute_state") == "pending":
        raise RenderCacheUnavailableError("Пересчёт проверки был прерван. Повторите проверку перед экспортом.")
    if summary.get("quality", {}).get("required_review_count", 0):
        raise ValueError("Перед финальным экспортом завершите обязательную проверку сцен и траекторий")
    confirmed_knockdowns = _confirmed_knockdowns(config)

    rebuild_id = uuid.uuid4().hex
    silent_output = cache_dir / f".rebuild-{rebuild_id}.silent.mp4"
    workspace_silent = cache_dir / f".rebuild-{rebuild_id}.workspace.silent.mp4"
    annotated_video = target_run / "annotated.mp4"
    workspace_video = target_run / _WORKSPACE_PREVIEW_FILENAME
    log_path = target_run / "analysis.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(UTC).isoformat()} {message}\n")

    def render_progress(value: float, description: str) -> None:
        normalized = min(1.0, max(0.0, (float(value) - 0.72) / 0.20))
        _emit(progress_callback, 0.05 + normalized * 0.80, description)

    def cancellation_probe() -> bool:
        _check_cancel(cancel_callback)
        return False

    log(f"render_rebuild_started observations={len(observations)} events={len(events)}")
    try:
        _render_video(
            normalized_video,
            silent_output,
            observations=observations,
            events=events,
            config=config,
            final_summary=summary,
            rounds_to_score=rounds_to_score,
            confirmed_knockdowns=confirmed_knockdowns,
            progress_callback=render_progress,
            cancel_callback=cancel_callback,
            workspace_silent_output=workspace_silent,
        )
        _check_cancel(cancel_callback)
        _emit(progress_callback, 0.88, "Кодируем обновлённый H.264")
        finalize_h264_video(
            silent_output,
            normalized_video,
            annotated_video,
            cancel_callback=cancellation_probe,
        )
        finalize_h264_video(
            workspace_silent, normalized_video, workspace_video,
            cancel_callback=cancellation_probe,
        )

        _emit(progress_callback, 0.95, "Обновляем превью таймлайна")
        try:
            manifest = _replace_hover_previews(
                workspace_video,
                target_run,
                cache_dir,
            )
            metadata = summary.get("metadata")
            if isinstance(metadata, dict):
                metadata["preview_manifest"] = manifest.name
        except Exception as exc:  # noqa: BLE001 - preserve the previous preview set
            # The previous preview set remains installed if rebuilding the new
            # one fails. It is still temporally aligned with the same video.
            log(f"preview_rebuild_failed={type(exc).__name__}: {exc}")

        metadata = summary.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["export_stale"] = False
            metadata["tracking_preview_stale"] = False
            metadata["workspace_preview"] = _WORKSPACE_PREVIEW_FILENAME
            metadata["workspace_preview_mode"] = "tracking_only"
            metadata["render_rebuilt_at"] = datetime.now(UTC).isoformat()
            metadata["last_render_processing_s"] = round(
                time.perf_counter() - started,
                3,
            )
        atomic_write_json(target_run / "summary.json", summary)
        log(
            f"render_rebuild_completed processing_s={time.perf_counter() - started:.3f}"
        )
        _emit(progress_callback, 1.0, "MP4 пересобран без повторного ML-анализа")
        return annotated_video
    except BaseException as exc:
        log(f"render_rebuild_failed={type(exc).__name__}: {exc}")
        raise
    finally:
        silent_output.unlink(missing_ok=True)
        workspace_silent.unlink(missing_ok=True)


__all__ = [
    "AnalysisCancelledError",
    "RenderCacheUnavailableError",
    "analyze_video",
    "rebuild_from_cache",
]
