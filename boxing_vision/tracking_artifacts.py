"""Versioned presentation/motion cache. No detector or pose model is loaded."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from .artifacts import atomic_write_json
from .contracts import BBox, DisplayTrack, Keypoint
from .events import retain_motion_proposals
from .tracking import TrackingFrame


def read_detection_frames(path: Path) -> list[TrackingFrame]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        frames = [TrackingFrame.from_dict(json.loads(line)) for line in handle if line.strip()]
    frames.sort(key=lambda frame: frame.timestamp_ms)
    if len({frame.timestamp_ms for frame in frames}) != len(frames):
        raise ValueError("Повторяющиеся timestamps в кэше детекций")
    return frames


def _write_gzip(path: Path, records: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_display_cache(path: Path) -> list[DisplayTrack]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        result = []
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            data["bbox"] = BBox(**data["bbox"])
            data["keypoints"] = {key: Keypoint(**point) for key, point in data.get("keypoints", {}).items()}
            result.append(DisplayTrack(**data))
        return result


def write_tracking_artifacts(cache: Path, frames, observations, diagnostics, events, config,
                             *, frame_states=(), source_cache: Path | None = None) -> dict:
    from .display_tracking import build_display_tracks

    states = {int(row["timestamp_ms"]): row for row in frame_states}
    frames = [replace(frame, scene_state=str(states.get(frame.timestamp_ms, {}).get("scene_state", frame.scene_state)))
              for frame in frames]
    source_cache = source_cache or cache
    video = source_cache / "normalized.mp4"
    dimensions = None
    if video.is_file():
        from .video import probe_video
        metadata = probe_video(video)
        dimensions = (metadata.width, metadata.height)
    display = build_display_tracks(frames, observations, diagnostics,
                                   prediction_ms=config.display_prediction_ms, frame_size=dimensions)
    proposals = retain_motion_proposals(frames, diagnostics, config, events=events)
    _write_gzip(cache / "display_tracks.jsonl.gz", (item.to_dict() for item in display))
    _write_gzip(cache / "punch_proposals.jsonl.gz", (item.to_dict() for item in proposals))
    fingerprints = hashlib.sha256(json.dumps(
        {"diagnostics": diagnostics, "frame_states": list(frame_states),
         "display_prediction_ms": config.display_prediction_ms},
        sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    manifest = {
        "version": 3, "coordinate_space": "normalized_video_pixels",
        "frame_size": list(dimensions) if dimensions else None,
        "analysis_fps": config.analysis_fps, "output_fps": config.output_fps,
        "prediction_limit_ms": config.display_prediction_ms,
        "identity_evidence_sha256": fingerprints,
        "display_observations": sum(str(item.display_state) == "OBSERVED" for item in display),
        "display_predictions": sum(str(item.display_state) == "PREDICTED" for item in display),
        "motion_proposals": len(proposals),
        "unresolved_motion_proposals": sum(item.status != "resolved" for item in proposals),
        "ground_truth_visibility_measured": False,
    }
    detections = source_cache / "detections.jsonl.gz"
    if detections.is_file():
        with detections.open("rb") as handle:
            manifest["detector_cache_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    atomic_write_json(cache / "display_manifest.json", manifest)
    return manifest
