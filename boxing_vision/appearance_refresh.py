"""Refresh part descriptors from original pixels without running either ML model.

Only a destination clone is written. Detection scores, boxes, poses, source IDs,
timestamps and scene states are evidence and are retained byte-for-value.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import cv2

from .artifacts import atomic_write_json
from .calibration import match_enrollment_box
from .pose import TwoFighterTracker, refresh_appearance_parts
from .tracking import TrackingFrame


def refresh_cached_appearance(
    source_cache: Path,
    destination_cache: Path,
    *,
    video: Path | None = None,
    progress: Callable[[float, str], None] = lambda *_: None,
    check_cancel: Callable[[], None] = lambda: None,
) -> dict:
    source_cache, destination_cache = (
        source_cache.resolve(),
        destination_cache.resolve(),
    )
    if (
        source_cache == destination_cache
        or destination_cache.is_relative_to(source_cache)
        or (
            source_cache.name == ".render_cache"
            and destination_cache.is_relative_to(source_cache.parent)
        )
    ):
        raise ValueError("Appearance refresh требует отдельный destination cache")
    targets = [
        destination_cache / name
        for name in (
            "detections.jsonl.gz",
            "identity_profile.json",
            "appearance_refresh.json",
        )
    ]
    if any(path.exists() for path in targets):
        raise ValueError(
            "Destination уже содержит appearance cache; используйте новый каталог"
        )
    config = json.loads((source_cache / "config.json").read_text())["config"]
    previous = json.loads((source_cache / "first_pass.json").read_text())
    profile = previous["identity_profile"]
    enrollment = config.get("enrollment_samples", [])
    if len(enrollment) != 3:
        raise ValueError("Нужны три сохранённых подтверждённых enrollment samples")
    with gzip.open(source_cache / "detections.jsonl.gz", "rt") as handle:
        frames = [TrackingFrame.from_dict(json.loads(line)) for line in handle]
    if not frames:
        raise ValueError("Пустой detector cache")
    chosen_indices = [
        min(
            range(len(frames)),
            key=lambda i: abs(frames[i].timestamp_ms - float(sample["time_s"]) * 1000),
        )
        for sample in enrollment
    ]
    if any(
        abs(frames[index].timestamp_ms - float(sample["time_s"]) * 1000) > 75
        for index, sample in zip(chosen_indices, enrollment)
    ):
        raise ValueError("Калибровочный кадр отсутствует в cache с допуском 75 мс")
    capture = cv2.VideoCapture(str(video or source_cache / "normalized.mp4"))
    if not capture.isOpened():
        raise ValueError("Исходное нормализованное видео недоступно")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError("Нужен нормализованный CFR источник с известным FPS")
    samples = {"fighter_a": [], "fighter_b": []}
    negatives = []
    enrollment_evidence = []
    next_video_index = 0
    destination_cache.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(targets[0], "wt", encoding="utf-8") as output:
            for index, frame in enumerate(frames):
                check_cancel()
                target = round(frame.timestamp_ms * fps / 1000)
                while next_video_index < target:
                    if not capture.grab():
                        raise ValueError("Исходное видео короче detector cache")
                    next_video_index += 1
                ok, image = capture.read()
                next_video_index += 1
                if not ok:
                    raise ValueError("Не удалось прочитать исходный кадр")
                frame.poses = refresh_appearance_parts(image, frame.poses)
                output.write(json.dumps(frame.to_dict(), ensure_ascii=False) + "\n")
                for sample_index, chosen_index in enumerate(chosen_indices):
                    if chosen_index != index:
                        continue
                    sample = enrollment[sample_index]
                    height, width = image.shape[:2]
                    selected = [
                        match_enrollment_box(frame.poses, sample[role], width, height)
                        for role in samples
                    ]
                    if selected[0] is selected[1]:
                        raise ValueError("Один detection выбран для A и B")
                    for role, pose in zip(samples, selected):
                        samples[role].append(pose)
                    negatives.extend(
                        pose
                        for pose in frame.poses
                        if all(pose is not selected_pose for selected_pose in selected)
                    )
                    enrollment_evidence.append(
                        {
                            "requested_ms": round(float(sample["time_s"]) * 1000),
                            "actual_ms": frame.timestamp_ms,
                            "source_ids": [pose.source_track_id for pose in selected],
                        }
                    )
                if index % 30 == 0:
                    progress(
                        index / len(frames),
                        "Обновляем признаки экипировки по исходным пикселям",
                    )
    finally:
        capture.release()
    policy = profile.get("policy", {})
    tracker = TwoFighterTracker(
        gallery_max_distance=policy.get("max_distance", 0.35),
        gallery_min_margin=policy.get("min_margin", 0.12),
        adaptive_identity_confidence_min=policy.get("adaptive_confidence_min", 0.9),
        adaptive_identity_margin_min=policy.get("adaptive_margin_min", 0.2),
    )
    tracker.enroll(samples, negatives)
    refreshed = tracker.export_identity_profile()
    # Preserve explicit user constraints and user gallery metadata. Adaptive
    # samples belong to the old descriptor space and are intentionally not copied.
    for key in profile:
        if key not in {
            "version",
            "policy",
            "core",
            "core_parts",
            "negative",
            "negative_parts",
            "adaptive",
            "adaptive_parts",
        }:
            refreshed[key] = deepcopy(profile[key])
    atomic_write_json(targets[1], refreshed)
    report = {
        "version": 1,
        "extractor": "pose-local-parts-v4",
        "frames": len(frames),
        "detector_rerun": False,
        "pose_rerun": False,
        "enrollment": enrollment_evidence,
        "source_cache": str(source_cache),
        "next_step": "Decode refreshed cache before rebuilding observations or MP4",
    }
    atomic_write_json(targets[2], report)
    progress(1.0, "Признаки обновлены; требуется повторное декодирование идентичности")
    return report
