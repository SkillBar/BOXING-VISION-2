from __future__ import annotations

import math
import shutil
import time
from collections import deque
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
)
from .config import DEFAULT_RUNS_DIR, AnalysisConfig
from .contracts import AnalysisResult, PoseObservation, PunchEvent
from .events import detect_punch_events
from .pose import TwoFighterTracker, create_pose_backend
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


class AnalysisCancelledError(RuntimeError):
    """Raised when the local user cancels the only active analysis job."""


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
                artifacts.events_path.unlink(missing_ok=True)
                artifacts.summary_path.unlink(missing_ok=True)
                (artifacts.run_dir / "boxing-vision-result.zip").unlink(missing_ok=True)
                shutil.rmtree(artifacts.clips_dir, ignore_errors=True)
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


def _important_events(events: Iterable[PunchEvent], limit: int = 24) -> list[PunchEvent]:
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


def _enrich_summary(
    summary: dict[str, object],
    *,
    config: AnalysisConfig,
    duration_s: float,
    processing_s: float,
    backend_name: str,
    observations: list[PoseObservation],
    events: list[PunchEvent],
) -> dict[str, object]:
    fighters = summary.get("fighters")
    if isinstance(fighters, dict):
        for fighter_id, corner in (("fighter_a", "red"), ("fighter_b", "blue")):
            fighter = fighters.get(fighter_id)
            if isinstance(fighter, dict):
                fighter["id"] = fighter_id
                fighter["corner"] = corner
                # The UI accepts a nested stats object while the renderer uses
                # the same values directly. Keeping both makes the JSON useful
                # outside Gradio without changing the render contract.
                fighter["stats"] = {
                    key: value
                    for key, value in fighter.items()
                    if key not in {"id", "name", "corner", "stats"}
                }

    winner_id = summary.get("winner_id")
    winner_name = str(summary.get("winner_name") or "Ничья")
    confidence = float(summary.get("confidence") or 0.0)
    summary["winner"] = {
        "fighter_id": winner_id,
        "name": winner_name,
        "confidence": round(confidence, 4),
        "label": "Экспериментальный прогноз" if winner_id else "Экспериментальная ничья",
    }
    tracking_confidences = [observation.track_confidence for observation in observations]
    event_confidences = [event.confidence for event in events if not event.is_replay]
    summary["metadata"] = {
        "duration_s": round(duration_s, 3),
        "processing_s": round(processing_s, 3),
        "backend": backend_name,
        "analysis_fps": config.analysis_fps,
        "output_fps": config.output_fps,
        "generated_at": datetime.now(UTC).isoformat(),
        "confirmed_knockdowns_suffered": _confirmed_knockdowns(config),
        "disclaimer": "Неофициальная экспериментальная AI-оценка; не является судейским решением.",
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
        "possible_knockdowns": sum(
            bool(event.evidence.get("possible_knockdown")) for event in events
        ),
    }
    return summary


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
) -> None:
    metadata = probe_video(normalized_video)
    timestamps = sorted({observation.timestamp_ms for observation in observations})
    by_timestamp: dict[int, list[PoseObservation]] = {}
    for observation in observations:
        by_timestamp.setdefault(observation.timestamp_ms, []).append(observation)

    fighter_names = {
        "fighter_a": config.fighter_a_name,
        "fighter_b": config.fighter_b_name,
    }
    renderer = FrameRenderer(
        fighter_names,
        pose_threshold=config.pose_score_threshold,
        trail_length=10,
        hud_width_ratio=0.29,
    )

    writer: cv2.VideoWriter | None = None
    observation_index = 0
    event_cursor = 0
    scored_cursor = 0
    active_events: deque[PunchEvent] = deque()
    seen_events: list[PunchEvent] = []
    ordered_events = sorted(events, key=lambda event: event.start_ms)
    events_by_peak = sorted(events, key=lambda event: event.peak_ms)
    running_summary = build_fight_summary(
        [],
        fighter_names=fighter_names,
        scheduled_rounds=rounds_to_score,
        confirmed_knockdowns_suffered=confirmed_knockdowns,
    )
    running_scores = score_rounds(
        [],
        scheduled_rounds=rounds_to_score,
        confirmed_knockdowns_suffered=confirmed_knockdowns,
    )
    rendered_frames = 0

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

            timestamp_ms = video_frame.timestamp_ms
            while (
                observation_index + 1 < len(timestamps)
                and timestamps[observation_index + 1] <= timestamp_ms
            ):
                observation_index += 1
            current_observations: list[PoseObservation] = []
            if timestamps:
                observation_timestamp = timestamps[observation_index]
                if abs(timestamp_ms - observation_timestamp) <= 180:
                    current_observations = by_timestamp[observation_timestamp]

            while event_cursor < len(ordered_events) and ordered_events[event_cursor].start_ms - 150 <= timestamp_ms:
                active_events.append(ordered_events[event_cursor])
                event_cursor += 1
            while active_events and active_events[0].end_ms + 450 < timestamp_ms:
                active_events.popleft()

            changed = False
            while scored_cursor < len(events_by_peak) and events_by_peak[scored_cursor].peak_ms <= timestamp_ms:
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
                running_scores = score_rounds(
                    seen_events,
                    scheduled_rounds=rounds_to_score,
                    confirmed_knockdowns_suffered=confirmed_knockdowns,
                )

            rendered = renderer.draw(
                frame,
                current_observations,
                list(active_events),
                running_summary,
                running_scores,
                timestamp_ms,
            )
            writer.write(rendered)
            rendered_frames += 1
            if rendered_frames % max(1, round(metadata.fps)) == 0:
                _emit(
                    progress_callback,
                    0.72 + 0.20 * min(1.0, timestamp_ms / max(1.0, metadata.duration_s * 1000.0)),
                    "Формируем размеченную трансляцию",
                )
    finally:
        if writer is not None:
            writer.release()
    if rendered_frames == 0 or not silent_output.is_file():
        raise VideoValidationError("В нормализованном видео нет декодируемых кадров")


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
    normalized_video = artifacts.work_dir / "normalized.mp4"

    def log(message: str) -> None:
        with artifacts.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(UTC).isoformat()} {message}\n")

    def report(fraction: float, description: str) -> None:
        _check_cancel(cancel_callback)
        _emit(progress_callback, fraction, description)

    def cancellation_probe() -> bool:
        _check_cancel(cancel_callback)
        return False

    log(f"job={artifacts.job_id} source_name={source.name}")
    report(0.01, "Проверяем видео")
    source_metadata = validate_video(source, max_duration_s=config.max_duration_s)
    input_metadata = source_metadata.to_dict()
    input_metadata["path"] = source.name
    log(f"input={input_metadata}")

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
    )
    tracker = TwoFighterTracker(("fighter_a", "fighter_b"), max_missing_frames=12)
    cut_detector = SceneCutDetector()
    replay_detector = ReplayDetector()
    observations: list[PoseObservation] = []
    replay_intervals: list[tuple[int, int]] = []
    replay_start: int | None = None
    analysed_frames = 0
    anchors_initialized = False

    for video_frame in iter_video_frames(normalized_video, target_fps=config.analysis_fps):
        _check_cancel(cancel_callback)
        if not anchors_initialized:
            if config.fighter_a_anchor is not None and config.fighter_b_anchor is not None:
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
            replay_intervals.append((replay_start, max(replay_start, video_frame.timestamp_ms - 1)))
            replay_start = None
        possible_replay = replay_detector.update(
            video_frame.image,
            video_frame.timestamp_ms,
            is_scene_cut=scene_cut,
        )
        if possible_replay:
            replay_start = video_frame.timestamp_ms
        if scene_cut:
            backend.reset()
            log(f"scene_cut timestamp_ms={video_frame.timestamp_ms} score={cut_score:.3f}")

        poses = backend.infer(video_frame.image)
        frame_observations = tracker.process(
            video_frame.frame_index,
            video_frame.timestamp_ms,
            poses,
            scene_cut=scene_cut,
        )
        observations.extend(frame_observations)
        assigned_fighters = {observation.fighter_id for observation in frame_observations}
        if assigned_fighters == {"fighter_a", "fighter_b"} and tracker.anchors:
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
                    video_frame.timestamp_ms / max(1.0, normalized_metadata.duration_s * 1000.0),
                ),
                "Отслеживаем бойцов и движения рук",
            )

    if replay_start is not None:
        replay_intervals.append((replay_start, round(normalized_metadata.duration_s * 1000)))
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
    cycle_s = max(1, config.round_length_s + config.rest_length_s)
    rounds_to_score = max(
        1,
        min(config.scheduled_rounds, math.ceil(normalized_metadata.duration_s / cycle_s)),
    )
    fighter_names = {
        "fighter_a": config.fighter_a_name,
        "fighter_b": config.fighter_b_name,
    }
    confirmed_knockdowns = _confirmed_knockdowns(config)
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
    )
    log(f"events={len(events)} rounds={rounds_to_score}")

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
    )
    report(0.93, "Кодируем итоговый H.264 и возвращаем звук")
    finalize_h264_video(
        artifacts.silent_video,
        normalized_video,
        artifacts.annotated_video,
        cancel_callback=cancellation_probe,
    )

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


__all__ = ["AnalysisCancelledError", "analyze_video"]
