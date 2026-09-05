"""Cache all people first, then decode global identities per continuous shot."""
from __future__ import annotations

import gzip
import json
from bisect import bisect_right
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2

from .artifacts import atomic_write_json
from .calibration import match_enrollment_box
from .config import AnalysisConfig
from .contracts import IdentityState, PoseObservation, ReviewStatus, SceneState
from .pose import TwoFighterTracker, refresh_appearance_parts
from .preflight import build_preflight_report
from .scenes import detect_shot_boundaries
from .video import ReplayDetector, iter_video_frames, probe_video


class EnrollmentRequiredError(ValueError):
    pass


@dataclass
class FirstPassResult:
    observations: list[PoseObservation]
    replay_intervals: list[tuple[int, int]]
    scenes: list[dict[str, Any]]
    tracklets: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    identity_profile: dict[str, Any]
    frame_states: list[dict[str, Any]]
    preflight: dict[str, Any]


def enroll_video(video: Path, backend: Any, tracker: TwoFighterTracker, config: AnalysisConfig) -> dict[str, Any]:
    """Recompute enrollment evidence on normalized pixels, not UI thumbnails."""
    samples: dict[str, list] = {"fighter_a": [], "fighter_b": []}
    if not config.enrollment_confirmed:
        raise EnrollmentRequiredError("Автоматическое предложение ещё не подтверждено. Проверьте три кадра и нажмите «Подтвердить».")
    negatives = []
    capture = cv2.VideoCapture(str(video))
    try:
        for sample in config.enrollment_samples:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(sample["time_s"]) * 1000)
            ok, image = capture.read()
            if not ok:
                raise EnrollmentRequiredError("Калибровочный кадр выходит за пределы видео")
            poses = backend.infer(image)
            h, w = image.shape[:2]
            chosen = [match_enrollment_box(poses, sample[role], w, h) for role in samples]
            if chosen[0] is chosen[1]:
                raise EnrollmentRequiredError("Один detection выбран для обоих бойцов")
            for role, pose in zip(samples, chosen):
                samples[role].append(pose)
            negatives.extend(pose for pose in poses if all(pose is not selected for selected in chosen))
        if config.enrollment_samples:
            tracker.enroll(samples, negatives)
        elif config.enrollment_mode != "legacy_anchor":
            raise EnrollmentRequiredError("Подтвердите три калибровочных кадра перед анализом")
    finally:
        capture.release()
    tracker.set_identity_overrides(config.identity_overrides)
    return tracker.export_identity_profile()


def _scheduled(timestamp_ms: int, shot: int, config: AnalysisConfig) -> str:
    override = config.scene_overrides.get(str(shot), config.scene_overrides.get(f"shot-{shot}"))
    if override:
        return str(override).upper()
    if getattr(config, "timing_mode", "scheduled") == "continuous":
        return SceneState.ACTIVE_FIGHT.value
    elapsed = timestamp_ms / 1000
    cycle = config.round_length_s + config.rest_length_s
    if elapsed >= cycle * config.scheduled_rounds:
        return SceneState.NON_FIGHT.value
    return SceneState.BREAK.value if elapsed % cycle >= config.round_length_s else SceneState.ACTIVE_FIGHT.value


def run_first_pass(video: Path, cache_dir: Path, backend: Any, tracker: TwoFighterTracker,
                   config: AnalysisConfig, *, progress: Callable[[float, str], None],
                   cancelled: Callable[[], bool] | None = None,
                   check_cancel: Callable[[], None] = lambda: None) -> FirstPassResult:
    from .detection_rescue import RescueController
    from .tracking import OfflineIdentityDecoder, ShotLocalBoTSORT, TrackingFrame

    enroll_video(video, backend, tracker, config)
    metadata = probe_video(video)
    boundaries = detect_shot_boundaries(video, fps=config.analysis_fps, cancelled=cancelled,
                                       max_duration_ms=10200)
    full_boundaries_ready = metadata.duration_s <= 10.2
    progress(.19, "Первый проход: люди и локальные траектории")
    local = ShotLocalBoTSORT(frame_rate=config.analysis_fps)
    rescue = RescueController()
    replay = ReplayDetector()
    all_observations: list[PoseObservation] = []
    diagnostics, tracklets, scenes, frame_states, replay_intervals = [], [], [], [], []
    buffer: list[TrackingFrame] = []
    replay_shots: set[int] = set()
    preflight = {"status": "pending", "window_ms": min(10000, round(metadata.duration_s * 1000)),
                 "pair_coverage": 0.0, "reason": None}
    cache_path = cache_dir / "detections.jsonl.gz"

    def check_preflight() -> None:
        prefix = [frame for frame in buffer if frame.timestamp_ms < preflight["window_ms"]]
        preview_decoder = OfflineIdentityDecoder(deepcopy(tracker),
            segment_identity_overrides=getattr(config, "segment_identity_overrides", {}))
        # Replay evidence can arrive before the open shot has been flushed.
        # Apply the same exclusion now, while retaining its original schedule
        # in the denominator just as for completed shots.
        preview_frames = [replace(frame, scene_state=SceneState.REPLAY.value)
                          if frame.shot_id in replay_shots else frame for frame in prefix]
        preview = preview_decoder.decode(preview_frames) if prefix else []
        observed = [obs for obs in all_observations + preview if obs.timestamp_ms < preflight["window_ms"]]
        expected = [row for row in frame_states if row["timestamp_ms"] < preflight["window_ms"]]
        expected += [{"timestamp_ms": frame.timestamp_ms, "shot_id": frame.shot_id,
                      "is_scene_cut": frame.is_scene_cut,
                      "scheduled_scene_state": frame.scene_state,
                      "scene_state": preview_frame.scene_state} for frame, preview_frame in zip(prefix, preview_frames)]
        detail = [row for row in diagnostics + preview_decoder.diagnostics if row["timestamp_ms"] < preflight["window_ms"]]
        preflight.update(build_preflight_report(observed, expected, detail,
                         window_ms=preflight["window_ms"], minimum_margin=config.identity_margin_min))
        preflight["identity_decoder_version"] = getattr(preview_decoder, "version", "unknown")
        atomic_write_json(cache_dir / "preflight_diagnostics.json", detail)
        atomic_write_json(cache_dir / "preflight.json", preflight)
        if preflight["status"] != "passed":
            atomic_write_json(cache_dir / "identity_profile.json", tracker.export_identity_profile())
        # A short occlusion or a non-fight opening is not an invalid upload.
        # Continue detection and offline reacquisition; identity and event gates
        # still reject uncertain intervals. Keep diagnostics, never fabricate IDs.

    def decode_buffer(end_ms: int) -> None:
        if not buffer:
            return
        shot = buffer[0].shot_id
        scheduled_states = [str(frame.scene_state) for frame in buffer]
        if shot in replay_shots:
            for frame in buffer:
                frame.scene_state = SceneState.REPLAY.value
            replay_intervals.append((buffer[0].timestamp_ms, end_ms))
        decoder = OfflineIdentityDecoder(tracker,
            segment_identity_overrides=getattr(config, "segment_identity_overrides", {}))
        decoded = decoder.decode(buffer)
        states = [frame.scene_state for frame in buffer]
        # Keep schedule boundaries frame-level. Missing identities do not turn a
        # real fight into a non-fight and cannot improve the coverage denominator.
        proposed = max(set(states), key=states.count)
        pair_frames: dict[int, set[str]] = {}
        for observation in decoded:
            if observation.identity_state in {IdentityState.FIGHTER_A, IdentityState.FIGHTER_B}:
                pair_frames.setdefault(observation.timestamp_ms, set()).add(observation.fighter_id)
        pair_count = sum(len(ids) == 2 for ids in pair_frames.values())
        explicit = str(shot) in config.scene_overrides or f"shot-{shot}" in config.scene_overrides
        uncertain_scene = proposed == "ACTIVE_FIGHT" and not explicit and pair_count < 3
        if uncertain_scene:
            decoded = [replace(obs, scene_state=SceneState.UNCERTAIN, review_status=ReviewStatus.NEEDS_REVIEW) for obs in decoded]
        scenes.append({"shot_id": shot, "start_ms": buffer[0].timestamp_ms, "end_ms": end_ms,
                       "scene_state": "UNCERTAIN" if uncertain_scene else proposed,
                       "review_status": "NEEDS_REVIEW" if uncertain_scene else "AUTO_CONFIRMED",
                       "reason": "insufficient_pair_evidence" if uncertain_scene else None})
        frame_states.extend({"timestamp_ms": frame.timestamp_ms, "shot_id": shot,
                             "is_scene_cut": frame.is_scene_cut,
                             "scheduled_scene_state": scheduled,
                             "scene_state": "UNCERTAIN" if uncertain_scene else str(frame.scene_state)}
                            for frame, scheduled in zip(buffer, scheduled_states))
        all_observations.extend(decoded)
        diagnostics.extend(decoder.diagnostics)
        tracklets.extend(decoder.tracklets)
        buffer.clear()

    # Streaming compressed cache includes OTHER/unassigned people and evidence.
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        previous_shot = -1
        for frame in iter_video_frames(video, target_fps=config.analysis_fps):
            check_cancel()
            if frame.timestamp_ms >= preflight["window_ms"] and preflight["status"] == "pending":
                check_preflight()
                progress(.23, "Продолжаем анализ; при потере бойца подсчёт его ударов приостановлен"
                         if preflight["status"] != "passed" else
                         "Короткая проверка завершена; продолжаем полный анализ")
                if not full_boundaries_ready:
                    boundaries = detect_shot_boundaries(video, fps=config.analysis_fps, cancelled=cancelled)
                    full_boundaries_ready = True
            shot = bisect_right(boundaries, frame.timestamp_ms) - 1
            cut = shot != previous_shot
            if cut:
                decode_buffer(frame.timestamp_ms)
                backend.reset()
                previous_shot = shot
            h, w = frame.image.shape[:2]
            if shot == 0 and config.ring_rois and getattr(config, "effective_region_mode", "manual") == "manual":
                tracker.set_ring_roi(config.ring_rois, frame_size=(w, h), shot_id=0)
            if config.enrollment_mode == "legacy_anchor" and frame.timestamp_ms == 0 and config.fighter_a_anchor and config.fighter_b_anchor:
                tracker.set_anchors(tuple(v * s for v, s in zip(config.fighter_a_anchor, (w, h))),
                                    tuple(v * s for v, s in zip(config.fighter_b_anchor, (w, h))))
            if replay.update(frame.image, frame.timestamp_ms, is_scene_cut=cut and shot > 0):
                replay_shots.add(shot)
            measured = backend.infer(frame.image)
            additional = rescue.maybe_infer(frame.image, measured, local.predictions,
                                            backend, frame.timestamp_ms, shot)
            if additional:
                measured = refresh_appearance_parts(frame.image, [*measured, *additional])
            poses = local.update(measured, frame.image, frame.timestamp_ms, shot)
            record = TrackingFrame(frame.frame_index, frame.timestamp_ms, shot, poses,
                                   _scheduled(frame.timestamp_ms, shot, config), cut)
            record.tracker_predictions = list(local.predictions)
            handle.write(json.dumps({**record.to_dict(), "rescue_diagnostics": rescue.last_diagnostics},
                                   ensure_ascii=False, allow_nan=False) + "\n")
            buffer.append(record)
            if len(buffer) % max(1, round(config.analysis_fps)) == 0:
                progress(.19 + .40 * frame.timestamp_ms / max(1, metadata.duration_s * 1000),
                         "Первый проход: люди, экипировка и траектории")
    if preflight["status"] == "pending":
        check_preflight()
    decode_buffer(round(metadata.duration_s * 1000))
    atomic_write_json(cache_dir / "first_pass.json", {"version": 1, "scenes": scenes,
                      "frame_states": frame_states, "identity_profile": tracker.export_identity_profile(),
                      "preflight": preflight, "replay_intervals": replay_intervals})
    atomic_write_json(cache_dir / "rescue_summary.json", rescue.summary)
    return FirstPassResult(all_observations, replay_intervals, scenes, tracklets, diagnostics,
                           tracker.export_identity_profile(), frame_states, preflight)
