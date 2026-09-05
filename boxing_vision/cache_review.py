"""Re-decode cached evidence after human review, without invoking ML."""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import tempfile
from bisect import bisect_left
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path

from .artifacts import atomic_write_json
from .contracts import IdentityState, PoseObservation, PunchEvent, SceneState
from .quality import apply_result_gate
from .tracking import OfflineIdentityDecoder, TrackingFrame


def _temporal_index(observations, frame_states):
    by_time = defaultdict(dict)
    for observation in observations:
        if observation.source_track_id is not None:
            key = (observation.shot_id, observation.source_track_id)
            if key in by_time[observation.timestamp_ms]:
                raise ValueError("Duplicate track observation within one timestamp")
            by_time[observation.timestamp_ms][key] = observation
    states = {int(row["timestamp_ms"]): row for row in frame_states}
    return by_time, states, sorted(states)


def _confirmed_at(timestamp, key, index):
    """Use adjacent detector frames, not adjacent surviving fighter observations.

    This distinction prevents one UNKNOWN/missing frame from being bridged by
    two observations that happen to share a track ID before and after it.
    """
    by_time, states, times = index
    position = bisect_left(times, timestamp)
    if position < len(times) and times[position] == timestamp:
        left = right = times[position]
    elif 0 < position < len(times):
        left, right = times[position - 1], times[position]
    else:
        return None
    if right - left > 150:
        return None
    for stamp in (left, right):
        state = states[stamp]
        if (
            state.get("scene_state") != "ACTIVE_FIGHT"
            or state.get("shot_id") != key[0]
            or state.get("is_scene_cut", False)
        ):
            return None
    before, after = by_time[left].get(key), by_time[right].get(key)
    if before is None or after is None or before.fighter_id != after.fighter_id:
        return None
    expected = {"fighter_a": "FIGHTER_A", "fighter_b": "FIGHTER_B"}.get(
        before.fighter_id
    )
    if expected is None:
        return None
    if any(
        obs.identity_state != expected
        or obs.scene_state != "ACTIVE_FIGHT"
        or obs.is_scene_cut
        or str(obs.review_status).upper() in {"NEEDS_REVIEW", "REJECTED"}
        or float(obs.identity_confidence or 0) < 0.55
        or float(obs.identity_margin or 0) < 0.12
        for obs in (before, after)
    ):
        return None
    return replace(
        before,
        timestamp_ms=timestamp,
        identity_confidence=min(before.identity_confidence, after.identity_confidence),
        identity_margin=min(before.identity_margin, after.identity_margin),
    )


def remap_dense_evidence(
    dense: list[PoseObservation],
    observations: list[PoseObservation],
    frame_states: list[dict],
):
    """Re-label existing dense poses only inside confirmed temporal membership."""
    index = _temporal_index(observations, frame_states)
    result = []
    for observation in dense:
        if observation.source_track_id is None:
            continue
        match = _confirmed_at(
            observation.timestamp_ms,
            (observation.shot_id, observation.source_track_id),
            index,
        )
        if match is not None:
            result.append(
                replace(
                    observation,
                    fighter_id=match.fighter_id,
                    identity_state=match.identity_state,
                    identity_confidence=match.identity_confidence,
                    identity_margin=match.identity_margin,
                    review_status=match.review_status,
                    scene_state=match.scene_state,
                    is_scene_cut=False,
                    segment_id=match.segment_id,
                    physical_track_id=match.physical_track_id,
                    identity_origin=match.identity_origin,
                )
            )
    return result


def _identity_context_unchanged(old_event, event, old_index, new_index):
    start = max(
        0,
        min(
            old_event.start_ms,
            event.start_ms,
            old_event.peak_ms - 400,
            event.peak_ms - 400,
        ),
    )
    end = max(
        old_event.end_ms, event.end_ms, old_event.peak_ms + 400, event.peak_ms + 400
    )
    stamps = sorted(
        {
            start,
            end,
            old_event.peak_ms,
            event.peak_ms,
            *(
                stamp
                for index in (old_index, new_index)
                for stamp in index[2]
                if start <= stamp <= end
            ),
        }
    )
    for stamp in stamps:
        for role in (event.attacker_id, event.defender_id):
            signatures = []
            for index in (old_index, new_index):
                position = bisect_left(index[2], stamp)
                if position >= len(index[2]):
                    return False
                frame = index[0][index[2][position]]
                candidates = [
                    key for key, obs in frame.items() if obs.fighter_id == role
                ]
                if len(candidates) != 1:
                    return False
                valid = _confirmed_at(stamp, candidates[0], index)
                if valid is None:
                    return False
                signatures.append(
                    (valid.shot_id, valid.source_track_id, valid.fighter_id)
                )
            if signatures[0] != signatures[1]:
                return False
    return True


def preserve_event_decisions(
    events: list[PunchEvent],
    old_events: list[PunchEvent],
    old_observations,
    observations,
    old_states,
    states,
):
    """One-to-one <=100 ms rematch; changed identity invalidates prior decisions."""
    old_index, new_index = (
        _temporal_index(old_observations, old_states),
        _temporal_index(observations, states),
    )
    pairs = sorted(
        (abs(old.peak_ms - event.peak_ms), old_number, number)
        for number, event in enumerate(events)
        for old_number, old in enumerate(old_events)
        if event.attacker_id == old.attacker_id
        and event.defender_id == old.defender_id
        and event.hand == old.hand
        and event.round == old.round
        and abs(old.peak_ms - event.peak_ms) <= 100
    )
    used_old, used_new = set(), set()
    for _, old_number, number in pairs:
        if old_number in used_old or number in used_new:
            continue
        old, event = old_events[old_number], events[number]
        if not _identity_context_unchanged(old, event, old_index, new_index):
            continue
        used_old.add(old_number)
        used_new.add(number)
        event.evidence = {**event.evidence, "cached_previous_event_id": old.event_id}
        if old.evidence.get("classification_model"):
            for key, value in old.evidence.items():
                if (
                    key.startswith(("classification_", "model_"))
                    or key == "geometry_technique"
                ):
                    event.evidence[key] = value
            event.technique = old.technique
            event.classification_confidence = old.classification_confidence
            event.model_version = old.model_version
            if str(old.review_status).lower() == "needs_review":
                event.review_status = old.review_status
            event.evidence["classification_reused_from_peak_ms"] = old.peak_ms
        if str(old.review_status).lower() in {
            "confirmed",
            "rejected",
            "deleted",
            "user_confirmed",
        }:
            event.review_status = old.review_status
            event.is_replay = old.is_replay
            event.evidence["review_preserved_same_identity_context"] = True
    return events


def _commit_staged(root: Path, staged: Path, paths: list[Path]):
    """Rollback normal write failures. Final summary is the commit marker.

    A process crash can interrupt multi-file replacements; a durable pending
    summary remains fail-closed and blocks export until review is retried.
    """
    backups = staged / "_previous"
    completed = []
    for relative in paths:
        source = root / relative
        if source.exists():
            destination = backups / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    try:
        for relative in paths:
            os.replace(staged / relative, root / relative)
            completed.append(relative)
    except BaseException:
        for relative in reversed(completed):
            previous = backups / relative
            if previous.exists():
                os.replace(previous, root / relative)
            else:
                (root / relative).unlink(missing_ok=True)
        raise


def redecode_from_cache(
    run_dir: str | Path,
    *,
    identity_overrides: dict[str, str] | None = None,
    scene_overrides: dict[str, str] | None = None,
    segment_identity_overrides: dict[str, str] | None = None,
    timing_mode: str | None = None,
    region_mode: str | None = None,
    _identity_review_record: dict | None = None,
) -> dict:
    # Lazy import avoids a cycle and reuses the exact scoring/cache contracts.
    from . import pipeline as p

    root = Path(run_dir).resolve()
    cache = root / p._RENDER_CACHE_DIRNAME
    payload = p._read_json_object(cache / p._RENDER_CACHE_CONFIG, label="Render-cache")
    config, rounds = p._config_from_render_cache(payload)
    if region_mode is not None:
        if region_mode not in {"none", "manual", "auto"}:
            raise ValueError("Неизвестный режим рабочей области")
        config = replace(config, region_mode=region_mode)
    if timing_mode is not None:
        if timing_mode not in {"continuous", "scheduled"}:
            raise ValueError("Неизвестный режим времени")
        config = replace(config, timing_mode=timing_mode)
        rounds = 1 if timing_mode == "continuous" else rounds
    first = p._read_json_object(cache / "first_pass.json", label="Кэш первого прохода")
    with gzip.open(cache / "detections.jsonl.gz", "rt", encoding="utf-8") as handle:
        frames = [
            TrackingFrame.from_dict(json.loads(line)) for line in handle if line.strip()
        ]
    frames.sort(key=lambda frame: frame.timestamp_ms)
    if not frames or len({frame.timestamp_ms for frame in frames}) != len(frames):
        raise ValueError("Кэш кадров пуст или содержит повторяющиеся timestamps")
    old_observations = list(
        p._read_observation_cache(cache / p._RENDER_CACHE_OBSERVATIONS)
    )
    old_events = p._events_from_json(root / "events.json")
    known_shots = {str(frame.shot_id) for frame in frames}
    known_tracks = {
        f"shot-{frame.shot_id}-track-{pose.source_track_id}"
        for frame in frames
        for pose in frame.poses
        if pose.source_track_id is not None
    }
    identities = dict(config.identity_overrides)
    scenes_override = {
        str(key).removeprefix("shot-"): str(value)
        for key, value in config.scene_overrides.items()
    }
    for key, value in (identity_overrides or {}).items():
        if key not in known_tracks or str(value) not in {
            state.value for state in IdentityState
        }:
            raise ValueError("Неизвестная траектория или недопустимая роль")
        identities[key] = str(value)
    for key, value in (scene_overrides or {}).items():
        key = str(key).removeprefix("shot-")
        if key not in known_shots or str(value) not in {
            state.value for state in SceneState
        }:
            raise ValueError("Неизвестная сцена или недопустимое состояние")
        scenes_override[key] = str(value)
    if any(
        key not in known_tracks
        or str(value) not in {state.value for state in IdentityState}
        for key, value in identities.items()
    ):
        raise ValueError("Сохранённые identity_overrides не соответствуют кэшу")
    if any(
        key not in known_shots
        or str(value) not in {state.value for state in SceneState}
        for key, value in scenes_override.items()
    ):
        raise ValueError("Сохранённые scene_overrides не соответствуют кэшу")
    # Reject simultaneous assignment of one person to two corners before writes.
    for frame in frames:
        roles = [
            identities.get(f"shot-{frame.shot_id}-track-{pose.source_track_id}")
            for pose in frame.poses
        ]
        for role in ("FIGHTER_A", "FIGHTER_B"):
            if roles.count(role) > 1:
                raise ValueError("В одном кадре два человека назначены одним бойцом")
    config = replace(
        config, identity_overrides=identities, scene_overrides=scenes_override,
        segment_identity_overrides={**getattr(config, "segment_identity_overrides", {}),
                                    **(segment_identity_overrides or {})},
    )
    config.validate()
    tracker = p._create_identity_tracker(config)
    profile_path = cache / "identity_profile.json"
    profile = p._read_json_object(profile_path, label="Профиль личности") if profile_path.is_file() else first["identity_profile"]
    if config.effective_region_mode == "none":
        profile = {**profile, "ring_rois": {}}
    tracker.import_identity_profile(profile)
    tracker.set_identity_overrides(identities)
    original_states = {
        int(row["timestamp_ms"]): row for row in first.get("frame_states", [])
    }
    observations, diagnostics, tracklets, scene_records, states, replays = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for shot_id, group in groupby(frames, key=lambda frame: frame.shot_id):
        shot = list(group)
        explicit = scenes_override.get(
            str(shot_id), scenes_override.get(f"shot-{shot_id}")
        )
        for frame in shot:
            previous_state = original_states.get(
                frame.timestamp_ms, {}
            ).get("scene_state", frame.scene_state)
            # A mode switch changes schedule-derived BREAK/NON_FIGHT only. It
            # must not accidentally unmark a replay or an uncertain scene.
            frame.scene_state = explicit or (
                "ACTIVE_FIGHT" if timing_mode == "continuous"
                and previous_state in {"ACTIVE_FIGHT", "BREAK", "NON_FIGHT"}
                else previous_state
            )
            states.append(
                {
                    "timestamp_ms": frame.timestamp_ms,
                    "shot_id": shot_id,
                    "is_scene_cut": frame.is_scene_cut,
                    "scene_state": str(frame.scene_state),
                }
            )
        decoder = OfflineIdentityDecoder(tracker,
            segment_identity_overrides=config.segment_identity_overrides)
        decoded = decoder.decode(shot)
        observations.extend(decoded)
        diagnostics.extend(decoder.diagnostics)
        tracklets.extend(decoder.tracklets)
        end_ms = round(shot[-1].timestamp_ms + 1000 / config.analysis_fps)
        shot_states = {str(frame.scene_state) for frame in shot}
        scene_state = explicit or (
            str(shot[0].scene_state) if len(shot_states) == 1 else "UNCERTAIN"
        )
        status = (
            "USER_CONFIRMED"
            if explicit
            else ("NEEDS_REVIEW" if scene_state == "UNCERTAIN" else "AUTO_CONFIRMED")
        )
        scene_records.append(
            {
                "shot_id": shot_id,
                "start_ms": shot[0].timestamp_ms,
                "end_ms": end_ms,
                "scene_state": scene_state,
                "review_status": status,
            }
        )
        for state, segment in groupby(shot, key=lambda frame: str(frame.scene_state)):
            segment = list(segment)
            if state == "REPLAY":
                replays.append(
                    (
                        segment[0].timestamp_ms,
                        round(segment[-1].timestamp_ms + 1000 / config.analysis_fps),
                    )
                )
    known_segments = {str(track.get("segment_id")) for track in tracklets if track.get("segment_id")}
    if any(key not in known_segments for key in config.segment_identity_overrides):
        raise ValueError("Неизвестный или устаревший сегмент: обновите кадр проверки")
    manual_at: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for row in diagnostics:
        segment = str(row.get("segment_id", ""))
        role = config.segment_identity_overrides.get(segment)
        if role in {"FIGHTER_A", "FIGHTER_B"}:
            manual_at[(int(row.get("shot_id", 0)), int(row["timestamp_ms"]), role)].add(segment)
    if any(len(segments) > 1 for segments in manual_at.values()):
        raise ValueError("В одном кадре два человека назначены одним бойцом")
    events = p.detect_punch_events(
        observations,
        replace(config, fight_start_s=0, fight_end_s=None),
        stances={
            "fighter_a": config.fighter_a_stance,
            "fighter_b": config.fighter_b_stance,
        },
        replay_intervals=replays,
    )
    # Review never constructs detector, pose, or punch neural models. Existing
    # classifier decisions are carried only across the same identity evidence.
    events = preserve_event_decisions(
        events,
        old_events,
        old_observations,
        observations,
        first.get("frame_states", []),
        states,
    )
    dense_path = cache / "dense_poses.jsonl.gz"
    dense = (
        remap_dense_evidence(
            list(p._read_observation_cache(dense_path)), observations, states
        )
        if dense_path.is_file()
        else None
    )
    old = p._read_json_object(root / "summary.json", label="summary.json")
    metadata = old.get("metadata", {})
    portraits = {
        role: info["portrait_filename"]
        for role, info in old.get("fighters", {}).items()
        if isinstance(info, dict) and info.get("portrait_filename")
    }
    summary = p.build_fight_summary(
        events,
        fighter_names={
            "fighter_a": config.fighter_a_name,
            "fighter_b": config.fighter_b_name,
        },
        scheduled_rounds=rounds,
        confirmed_knockdowns_suffered=p._confirmed_knockdowns(config),
    )
    summary = p._enrich_summary(
        summary,
        config=config,
        duration_s=float(metadata.get("duration_s", 0)),
        processing_s=float(metadata.get("processing_s", 0)),
        backend_name=str(metadata.get("backend", "cached")),
        observations=observations,
        events=events,
        portrait_filenames=portraits,
        frame_states=states,
        tracking_diagnostics=diagnostics,
    )
    review_items = [
        dict(scene, kind="scene", review_id=f"scene-{scene['shot_id']}", required=True)
        for scene in scene_records
        if scene["review_status"] == "NEEDS_REVIEW"
    ]
    review_items.extend(
        dict(track, kind="tracklet", required=True)
        for track in tracklets
        if str(track.get("review_status")) == "NEEDS_REVIEW"
        and track.get("eligible_for_review", True)
        and str(track.get("identity_state")) != "OTHER"
    )
    summary["metadata"].update(
        {
            key: value
            for key, value in metadata.items()
            if key in {"preview_manifest", "demo_portraits", "source_name"}
        }
    )
    summary["metadata"].update(
        export_stale=True,
        tracking_preview_stale=True,
        identity_reviewed_at=datetime.now(UTC).isoformat(),
        review_recompute="cached_evidence_no_ml",
        review_recompute_state="complete",
    )
    summary["quality"]["required_review_count"] = len(review_items)
    summary = apply_result_gate(summary)
    previous_review = (
        p._read_json_object(root / "review.json", label="review.json")
        if (root / "review.json").is_file()
        else {}
    )
    history = list(previous_review.get("event_review_history", []))
    identity_history = list(previous_review.get("identity_review_history", []))
    if _identity_review_record is not None:
        identity_history.append(_identity_review_record)
    carried = {event.evidence.get("cached_previous_event_id") for event in events}
    history.extend(
        {
            "event": event.to_dict(),
            "applied_to_current_candidates": event.event_id in carried,
        }
        for event in old_events
        if str(event.review_status).lower()
        in {"confirmed", "rejected", "deleted", "user_confirmed"}
    )
    # All data are prepared before changing existing files. The pending summary
    # is durable, while the complete summary is always the last replacement.
    old.setdefault("metadata", {})["export_stale"] = True
    old["metadata"]["tracking_preview_stale"] = True
    old["metadata"]["review_recompute_state"] = "pending"
    old.setdefault("quality", {})["winner_visible"] = False
    with tempfile.TemporaryDirectory(prefix=".review-stage-", dir=root) as directory:
        staged = Path(directory)
        staged_cache = staged / p._RENDER_CACHE_DIRNAME
        p._write_observation_cache(
            staged_cache / p._RENDER_CACHE_OBSERVATIONS, observations
        )
        from .tracking_artifacts import write_tracking_artifacts
        display_manifest = write_tracking_artifacts(staged_cache, frames, observations,
            diagnostics, events, replace(config, fight_start_s=0, fight_end_s=None), frame_states=states,
            source_cache=cache)
        summary["quality"].update(display_manifest)
        atomic_write_json(staged / "events.json", [event.to_dict() for event in events])
        p._write_jsonl(staged / "tracklets.jsonl", tracklets)
        p._write_jsonl(staged / "tracking_diagnostics.jsonl", diagnostics)
        atomic_write_json(staged / "scenes.json", scene_records)
        atomic_write_json(
            staged / "review.json",
            {
                "version": 2,
                "required_count": len(review_items),
                "items": review_items,
                "event_review_history": history,
                "identity_review_history": identity_history,
            },
        )
        atomic_write_json(
            staged_cache / p._RENDER_CACHE_CONFIG,
            {**payload, "version": p._RENDER_CACHE_VERSION, "rounds_to_score": rounds,
             "config": config.to_dict()},
        )
        atomic_write_json(
            staged_cache / "first_pass.json",
            {
                **first,
                "scenes": scene_records,
                "frame_states": states,
                "replay_intervals": replays,
                "identity_profile": tracker.export_identity_profile(),
            },
        )
        atomic_write_json(
            staged / "identity_profile.json", tracker.export_identity_profile()
        )
        if dense is not None:
            p._write_observation_cache(staged_cache / "dense_poses.jsonl.gz", dense)
        atomic_write_json(staged / "summary.json", summary)
        paths = sorted(
            path.relative_to(staged)
            for path in staged.rglob("*")
            if path.is_file() and path.name != "summary.json"
        )
        paths.append(Path("summary.json"))
        atomic_write_json(root / "summary.json", apply_result_gate(old))
        _commit_staged(root, staged, paths)
    return summary


def get_identity_correction_frame(run_dir: str | Path, timestamp_ms: float) -> dict:
    """Resolve clicks against cached real detector boxes, never display forecasts."""
    from . import pipeline as p
    from .tracking_artifacts import read_detection_frames

    timestamp = float(timestamp_ms)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Некорректное время кадра")
    root = Path(run_dir).resolve()
    frames = read_detection_frames(root / p._RENDER_CACHE_DIRNAME / "detections.jsonl.gz")
    if not frames:
        raise ValueError("Кэш детекций пуст")
    frame = min(frames, key=lambda item: abs(item.timestamp_ms - timestamp))
    if abs(frame.timestamp_ms - timestamp) > 100:
        raise ValueError("В выбранный момент нет реального кадра детектора")
    diagnostics = []
    diagnostic_path = root / "tracking_diagnostics.jsonl"
    if diagnostic_path.is_file():
        with diagnostic_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("timestamp_ms") == frame.timestamp_ms and row.get("shot_id", 0) == frame.shot_id:
                    diagnostics.append(row)
    indexed = {str(row.get("source_track_id")): row for row in diagnostics}
    metadata = p.probe_video(root / p._RENDER_CACHE_DIRNAME / p._RENDER_CACHE_VIDEO)
    candidates = []
    for pose in frame.poses:
        if pose.source_track_id is None:
            continue
        box = pose.bbox
        if (not all(math.isfinite(value) for value in (box.x1, box.y1, box.x2, box.y2))
                or box.width <= 0 or box.height <= 0
                or box.x2 <= 0 or box.y2 <= 0 or box.x1 >= metadata.width or box.y1 >= metadata.height):
            continue
        detail = indexed.get(str(pose.source_track_id), {})
        candidates.append({
            "source_track_id": pose.source_track_id,
            "segment_id": detail.get("segment_id"),
            "identity_state": detail.get("identity_state", "UNKNOWN"),
            "bbox": pose.bbox.to_dict(),
            "detector_confidence": pose.detector_confidence,
        })
    return {"timestamp_ms": frame.timestamp_ms, "shot_id": frame.shot_id,
            "width": metadata.width, "height": metadata.height,
            "candidates": candidates}


def correct_identity_at(run_dir: str | Path, *, timestamp_ms: float,
                        source_track_id: str | int, identity_state: str,
                        segment_id: str | None = None) -> dict:
    """One click anchors one deterministic segment, not a recycled source ID."""
    from . import pipeline as p
    from .tracking_artifacts import read_detection_frames

    if str(identity_state) not in {state.value for state in IdentityState}:
        raise ValueError("Недопустимая роль")
    root = Path(run_dir).resolve()
    info = get_identity_correction_frame(root, timestamp_ms)
    candidates = [item for item in info["candidates"] if str(item["source_track_id"]) == str(source_track_id)]
    if len(candidates) != 1:
        raise ValueError("Выбранный человек отсутствует в кадре детектора")
    candidate = candidates[0]
    cache = root / p._RENDER_CACHE_DIRNAME
    payload = p._read_json_object(cache / p._RENDER_CACHE_CONFIG, label="Render-cache")
    config, _ = p._config_from_render_cache(payload)
    # Legacy runs have no segment IDs. Resolve the actual shot under the new
    # decoder in memory, without changing either the run or the detector cache.
    if not candidate.get("segment_id"):
        first = p._read_json_object(cache / "first_pass.json", label="Кэш первого прохода")
        tracker = p._create_identity_tracker(config)
        profile_path = cache / "identity_profile.json"
        profile = p._read_json_object(profile_path, label="Профиль личности") if profile_path.is_file() else first["identity_profile"]
        if config.effective_region_mode == "none":
            profile = {**profile, "ring_rois": {}}
        tracker.import_identity_profile(profile)
        tracker.set_identity_overrides(config.identity_overrides)
        shot = [frame for frame in read_detection_frames(cache / "detections.jsonl.gz")
                if frame.shot_id == info["shot_id"]]
        decoder = OfflineIdentityDecoder(tracker, segment_identity_overrides=config.segment_identity_overrides)
        decoder.decode(shot)
        matches = [row for row in decoder.diagnostics if row["timestamp_ms"] == info["timestamp_ms"]
                   and str(row.get("source_track_id")) == str(source_track_id)]
        candidate["segment_id"] = matches[0].get("segment_id") if len(matches) == 1 else None
    resolved_segment = candidate.get("segment_id")
    if not resolved_segment or segment_id is not None and str(segment_id) != resolved_segment:
        raise ValueError("Сегмент изменился: обновите кадр и повторите выбор")
    return redecode_from_cache(root, segment_identity_overrides={resolved_segment: str(identity_state)},
        _identity_review_record={
            "timestamp_ms": info["timestamp_ms"], "shot_id": info["shot_id"],
            "source_track_id": source_track_id, "segment_id": resolved_segment,
            "identity_state": str(identity_state), "created_at": datetime.now(UTC).isoformat(),
        })
