from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from boxing_vision import cache_review
from boxing_vision.artifacts import atomic_write_json
from boxing_vision.config import AnalysisConfig
from boxing_vision.contracts import (
    BBox,
    IdentityState,
    Keypoint,
    PoseObservation,
    PunchEvent,
)
from boxing_vision.pose import RawPose, TwoFighterTracker
from boxing_vision.tracking import OfflineIdentityDecoder, TrackingFrame


def observation(stamp, role="fighter_a", source=1, **changes):
    base = PoseObservation(
        stamp // 100,
        stamp,
        role,
        BBox(0, 0, 100, 200),
        {},
        source_track_id=source,
        shot_id=0,
        identity_confidence=0.95,
        identity_margin=0.3,
    )
    return replace(base, **changes)


def states(timestamps, **changes):
    return [
        dict(
            timestamp_ms=stamp,
            shot_id=0,
            scene_state="ACTIVE_FIGHT",
            is_scene_cut=False,
            **changes,
        )
        for stamp in timestamps
    ]


def punch(identifier="new", peak=1000, **changes):
    event = PunchEvent(
        identifier,
        1,
        peak - 100,
        peak,
        peak + 100,
        "fighter_a",
        "fighter_b",
        "left",
        "hook",
        "body",
        "unclear",
        0.7,
        50,
    )
    return replace(event, **changes)


def test_dense_relabel_uses_temporal_membership_not_last_role():
    observations = [
        observation(0),
        observation(100, identity_state=IdentityState.UNKNOWN),
        observation(200),
    ]
    dense = [observation(stamp) for stamp in (0, 33, 67, 100, 133, 167, 200)]
    mapped = cache_review.remap_dense_evidence(
        dense, observations, states((0, 100, 200))
    )
    assert [obs.timestamp_ms for obs in mapped] == [0, 200]


@pytest.mark.parametrize("kind", ["missing", "cut", "break", "track", "role", "shot"])
def test_dense_never_bridges_unknown_cut_break_or_track_change(kind):
    observations = [observation(0), observation(100)]
    frame_states = states((0, 100))
    if kind == "missing":
        observations.pop()
    elif kind == "cut":
        frame_states[1]["is_scene_cut"] = True
    elif kind == "break":
        frame_states[1]["scene_state"] = "BREAK"
    elif kind == "track":
        observations[1].source_track_id = 9
    elif kind == "role":
        observations[1] = observation(100, "fighter_b")
    else:
        frame_states[1]["shot_id"] = 1
        observations[1].shot_id = 1
    assert not cache_review.remap_dense_evidence(
        [observation(50)], observations, frame_states
    )


def identity_sequence():
    return [
        observation(stamp, role, source)
        for stamp in range(0, 2100, 100)
        for role, source in (("fighter_a", 1), ("fighter_b", 2))
    ]


@pytest.mark.parametrize("status", ["confirmed", "rejected"])
def test_review_survives_only_one_to_one_same_identity_rematch(status):
    original = identity_sequence()
    frame_states = states(range(0, 2100, 100))
    old = punch(
        "old",
        review_status=status,
        technique="unknown",
        model_version="acm-v1",
        evidence={
            "classification_model": "acm40960-lstm-v1",
            "model_score_uncalibrated": 0.9,
        },
    )
    result = cache_review.preserve_event_decisions(
        [punch(), punch("duplicate", 1030)],
        [old],
        original,
        original,
        frame_states,
        frame_states,
    )
    assert result[0].review_status == status
    assert result[0].technique == "unknown" and result[0].model_version == "acm-v1"
    assert result[1].review_status == "unreviewed"
    assert result[0].evidence["review_preserved_same_identity_context"]


@pytest.mark.parametrize("kind", ["unknown", "other_hand", "time", "source", "scene"])
def test_changed_context_never_inherits_human_confirmation(kind):
    original = identity_sequence()
    updated = [replace(obs) for obs in original]
    old_states = states(range(0, 2100, 100))
    new_states = states(range(0, 2100, 100))
    event = punch()
    if kind == "unknown":
        updated[20].identity_state = "UNKNOWN"
    elif kind == "other_hand":
        event.hand = "right"
    elif kind == "time":
        event.peak_ms += 101
    elif kind == "source":
        for obs in updated:
            if obs.fighter_id == "fighter_a":
                obs.source_track_id = 99
    else:
        new_states[10]["scene_state"] = "BREAK"
    result = cache_review.preserve_event_decisions(
        [event],
        [punch("old", review_status="confirmed")],
        original,
        updated,
        old_states,
        new_states,
    )
    assert result[0].review_status == "unreviewed"


def test_partial_commit_rolls_back_previous_artifacts(tmp_path, monkeypatch):
    root, stage = tmp_path / "run", tmp_path / "stage"
    for folder in (root, stage):
        folder.mkdir()
    for name in ("events.json", "summary.json"):
        atomic_write_json(root / name, {"generation": "old"})
        atomic_write_json(stage / name, {"generation": "new"})
    original = cache_review.os.replace
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated disk error")
        original(source, destination)

    monkeypatch.setattr(cache_review.os, "replace", fail_second)
    with pytest.raises(OSError):
        cache_review._commit_staged(
            root, stage, [Path("events.json"), Path("summary.json")]
        )
    assert json.loads((root / "events.json").read_text())["generation"] == "old"
    assert json.loads((root / "summary.json").read_text())["generation"] == "old"


@pytest.fixture
def cached_run(tmp_path, monkeypatch):
    from boxing_vision import pipeline as p

    cache = tmp_path / p._RENDER_CACHE_DIRNAME
    cache.mkdir()
    config = AnalysisConfig(punch_model="auto", scheduled_rounds=1)
    frame_states = states(range(0, 2100, 100))
    observations = identity_sequence()
    atomic_write_json(
        cache / p._RENDER_CACHE_CONFIG,
        {"version": p._RENDER_CACHE_VERSION, "config": config.to_dict()},
    )
    atomic_write_json(
        cache / "first_pass.json",
        {"identity_profile": {}, "frame_states": frame_states},
    )
    atomic_write_json(
        tmp_path / "summary.json",
        {
            "metadata": {"duration_s": 2.1, "export_stale": False},
            "quality": {"winner_visible": False},
        },
    )
    atomic_write_json(
        tmp_path / "events.json", [punch("old", review_status="confirmed").to_dict()]
    )
    p._write_observation_cache(cache / p._RENDER_CACHE_OBSERVATIONS, observations)
    p._write_observation_cache(cache / "dense_poses.jsonl.gz", observations)
    with gzip.open(cache / "detections.jsonl.gz", "wt") as output:
        for row in frame_states:
            frame = TrackingFrame(
                row["timestamp_ms"] // 100,
                row["timestamp_ms"],
                0,
                [RawPose(BBox(0, 0, 100, 200), {}, source_track_id=i) for i in (1, 2)],
            )
            output.write(json.dumps(frame.to_dict()) + "\n")

    class FakeTracker:
        def import_identity_profile(self, value):
            pass

        def set_identity_overrides(self, value):
            pass

        def export_identity_profile(self):
            return {}

    class FakeDecoder:
        def __init__(self, tracker, *, segment_identity_overrides=None):
            self.segment_identity_overrides = dict(segment_identity_overrides or {})
            self.diagnostics = []
            self.tracklets = []

        def decode(self, frames):
            return observations

    def forbidden(*args, **kwargs):
        raise AssertionError("ML invoked in cache-only review")

    monkeypatch.setattr(p, "_create_identity_tracker", lambda config: FakeTracker())
    monkeypatch.setattr(cache_review, "OfflineIdentityDecoder", FakeDecoder)
    monkeypatch.setattr(p, "detect_punch_events", lambda *args, **kwargs: [punch()])
    from boxing_vision.pose import RTMLibPoseBackend
    from boxing_vision.punch_models import AcmPunchClassifier

    monkeypatch.setattr(AcmPunchClassifier, "__init__", forbidden)
    monkeypatch.setattr(RTMLibPoseBackend, "__init__", forbidden)
    return tmp_path


def test_cache_review_runs_without_neural_inference_and_preserves_review(cached_run):
    result = cache_review.redecode_from_cache(cached_run)
    assert result["metadata"]["export_stale"] is True
    assert result["metadata"]["review_recompute_state"] == "complete"
    assert result["metadata"]["review_recompute"] == "cached_evidence_no_ml"
    assert (
        json.loads((cached_run / "events.json").read_text())[0]["review_status"]
        == "confirmed"
    )


def test_conflicting_identity_overrides_rejected_before_any_writes(cached_run):
    summary = (cached_run / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="два человека"):
        cache_review.redecode_from_cache(
            cached_run,
            identity_overrides={
                "shot-0-track-1": "FIGHTER_A",
                "shot-0-track-2": "FIGHTER_A",
            },
        )
    assert (cached_run / "summary.json").read_bytes() == summary


def test_unknown_override_target_rejected_before_any_writes(cached_run):
    with pytest.raises(ValueError, match="Неизвестная сцена"):
        cache_review.redecode_from_cache(
            cached_run, scene_overrides={"999": "ACTIVE_FIGHT"}
        )


@pytest.fixture
def actual_segment_cache(tmp_path, monkeypatch):
    """Use the real deterministic decoder; all neural entry points are fatal."""
    from boxing_vision import pipeline as p
    from boxing_vision.pose import RTMLibPoseBackend
    from boxing_vision.punch_models import AcmPunchClassifier

    def forbidden(*args, **kwargs):
        pytest.fail("Neural inference is forbidden during cached identity correction")

    monkeypatch.setattr(p, "create_pose_backend", forbidden)
    monkeypatch.setattr(RTMLibPoseBackend, "__init__", forbidden)
    monkeypatch.setattr(AcmPunchClassifier, "__init__", forbidden)
    monkeypatch.setattr(p, "probe_video", lambda _: SimpleNamespace(width=640, height=360, fps=30, duration_s=2.1))

    def pose(source, left, appearance):
        points = {name: Keypoint(left + x, y, .95) for name, x, y in (
            ("nose", 50, 30), ("left_shoulder", 25, 65), ("right_shoulder", 75, 65),
            ("left_elbow", 15, 100), ("right_elbow", 85, 100),
            ("left_wrist", 25, 120), ("right_wrist", 75, 120),
            ("left_hip", 30, 150), ("right_hip", 70, 150))}
        return RawPose(BBox(left, 10, left + 100, 250), points, .95, appearance,
                       source_track_id=source, detector_confidence=.95, pose_confidence=.95)

    poses = [pose(1, 30, (1, 0, 0)), pose(2, 280, (0, 1, 0))]
    tracker = TwoFighterTracker(("fighter_a", "fighter_b"))
    tracker.enroll({"fighter_a": [poses[0]] * 3, "fighter_b": [poses[1]] * 3})
    profile = tracker.export_identity_profile()
    frames = [TrackingFrame(index, stamp, 0, poses) for index, stamp in enumerate(range(0, 2100, 100))]
    frames[10].tracker_predictions = [{"source_track_id": 99, "bbox": BBox(450, 10, 550, 250).to_dict()}]
    decoder = OfflineIdentityDecoder(tracker)
    observations = decoder.decode(frames)
    assert observations
    cache = tmp_path / p._RENDER_CACHE_DIRNAME
    cache.mkdir()
    config = AnalysisConfig(punch_model="auto", scheduled_rounds=1, analysis_fps=10)
    atomic_write_json(cache / p._RENDER_CACHE_CONFIG, {"version": p._RENDER_CACHE_VERSION, "config": config.to_dict()})
    atomic_write_json(cache / "first_pass.json", {"identity_profile": profile, "frame_states": states(range(0, 2100, 100))})
    atomic_write_json(tmp_path / "summary.json", {"metadata": {"duration_s": 2.1, "export_stale": False}, "quality": {"winner_visible": False}})
    atomic_write_json(tmp_path / "events.json", [])
    atomic_write_json(tmp_path / "review.json", {"items": [], "identity_review_history": []})
    p._write_observation_cache(cache / p._RENDER_CACHE_OBSERVATIONS, observations)
    with gzip.open(cache / "detections.jsonl.gz", "wt") as output:
        for frame in frames:
            output.write(json.dumps(frame.to_dict()) + "\n")
    p._write_jsonl(tmp_path / "tracking_diagnostics.jsonl", decoder.diagnostics)
    p._write_jsonl(tmp_path / "tracklets.jsonl", decoder.tracklets)
    return tmp_path


def test_correction_frame_has_real_bbox_and_excludes_motion_predictions(actual_segment_cache):
    frame = cache_review.get_identity_correction_frame(actual_segment_cache, 1033)
    assert frame["timestamp_ms"] == 1000
    assert (frame["width"], frame["height"]) == (640, 360)
    assert {item["source_track_id"] for item in frame["candidates"]} == {1, 2}
    assert frame["candidates"][0]["bbox"]["x1"] == 30
    assert all(item["segment_id"] for item in frame["candidates"])


@pytest.mark.parametrize("stamp", [-1, float("nan"), float("inf"), 2300])
def test_correction_rejects_invalid_or_outside_cache_time(actual_segment_cache, stamp):
    before = (actual_segment_cache / "summary.json").read_bytes()
    with pytest.raises(ValueError):
        cache_review.correct_identity_at(actual_segment_cache, timestamp_ms=stamp,
                                         source_track_id=1, identity_state="FIGHTER_A")
    assert (actual_segment_cache / "summary.json").read_bytes() == before


def test_forecast_source_cannot_be_identity_corrected(actual_segment_cache):
    with pytest.raises(ValueError, match="отсутствует"):
        cache_review.correct_identity_at(actual_segment_cache, timestamp_ms=1000,
                                         source_track_id=99, identity_state="FIGHTER_A")


def test_invalid_cached_bbox_cannot_be_identity_corrected(actual_segment_cache):
    root = actual_segment_cache
    path = root / ".render_cache/detections.jsonl.gz"
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    rows[10]["poses"][0]["bbox"]["x2"] = 10  # x2 < x1: no visible detector rectangle.
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    before = (root / "summary.json").read_bytes()
    with pytest.raises(ValueError):
        cache_review.correct_identity_at(root, timestamp_ms=1000, source_track_id=1, identity_state="OTHER")
    assert (root / "summary.json").read_bytes() == before


def test_stale_segment_is_rejected_before_artifact_changes(actual_segment_cache):
    before = (actual_segment_cache / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="Сегмент изменился"):
        cache_review.correct_identity_at(actual_segment_cache, timestamp_ms=1000,
                                         source_track_id=1, identity_state="FIGHTER_A",
                                         segment_id="shot-0-track-1-segment-9999")
    assert (actual_segment_cache / "summary.json").read_bytes() == before


def test_segment_correction_records_history_without_whole_source_override(actual_segment_cache):
    root = actual_segment_cache
    frame = cache_review.get_identity_correction_frame(root, 1000)
    selected = next(item for item in frame["candidates"] if item["source_track_id"] == 1)
    result = cache_review.correct_identity_at(root, timestamp_ms=1000, source_track_id=1,
                                             identity_state="OTHER", segment_id=selected["segment_id"])
    config = json.loads((root / ".render_cache" / "config.json").read_text())["config"]
    assert config["identity_overrides"] == {}
    assert config["segment_identity_overrides"] == {selected["segment_id"]: "OTHER"}
    assert result["metadata"]["export_stale"] is True
    assert result["metadata"]["review_recompute"] == "cached_evidence_no_ml"
    history = json.loads((root / "review.json").read_text())["identity_review_history"]
    assert len(history) == 1
    assert history[0]["timestamp_ms"] == 1000
    assert history[0]["segment_id"] == selected["segment_id"]
    assert history[0]["identity_state"] == "OTHER"


def test_correction_does_not_relabel_later_segment_of_recycled_source(actual_segment_cache):
    root = actual_segment_cache
    path = root / ".render_cache/detections.jsonl.gz"
    with gzip.open(path, "rt") as handle:
        frames = [json.loads(line) for line in handle]
    # Same local ID, but a clear geometry jump creates a different physical segment.
    for frame in frames:
        if frame["timestamp_ms"] >= 1100:
            pose = next(item for item in frame["poses"] if item["source_track_id"] == 1)
            for name in ("x1", "x2"):
                pose["bbox"][name] += 420
            for point in pose["keypoints"].values():
                point["x"] += 420
    with gzip.open(path, "wt") as handle:
        for frame in frames:
            handle.write(json.dumps(frame) + "\n")
    cache_review.correct_identity_at(root, timestamp_ms=500, source_track_id=1,
                                    identity_state="OTHER", segment_id="shot-0-track-1-segment-0")
    with gzip.open(root / ".render_cache/observations.jsonl.gz", "rt") as handle:
        observations = [json.loads(line) for line in handle]
    later = [row for row in observations if row["source_track_id"] == 1 and row["fighter_id"] == "fighter_a"]
    assert later, "Unrelated later segment must retain its independent enrollment identity"
    assert all(row["timestamp_ms"] >= 1100 for row in later)
    assert {row["segment_id"] for row in later} == {"shot-0-track-1-segment-1100"}


def test_legacy_diagnostics_resolve_segment_in_memory_without_neural_inference(actual_segment_cache):
    root = actual_segment_cache
    diagnostic_file = root / "tracking_diagnostics.jsonl"
    rows = [json.loads(line) for line in diagnostic_file.read_text().splitlines()]
    for row in rows:
        row.pop("segment_id", None)
    from boxing_vision import pipeline as p
    p._write_jsonl(diagnostic_file, rows)
    assert all(item["segment_id"] is None for item in cache_review.get_identity_correction_frame(root, 1000)["candidates"])
    result = cache_review.correct_identity_at(root, timestamp_ms=1000, source_track_id=1, identity_state="OTHER")
    assert result["metadata"]["export_stale"]
    config = json.loads((root / ".render_cache" / "config.json").read_text())["config"]
    assert config["identity_overrides"] == {}
    assert list(config["segment_identity_overrides"]) == ["shot-0-track-1-segment-0"]


def test_duplicate_segment_role_is_rejected_without_changing_existing_review(actual_segment_cache):
    root = actual_segment_cache
    info = cache_review.get_identity_correction_frame(root, 1000)
    first, second = info["candidates"]
    cache_review.correct_identity_at(root, timestamp_ms=1000, source_track_id=first["source_track_id"],
                                    identity_state="FIGHTER_A", segment_id=first["segment_id"])
    before = {name: (root / name).read_bytes() for name in ("summary.json", "review.json", ".render_cache/config.json")}
    with pytest.raises(ValueError, match="два|одним бойцом|конфликт"):
        cache_review.correct_identity_at(root, timestamp_ms=1000, source_track_id=second["source_track_id"],
                                        identity_state="FIGHTER_A", segment_id=second["segment_id"])
    assert {name: (root / name).read_bytes() for name in before} == before


def test_continuous_review_mode_changes_schedule_but_preserves_replay(actual_segment_cache):
    root = actual_segment_cache
    path = root / ".render_cache/first_pass.json"
    first = json.loads(path.read_text())
    for row in first["frame_states"]:
        if 1000 <= row["timestamp_ms"] < 1700:
            row["scene_state"] = "BREAK"
        elif row["timestamp_ms"] >= 1700:
            row["scene_state"] = "REPLAY"
    atomic_write_json(path, first)
    result = cache_review.redecode_from_cache(root, timing_mode="continuous")
    updated = json.loads(path.read_text())["frame_states"]
    assert all(row["scene_state"] == "ACTIVE_FIGHT" for row in updated if row["timestamp_ms"] < 1700)
    assert all(row["scene_state"] == "REPLAY" for row in updated if row["timestamp_ms"] >= 1700)
    assert result["metadata"]["timing_mode"] == "continuous"
