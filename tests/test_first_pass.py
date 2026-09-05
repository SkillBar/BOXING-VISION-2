import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxing_vision import first_pass
from boxing_vision.config import AnalysisConfig
from boxing_vision.contracts import BBox, PoseObservation
from boxing_vision.video import VideoFrame


class Backend:
    def __init__(self):
        self.inferences = 0
        self.resets = 0

    def infer(self, image):
        self.inferences += 1
        return []

    def reset(self):
        self.resets += 1


class Tracker:
    def export_identity_profile(self):
        return {"fixture": "synthetic_test_only"}


@pytest.fixture
def first_pass_fakes(monkeypatch):
    mode = {"decode": "prefix_only", "scans": [], "scheduled": "ACTIVE_FIGHT", "identity_confidence": 0.95, "predictions": []}

    def detect(path, **kwargs):
        bound = kwargs.get("max_duration_ms")
        mode["scans"].append(bound)
        return [stamp for stamp in [0, 5000, 11000] if bound is None or stamp <= bound]

    def source(path, **kwargs):
        for index in range(120):
            yield VideoFrame(index, index * 100, np.zeros((16, 24, 3), np.uint8))

    class Local:
        def __init__(self, **kwargs):
            self.predictions = mode["predictions"]

        def update(self, poses, image, stamp, shot):
            return poses

    class Decoder:
        def __init__(self, tracker, *, segment_identity_overrides=None):
            self.segment_identity_overrides = dict(segment_identity_overrides or {})
            self.diagnostics = []
            self.tracklets = []

        def decode(self, frames):
            if mode["decode"] == "none":
                return []
            result = []
            for frame in frames:
                if mode["decode"] == "prefix_only" and frame.timestamp_ms >= 10000:
                    continue
                for role, source in (("fighter_a", 1), ("fighter_b", 2)):
                    if mode["decode"] == "returning_b" and role == "fighter_b" and 2000 <= frame.timestamp_ms < 11000:
                        continue
                    result.append(
                        PoseObservation(
                            frame.frame_index,
                            frame.timestamp_ms,
                            role,
                            BBox(0, 0, 10, 20),
                            {},
                            source_track_id=source,
                            shot_id=frame.shot_id,
                            scene_state=frame.scene_state,
                            identity_confidence=mode["identity_confidence"],
                            identity_margin=0.3,
                        )
                    )
            return result

    class Replay:
        def update(self, *args, **kwargs):
            return False

    monkeypatch.setattr(first_pass, "enroll_video", lambda *args: {})
    monkeypatch.setattr(
        first_pass, "probe_video", lambda _: SimpleNamespace(duration_s=12)
    )
    monkeypatch.setattr(first_pass, "detect_shot_boundaries", detect)
    monkeypatch.setattr(first_pass, "iter_video_frames", source)
    monkeypatch.setattr(first_pass, "ReplayDetector", Replay)
    monkeypatch.setattr(first_pass, "_scheduled", lambda *args: mode["scheduled"])
    monkeypatch.setattr("boxing_vision.tracking.ShotLocalBoTSORT", Local)
    monkeypatch.setattr("boxing_vision.tracking.OfflineIdentityDecoder", Decoder)
    return mode


def test_preflight_uncertainty_continues_full_scan_without_fabricating_identities(
    tmp_path, first_pass_fakes
):
    first_pass_fakes["decode"] = "none"
    backend = Backend()
    result = first_pass.run_first_pass(
            Path("unused"),
            tmp_path,
            backend,
            Tracker(),
            AnalysisConfig(analysis_fps=10),
            progress=lambda *args: None,
    )
    assert first_pass_fakes["scans"] == [10200, None]
    assert backend.inferences == 120
    assert (tmp_path / "preflight.json").is_file()
    assert (tmp_path / "preflight_diagnostics.json").is_file()
    report = json.loads((tmp_path / "preflight.json").read_text())
    assert report["fighter_coverage"] == {"fighter_a": 0, "fighter_b": 0}
    assert report["active_frames"] == 100
    assert report["blocking"] is False
    assert report["status"] == "needs_review"
    assert result.observations == []
    assert (tmp_path / "first_pass.json").exists()
    with gzip.open(tmp_path / "detections.jsonl.gz", "rt") as handle:
        assert len(handle.readlines()) == 120


def test_passed_preflight_scans_rest_and_preserves_cut_without_fighter_observations(
    tmp_path, first_pass_fakes
):
    backend = Backend()
    result = first_pass.run_first_pass(
        Path("unused"),
        tmp_path,
        backend,
        Tracker(),
        AnalysisConfig(analysis_fps=10),
        progress=lambda *args: None,
    )
    assert result.preflight["status"] == "passed"
    assert (
        first_pass_fakes["scans"][0] is not None
        and first_pass_fakes["scans"][-1] is None
    )
    assert backend.inferences == 120  # Prefix detector results were not recomputed.
    assert not any(obs.timestamp_ms >= 10000 for obs in result.observations)
    cuts = [row["timestamp_ms"] for row in result.frame_states if row["is_scene_cut"]]
    assert cuts == [0, 5000, 11000]


def test_confirmed_people_during_break_cannot_pass_preflight(
    tmp_path, first_pass_fakes
):
    first_pass_fakes["scheduled"] = "BREAK"
    result = first_pass.run_first_pass(
            Path("unused"),
            tmp_path,
            Backend(),
            Tracker(),
            AnalysisConfig(analysis_fps=10),
            progress=lambda *args: None,
    )
    assert result.preflight["status"] == "needs_review"
    assert result.preflight["blocking"] is False
    assert first_pass_fakes["scans"] == [10200, None]


@pytest.mark.parametrize("confidence", [0.2, 0.549, float("nan")])
def test_low_emitted_identity_confidence_cannot_pass_preflight(
    tmp_path, first_pass_fakes, confidence
):
    # An invalid/weak emitted identity score remains disqualifying even when
    # a producer labels it AUTO_CONFIRMED. Pose quality is a separate field.
    first_pass_fakes["identity_confidence"] = confidence
    backend = Backend()
    result = first_pass.run_first_pass(
        Path("unused"), tmp_path, backend, Tracker(),
        AnalysisConfig(analysis_fps=10), progress=lambda *args: None,
    )
    assert json.loads((tmp_path / "preflight.json").read_text())["pair_coverage"] == 0
    assert result.preflight["blocking"] is False
    assert first_pass_fakes["scans"] == [10200, None]
    assert backend.inferences == 120


def test_preflight_uses_same_inclusive_confidence_boundary_as_export(
    tmp_path, first_pass_fakes
):
    first_pass_fakes["identity_confidence"] = 0.55
    result = first_pass.run_first_pass(
        Path("unused"), tmp_path, Backend(), Tracker(),
        AnalysisConfig(analysis_fps=10), progress=lambda *args: None,
    )
    assert result.preflight["status"] == "passed"


def test_cancelled_first_pass_stops_and_does_not_publish_complete_cache(
    tmp_path, first_pass_fakes
):
    backend, calls = Backend(), 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls >= 20:
            raise InterruptedError("cancel")

    with pytest.raises(InterruptedError):
        first_pass.run_first_pass(
            Path("unused"),
            tmp_path,
            backend,
            Tracker(),
            AnalysisConfig(analysis_fps=10),
            progress=lambda *args: None,
            check_cancel=cancel,
        )
    assert backend.inferences < 20
    assert not (tmp_path / "first_pass.json").exists()


def test_unconfirmed_enrollment_rejects_before_opening_source(monkeypatch):
    monkeypatch.setattr(
        first_pass.cv2,
        "VideoCapture",
        lambda _: pytest.fail("Opened before enrollment confirmation"),
    )
    with pytest.raises(first_pass.EnrollmentRequiredError):
        first_pass.enroll_video(
            Path("unused"),
            Backend(),
            Tracker(),
            AnalysisConfig(enrollment_confirmed=False),
        )


def test_known_replay_in_open_shot_cannot_pass_preflight(
    tmp_path, first_pass_fakes, monkeypatch
):
    """An already-confirmed replay must not become ACTIVE in the preview decoder."""
    class Replay:
        def update(self, image, timestamp_ms, **kwargs):
            return timestamp_ms == 7000

    monkeypatch.setattr(first_pass, "ReplayDetector", Replay)
    backend = Backend()
    result = first_pass.run_first_pass(
        Path("unused"), tmp_path, backend, Tracker(),
        AnalysisConfig(analysis_fps=10), progress=lambda *args: None,
    )

    report = json.loads((tmp_path / "preflight.json").read_text())
    # The 5–10 s shot is still open, but it was classified as replay at 7 s.
    # All 100 frames were scheduled ACTIVE; only 0–5 s may confirm identities.
    assert report["active_frames"] == 100
    assert report["confirmed_pair_frames"] == 50
    assert report["pair_coverage"] == .5
    assert report["required_pair_coverage"] == .7
    assert report["status"] == "needs_review"
    assert report["unconfirmed_intervals"]["fighter_a"] == [
        {"start_ms": 5000, "end_ms": 10000}
    ]
    assert (tmp_path / "preflight_diagnostics.json").is_file()
    assert first_pass_fakes["scans"] == [10200, None]
    assert backend.inferences == 120
    assert result.preflight["blocking"] is False
    assert result.replay_intervals == [(5000, 11000)]


def test_inactive_b_in_prefix_can_return_after_ten_seconds(tmp_path, first_pass_fakes):
    first_pass_fakes["decode"] = "returning_b"
    progress = []
    result = first_pass.run_first_pass(
        Path("unused"), tmp_path, Backend(), Tracker(),
        AnalysisConfig(analysis_fps=10, region_mode="none", timing_mode="continuous"),
        progress=lambda value, text: progress.append(text),
    )
    assert result.preflight["fighter_coverage"]["fighter_b"] == .2
    assert result.preflight["blocking"] is False
    blue = [o for o in result.observations if o.fighter_id == "fighter_b"]
    assert not any(2000 <= o.timestamp_ms < 11000 for o in blue)
    assert sum(o.timestamp_ms >= 11000 for o in blue) == 10
    assert result.frame_states[-1]["timestamp_ms"] == 11900
    assert any("приостановлен" in text for text in progress)
    assert not any("пройдена" in text for text in progress)


def test_closed_replay_preserves_scheduled_denominator_after_final_decode(
    tmp_path, first_pass_fakes, monkeypatch
):
    from boxing_vision.preflight import build_preflight_report

    def detect(path, **kwargs):
        bound = kwargs.get("max_duration_ms")
        first_pass_fakes["scans"].append(bound)
        return [stamp for stamp in [0, 4000, 7000, 11000]
                if bound is None or stamp <= bound]

    class Replay:
        def update(self, image, timestamp_ms, **kwargs):
            return timestamp_ms == 6000

    monkeypatch.setattr(first_pass, "detect_shot_boundaries", detect)
    monkeypatch.setattr(first_pass, "ReplayDetector", Replay)
    result = first_pass.run_first_pass(
        Path("unused"), tmp_path, Backend(), Tracker(),
        AnalysisConfig(analysis_fps=10), progress=lambda *args: None,
    )
    final_report = build_preflight_report(
        result.observations, result.frame_states, result.diagnostics,
        window_ms=10000,
    )
    for report in (result.preflight, final_report):
        assert report["active_frames"] == 100
        assert report["confirmed_pair_frames"] == 70
        assert report["pair_coverage"] == .7
        assert report["required_pair_coverage"] == .7
        assert report["status"] == "passed"
    replay_frames = [row for row in result.frame_states
                     if 4000 <= row["timestamp_ms"] < 7000]
    assert len(replay_frames) == 30
    assert all(row["scene_state"] == "REPLAY" for row in replay_frames)
    assert all(row["scheduled_scene_state"] == "ACTIVE_FIGHT"
               for row in replay_frames)
    assert result.replay_intervals == [(4000, 7000)]


def test_first_pass_caches_motion_predictions_separately_from_detector_poses(tmp_path, first_pass_fakes):
    first_pass_fakes["predictions"] = [{"source_track_id": 99, "bbox": {"x1": 1, "y1": 2, "x2": 8, "y2": 12}}]
    first_pass.run_first_pass(Path("unused"), tmp_path, Backend(), Tracker(),
                              AnalysisConfig(analysis_fps=10), progress=lambda *args: None)
    with gzip.open(tmp_path / "detections.jsonl.gz", "rt") as handle:
        cached = json.loads(next(handle))
    assert cached["poses"] == []
    assert cached["tracker_predictions"] == first_pass_fakes["predictions"]


def test_unrestricted_first_pass_does_not_apply_stale_manual_polygon(tmp_path, first_pass_fakes):
    # Tracker deliberately has no set_ring_roi: unrestricted mode must not call it.
    config = AnalysisConfig(analysis_fps=10, region_mode="none",
                            ring_rois=((0, 0), (1, 0), (1, 1), (0, 1)))
    result = first_pass.run_first_pass(Path("unused"), tmp_path, Backend(), Tracker(),
                                       config, progress=lambda *args: None)
    assert result.preflight["status"] == "passed"


def test_continuous_timing_never_inserts_scheduled_break_and_keeps_explicit_scene():
    assert first_pass._scheduled(185000, 0, AnalysisConfig()) == "BREAK"
    assert first_pass._scheduled(185000, 0, AnalysisConfig(timing_mode="continuous")) == "ACTIVE_FIGHT"
    assert first_pass._scheduled(185000, 0, AnalysisConfig(timing_mode="continuous", scene_overrides={"0": "REPLAY"})) == "REPLAY"
