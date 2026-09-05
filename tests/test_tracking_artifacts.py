from dataclasses import replace

from boxing_vision.config import AnalysisConfig
from boxing_vision.contracts import BBox, DisplayTrack, PoseObservation
from boxing_vision.pipeline import _config_from_render_cache, _enrich_summary
from boxing_vision.scoring import build_fight_summary
from boxing_vision.tracking_artifacts import _write_gzip, read_display_cache


def test_display_cache_roundtrip_does_not_infer_identity_from_label(tmp_path):
    path = tmp_path / "display.jsonl.gz"
    original = DisplayTrack(timestamp_ms=1000, evidence_timestamp_ms=800,
        bbox=BBox(10, 20, 100, 200), fighter_id="fighter_b", display_state="PREDICTED")
    _write_gzip(path, [original.to_dict()])
    actual, = read_display_cache(path)
    assert actual == original
    assert actual.identity_state == "UNKNOWN"
    assert actual.keypoints == {}


def test_filtered_conflict_cannot_disappear_from_summary_quality():
    observations = [PoseObservation(0, 0, role, BBox(0, 0, 100, 200), {},
                    source_track_id=index, identity_margin=.5)
                    for index, role in enumerate(("fighter_a", "fighter_b"))]
    summary = _enrich_summary(build_fight_summary([], scheduled_rounds=1),
        config=AnalysisConfig(analysis_fps=1), duration_s=1, processing_s=0,
        backend_name="test", observations=observations, events=[],
        frame_states=[{"timestamp_ms": 0, "scene_state": "ACTIVE_FIGHT"}],
        tracking_diagnostics=[{"timestamp_ms": 0, "reason": "segment_identity_conflict"}])
    assert summary["quality"]["identity_verified_coverage"] == 1
    assert summary["quality"]["identity_swap_suspected"] is True
    assert summary["quality"]["winner_visible"] is False


def test_locally_resolved_recycled_source_is_not_itself_global_identity_swap():
    first = PoseObservation(0, 0, "fighter_a", BBox(0, 0, 100, 200), {},
                            source_track_id=1, segment_id="prefix", identity_margin=.5)
    suffix = replace(first, timestamp_ms=1000, fighter_id="fighter_b",
                     identity_state="FIGHTER_B", segment_id="suffix")
    summary = _enrich_summary(build_fight_summary([], scheduled_rounds=1),
        config=AnalysisConfig(analysis_fps=1), duration_s=2, processing_s=0,
        backend_name="test", observations=[first, suffix], events=[])
    assert summary["quality"]["identity_swap_suspected"] is False
    assert summary["quality"]["winner_visible"] is False  # still incomplete pair


def test_render_cache_v1_v2_v3_remain_readable():
    for version in (1, 2, 3):
        config, rounds = _config_from_render_cache({"version": version,
            "config": {"scheduled_rounds": 1}, "rounds_to_score": 1})
        assert config.timing_mode == "scheduled"
        assert rounds == 1


def test_unreviewed_conflict_boundary_remains_in_quality_before_filtering():
    row = {"timestamp_ms": 0, "segment_reason": "identity_conflict_boundary"}
    summary = _enrich_summary(build_fight_summary([], scheduled_rounds=1),
        config=AnalysisConfig(), duration_s=1, processing_s=0,
        backend_name="test", observations=[], events=[], tracking_diagnostics=[row])
    assert summary["quality"]["unresolved_identity_conflict_samples"] == 1
    assert summary["quality"]["identity_swap_suspected"]


def test_render_pose_sampler_does_not_interpolate_across_identity_segments():
    from boxing_vision.pipeline import _sample_pose_tracks
    first = PoseObservation(0, 0, "fighter_a", BBox(0, 0, 100, 200), {},
                            source_track_id=1, segment_id="first", physical_track_id="physical")
    second = replace(first, timestamp_ms=200, bbox=BBox(100, 0, 200, 200), segment_id="second")
    sampled, = _sample_pose_tracks({"fighter_a": [first, second]}, {}, 100)
    assert sampled.bbox == first.bbox  # hold previous, never blend across boundary
    assert sampled.segment_id == "first"
    assert sampled.physical_track_id == "physical"
