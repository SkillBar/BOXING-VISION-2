from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from boxing_vision.config import AnalysisConfig
from boxing_vision.contracts import (
    BBox,
    IdentityState,
    Keypoint,
    PoseObservation,
    PunchEvent,
    ReviewStatus,
    SceneState,
)
from boxing_vision.pipeline import (
    AnalysisCancelledError,
    RenderCacheUnavailableError,
    _create_identity_tracker,
    _read_observation_cache,
    _renderable_events,
    _sample_pose_tracks,
    _write_observation_cache,
    analyze_video,
    rebuild_from_cache,
)
from boxing_vision.video import VideoValidationError

FFMPEG = shutil.which("ffmpeg")


def test_pipeline_wires_complete_adaptive_identity_policy() -> None:
    tracker = _create_identity_tracker(
        AnalysisConfig(
            identity_gallery_distance_max=0.31,
            identity_margin_min=0.14,
            adaptive_identity_confidence_min=0.86,
            adaptive_identity_margin_min=0.24,
        )
    )

    assert tracker.gallery_max_distance == 0.31
    assert tracker.gallery_min_margin == 0.14
    assert tracker.adaptive_identity_confidence_min == 0.86
    assert tracker.adaptive_identity_margin_min == 0.24


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg unavailable")
def test_pipeline_backend_none_creates_complete_browser_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp4"
    portrait = tmp_path / "fighter-a-private-name.png"
    Image.new("RGB", (320, 480), (180, 40, 40)).save(portrait)
    subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x96:rate=12:duration=0.7",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.7",
            "-shortest",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    config = AnalysisConfig(
        backend="none",
        scheduled_rounds=1,
        round_length_s=60,
        output_height=360,
        analysis_fps=5,
        fighter_a_anchor=(0.25, 0.5),
        fighter_b_anchor=(0.75, 0.5),
        fighter_a_record="14-2",
        fighter_a_portrait_path=portrait,
        hud_mode="compact",
    )

    result = analyze_video(source, config, runs_dir=tmp_path / "runs")

    assert result.annotated_video.is_file()
    assert result.events_path.is_file()
    assert result.summary_path.is_file()
    assert result.log_path.is_file()
    assert result.events == []
    metadata = result.summary["metadata"]
    assert metadata["backend"] == "unavailable"
    assert metadata["scheduled_rounds"] == 1
    assert metadata["round_length_s"] == 60
    assert metadata["rest_length_s"] == 60
    assert metadata["fight_start_s"] == 0.0
    assert metadata["fight_end_s"] == pytest.approx(metadata["duration_s"], abs=0.05)
    assert metadata["output_fps"] == 30
    assert metadata["hud_mode"] == "compact"
    assert metadata["render_cache_available"] is True
    assert metadata["export_stale"] is False
    assert metadata["preview_manifest"] == "preview_manifest.json"
    for key, filename in {
        "body_map_manifest": "body_map_manifest.json",
        "model_manifest": "model_manifest.json",
        "tracklets": "tracklets.jsonl",
        "tracking_diagnostics": "tracking_diagnostics.jsonl",
        "scenes": "scenes.json",
        "identity_profile": "identity_profile.json",
        "review": "review.json",
    }.items():
        assert metadata[key] == filename
        assert (result.run_dir / filename).is_file()
    assert (
        result.run_dir / "assets" / "body-map-v3-base.png"
    ).is_file()
    review = json.loads((result.run_dir / "review.json").read_text(encoding="utf-8"))
    scenes = json.loads((result.run_dir / "scenes.json").read_text(encoding="utf-8"))
    assert review["required_count"] == 1
    assert review["items"][0]["reason"] == "fighters_not_confirmed"
    assert scenes == [
        {
            "shot_id": 0,
            "start_ms": 0,
            "end_ms": pytest.approx(metadata["duration_s"] * 1000, abs=2),
            "scene_state": "UNCERTAIN",
            "review_status": "NEEDS_REVIEW",
        }
    ]
    assert result.summary["quality"]["identity_verified_coverage"] == 0.0
    assert result.summary["quality"]["winner_visible"] is False
    fighter_a = result.summary["fighters"]["fighter_a"]
    fighter_b = result.summary["fighters"]["fighter_b"]
    assert fighter_a["record"] == "14-2"
    assert fighter_a["portrait_filename"] == "profiles/fighter_a.webp"
    assert "record" not in fighter_b
    assert "portrait_filename" not in fighter_b
    stored_portrait = result.run_dir / fighter_a["portrait_filename"]
    with Image.open(stored_portrait) as processed:
        assert processed.size == (256, 256)
    serialized_summary = result.summary_path.read_text(encoding="utf-8")
    assert str(portrait) not in serialized_summary
    assert fighter_a["landed_targets"]["head"] == {"landed": 0, "thrown": 0}
    assert fighter_b["received_landed_targets"]["body"] == {
        "landed": 0,
        "thrown": 0,
    }
    assert (result.run_dir / "preview_manifest.json").is_file()
    render_cache = result.run_dir / ".render_cache"
    assert (render_cache / "normalized.mp4").is_file()
    assert (render_cache / "observations.jsonl.gz").is_file()
    config_cache = (render_cache / "config.json").read_text(encoding="utf-8")
    assert str(portrait) not in config_cache

    def forbid_ml(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("render-only rebuild must not create an ML backend")

    monkeypatch.setattr("boxing_vision.pipeline.create_pose_backend", forbid_ml)
    rebuilt = rebuild_from_cache(result.run_dir)

    assert rebuilt == result.annotated_video
    assert rebuilt.is_file()
    rebuilt_summary = result.summary_path.read_text(encoding="utf-8")
    assert '"export_stale": false' in rebuilt_summary
    assert "render_rebuild_completed" in result.log_path.read_text(encoding="utf-8")


def test_pipeline_honors_immediate_cancellation_before_creating_job(
    tmp_path: Path,
) -> None:
    with pytest.raises(AnalysisCancelledError):
        analyze_video(
            tmp_path / "missing.mp4",
            AnalysisConfig(),
            runs_dir=tmp_path / "runs",
            cancel_callback=lambda: True,
        )
    assert not (tmp_path / "runs").exists()


def test_render_rebuild_explains_missing_cache_for_legacy_run(tmp_path: Path) -> None:
    legacy_run = tmp_path / "legacy-run"
    legacy_run.mkdir()

    with pytest.raises(
        RenderCacheUnavailableError, match="Запустите исходное видео заново"
    ):
        rebuild_from_cache(legacy_run)


def test_observation_render_cache_round_trips_pose_data(tmp_path: Path) -> None:
    observation = PoseObservation(
        frame_index=12,
        timestamp_ms=400,
        fighter_id="fighter_b",
        bbox=BBox(10.5, 20.25, 80.75, 190.0, 0.88),
        keypoints={"left_wrist": Keypoint(42.25, 61.5, 0.93)},
        track_confidence=0.91,
        is_scene_cut=True,
        source_track_id="shot-4-track-9",
        detector_bbox=BBox(8, 18, 84, 194, 0.94),
        shot_id=4,
        identity_state=IdentityState.FIGHTER_B,
        identity_confidence=0.97,
        identity_margin=0.31,
        scene_state=SceneState.ACTIVE_FIGHT,
        review_status=ReviewStatus.USER_CONFIRMED,
    )
    cache_path = tmp_path / "observations.jsonl.gz"

    _write_observation_cache(cache_path, [observation])
    restored = _read_observation_cache(cache_path)

    assert [item.to_dict() for item in restored] == [observation.to_dict()]


def test_render_sampler_interpolates_short_pose_gaps_without_crossing_cut() -> None:
    before = PoseObservation(
        frame_index=0,
        timestamp_ms=0,
        fighter_id="fighter_a",
        bbox=BBox(0, 10, 100, 210),
        keypoints={"left_wrist": Keypoint(20, 50, 0.8)},
        track_confidence=0.8,
    )
    after = PoseObservation(
        frame_index=1,
        timestamp_ms=200,
        fighter_id="fighter_a",
        bbox=BBox(100, 20, 200, 220),
        keypoints={"left_wrist": Keypoint(120, 70, 1.0)},
        track_confidence=1.0,
    )
    tracks = {"fighter_a": [before, after]}
    sampled = _sample_pose_tracks(tracks, {"fighter_a": 0}, 100)

    assert sampled[0].bbox.x1 == pytest.approx(50)
    assert sampled[0].keypoints["left_wrist"].x == pytest.approx(70)
    assert sampled[0].track_confidence == pytest.approx(0.9)

    after.is_scene_cut = True
    held = _sample_pose_tracks(tracks, {"fighter_a": 0}, 100)
    assert held[0].bbox.x1 == 0


def test_render_only_overlay_excludes_rejected_review_events() -> None:
    visible = PunchEvent(
        "visible",
        1,
        100,
        200,
        300,
        "fighter_a",
        "fighter_b",
        "left",
        "jab",
        "head",
        "likely_landed",
        0.9,
        70,
    )
    rejected = PunchEvent(
        "rejected",
        1,
        400,
        500,
        600,
        "fighter_b",
        "fighter_a",
        "right",
        "hook",
        "body",
        "missed",
        0.8,
        40,
        review_status="rejected",
    )

    assert _renderable_events([visible, rejected]) == [visible]


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg unavailable")
def test_preview_failure_is_best_effort_and_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp4"
    subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=10:duration=0.2",
            "-c:v",
            "mpeg4",
            str(source),
        ],
        check=True,
    )

    def fail_previews(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("simulated preview codec failure")

    monkeypatch.setattr(
        "boxing_vision.pipeline.generate_hover_previews",
        fail_previews,
    )
    result = analyze_video(
        source,
        AnalysisConfig(
            backend="none",
            scheduled_rounds=1,
            round_length_s=60,
            output_height=360,
            analysis_fps=5,
        ),
        runs_dir=tmp_path / "runs",
    )

    assert result.annotated_video.is_file()
    assert "preview_manifest" not in result.summary["metadata"]
    assert not (result.run_dir / "preview_manifest.json").exists()
    assert not (result.run_dir / "previews").exists()
    log = result.log_path.read_text(encoding="utf-8")
    assert "preview_generation_failed=RuntimeError" in log
    assert "simulated preview codec failure" in log


def test_pipeline_failure_removes_large_work_files_but_keeps_log(
    tmp_path: Path,
) -> None:
    with pytest.raises(VideoValidationError):
        analyze_video(
            tmp_path / "missing.mp4",
            AnalysisConfig(),
            runs_dir=tmp_path / "runs",
        )

    jobs = list((tmp_path / "runs").iterdir())
    assert len(jobs) == 1
    assert not (jobs[0] / ".work").exists()
    log = (jobs[0] / "analysis.log").read_text(encoding="utf-8")
    assert "failed=" in log
