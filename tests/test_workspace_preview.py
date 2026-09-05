from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from boxing_vision import pipeline, ui
from boxing_vision.config import AnalysisConfig
from boxing_vision.contracts import (
    BBox,
    Keypoint,
    PoseObservation,
    PunchEvent,
    RenderFrameContext,
)
from boxing_vision.render import FrameRenderer

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def fighters() -> list[PoseObservation]:
    return [PoseObservation(
        frame_index=30, timestamp_ms=1000, fighter_id=role,
        bbox=BBox(x, 90, x + 125, 335),
        keypoints={"left_shoulder": Keypoint(x + 35, 145, .99),
                   "right_shoulder": Keypoint(x + 85, 145, .99),
                   "left_elbow": Keypoint(x + 15, 200, .99),
                   "right_elbow": Keypoint(x + 100, 200, .99),
                   "left_wrist": Keypoint(x + 65, 180, .99),
                   "right_wrist": Keypoint(x + 55, 180, .99),
                   "left_hip": Keypoint(x + 35, 260, .99),
                   "right_hip": Keypoint(x + 85, 260, .99)},
        identity_state=identity, identity_confidence=.95, identity_margin=.4,
        review_status="AUTO_CONFIRMED", scene_state="ACTIVE_FIGHT",
    ) for role, identity, x in (("fighter_a", "FIGHTER_A", 160), ("fighter_b", "FIGHTER_B", 350))]


def hit() -> PunchEvent:
    return PunchEvent(event_id="hit", round=1, start_ms=850, peak_ms=1000, end_ms=1150,
        attacker_id="fighter_a", defender_id="fighter_b", hand="left", technique="hook",
        target="body", outcome="likely_landed", confidence=.9, impact_proxy_0_100=65)


def test_compact_preview_snapshot_has_tracking_but_no_hud_and_export_is_unchanged() -> None:
    frame = np.full((360, 640, 3), 40, np.uint8)
    context = RenderFrameContext(1000, 0, False, "ACTIVE_FIGHT")
    previews = []
    exported = FrameRenderer(hud_mode="compact").draw(frame, fighters(), [hit()], timestamp_ms=1000,
        frame_context=context, tracking_frame_callback=previews.append)
    expected_export = FrameRenderer(hud_mode="compact").draw(frame, fighters(), [hit()], timestamp_ms=1000,
        frame_context=context)
    clean = FrameRenderer(hud_mode="compact").draw(frame, fighters(), [hit()], timestamp_ms=1000,
        frame_context=context, include_hud=False)
    assert len(previews) == 1
    assert np.array_equal(exported, expected_export)
    assert np.array_equal(previews[0], clean)
    assert np.array_equal(clean[:70], frame[:70]), "no timer or stat-card background in player"
    assert np.count_nonzero(clean[90:335] != frame[90:335]) > 100, "fighter tracking must remain visible"
    assert not np.array_equal(exported[:70], clean[:70]), "export retains its HUD"


def test_workspace_snapshot_cannot_mutate_export_frame() -> None:
    frame = np.full((360, 640, 3), 40, np.uint8)
    expected = FrameRenderer(hud_mode="compact").draw(frame, fighters(), timestamp_ms=1000)
    actual = FrameRenderer(hud_mode="compact").draw(frame, fighters(), timestamp_ms=1000,
        tracking_frame_callback=lambda snapshot: snapshot.fill(0))
    assert np.array_equal(actual, expected)
    assert np.all(frame == 40)


def test_clean_preview_does_not_show_paused_banners_or_stale_events_during_break() -> None:
    frame = np.full((360, 640, 3), 40, np.uint8)
    context = RenderFrameContext(1000, 0, False, "BREAK")
    clean = FrameRenderer(hud_mode="compact").draw(frame, fighters(), [hit()], timestamp_ms=1000,
        frame_context=context, include_hud=False)
    assert np.array_equal(clean, frame)


@pytest.mark.parametrize("mode", ["technical", "none"])
def test_compact_snapshot_api_cannot_silently_capture_technical_event_labels(mode) -> None:
    with pytest.raises(ValueError, match="compact renderer"):
        FrameRenderer(hud_mode=mode).draw(np.zeros((32, 32, 3), np.uint8), [], tracking_frame_callback=lambda _: None)


def test_ui_prefers_clean_player_media_but_exports_annotated_and_supports_legacy(tmp_path: Path) -> None:
    annotated = tmp_path / "annotated.mp4"
    annotated.write_bytes(b"export")
    preview = tmp_path / "workspace-preview.mp4"
    (tmp_path / "events.json").write_text("[]")
    (tmp_path / "summary.json").write_text(json.dumps({"metadata": {"duration_s": 1}}))
    assert ui._load_existing_run(tmp_path)["workspace_video"] == str(annotated)
    preview.touch()
    assert ui._workspace_video_path(tmp_path, annotated) == annotated
    preview.write_bytes(b"tracking preview")
    state = ui._load_existing_run(tmp_path)
    assert state["workspace_video"] == str(preview)
    assert state["annotated_video"] == str(annotated)
    mp4_download, _ = ui._available_export_files(state)
    assert mp4_download["value"] == str(annotated)


def streams(path: Path) -> list[dict]:
    result = subprocess.run([str(FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"]


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="ffmpeg / ffprobe required")
@pytest.mark.parametrize("mode", ["compact", "technical", "none"])
def test_pipeline_and_cache_rebuild_make_synced_clean_h264_without_second_decode_or_ml(
    tmp_path: Path, monkeypatch, mode,
) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=0x455565:s=240x160:r=30:d=0.6",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)], check=True)
    result = pipeline.analyze_video(source, AnalysisConfig(backend="none", scheduled_rounds=1,
        round_length_s=60, output_height=360, analysis_fps=5, hud_mode=mode), runs_dir=tmp_path / "runs")
    preview = result.run_dir / "workspace-preview.mp4"
    assert preview.is_file()
    assert result.summary["metadata"]["workspace_preview"] == preview.name
    assert result.summary["metadata"]["workspace_preview_mode"] == "tracking_only"

    def forbid_ml(*args, **kwargs):
        raise AssertionError("render-only update must not invoke ML")

    monkeypatch.setattr(pipeline, "create_pose_backend", forbid_ml)
    monkeypatch.setattr(pipeline, "detect_punch_events", forbid_ml)
    original_iter = pipeline.iter_video_frames
    decodes = []

    def count_decode(path, *args, **kwargs):
        decodes.append(path)
        yield from original_iter(path, *args, **kwargs)

    monkeypatch.setattr(pipeline, "iter_video_frames", count_decode)
    assert pipeline.rebuild_from_cache(result.run_dir) == result.annotated_video
    assert decodes == [result.run_dir / ".render_cache" / "normalized.mp4"]
    output_streams = streams(preview)
    video = next(stream for stream in output_streams if stream["codec_type"] == "video")
    audio = next(stream for stream in output_streams if stream["codec_type"] == "audio")
    assert video["codec_name"] == "h264" and video["pix_fmt"] == "yuv420p"
    assert audio["codec_name"] == "aac"
    assert abs(float(video["duration"]) - float(audio["duration"])) <= 1 / 30
    export_video = next(stream for stream in streams(result.annotated_video) if stream["codec_type"] == "video")
    assert video["nb_frames"] == export_video["nb_frames"]
    # There are no confirmed people in this diagnostic fixture. Clean preview
    # must therefore preserve the source, not burn the export's warning/cards.
    capture = cv2.VideoCapture(str(preview))
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert np.max(np.std(frame.astype(float), axis=(0, 1))) < 2
    assert not list((result.run_dir / ".render_cache").glob(".rebuild-*.mp4"))
