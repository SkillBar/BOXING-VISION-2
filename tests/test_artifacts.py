from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from boxing_vision.artifacts import (
    atomic_write_json,
    create_job_artifacts,
    extract_event_clips,
    finalize_h264_video,
    persist_fighter_portrait,
)
from boxing_vision.contracts import PunchEvent
from boxing_vision.video import normalize_video

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _event(event_id: str = "event-001") -> PunchEvent:
    return PunchEvent(
        event_id=event_id,
        round=1,
        start_ms=2400,
        peak_ms=2500,
        end_ms=2600,
        attacker_id="fighter_a",
        defender_id="fighter_b",
        hand="left",
        technique="jab",
        target="head",
        outcome="likely_landed",
        confidence=0.82,
        impact_proxy_0_100=64,
    )


def _ffmpeg(*arguments: str) -> None:
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _streams(path: Path) -> list[dict[str, object]]:
    if FFPROBE is None:
        pytest.skip("ffprobe is not installed")
    completed = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)["streams"]


def _duration(path: Path) -> float:
    if FFPROBE is None:
        pytest.skip("ffprobe is not installed")
    completed = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def _probe_payload(path: Path) -> dict[str, object]:
    if FFPROBE is None:
        pytest.skip("ffprobe is not installed")
    completed = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _make_video(path: Path, *, duration: float, audio: bool) -> None:
    arguments = [
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=96x64:rate=10:duration={duration}",
    ]
    if audio:
        arguments.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=44100:duration={duration}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:a",
                "aac",
                "-shortest",
            ]
        )
    arguments.extend(["-c:v", "mpeg4", "-q:v", "5", str(path)])
    _ffmpeg(*arguments)


def test_create_job_artifacts_builds_isolated_contract(tmp_path: Path) -> None:
    artifacts = create_job_artifacts(tmp_path / "runs", "demo-001")

    assert artifacts.job_id == "demo-001"
    assert artifacts.run_dir.is_dir()
    assert artifacts.clips_dir.is_dir()
    assert artifacts.work_dir.is_dir()
    assert artifacts.silent_video == artifacts.work_dir / "annotated_silent.mp4"
    assert artifacts.annotated_video == artifacts.run_dir / "annotated.mp4"
    assert artifacts.events_path == artifacts.run_dir / "events.json"
    assert artifacts.summary_path == artifacts.run_dir / "summary.json"
    assert artifacts.log_path == artifacts.run_dir / "analysis.log"

    with pytest.raises(FileExistsError):
        create_job_artifacts(tmp_path / "runs", "demo-001")


@pytest.mark.parametrize("unsafe_id", ["../escape", "nested/job", "", "a b", ".."])
def test_create_job_artifacts_rejects_unsafe_ids(
    tmp_path: Path, unsafe_id: str
) -> None:
    with pytest.raises(ValueError):
        create_job_artifacts(tmp_path, unsafe_id)


def test_atomic_write_json_handles_unicode_paths_and_dataclasses(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "events.json"
    payload = {
        "title": "Красный угол",
        "asset": tmp_path / "video.mp4",
        "event": _event(),
    }

    returned = atomic_write_json(target, payload)

    assert returned == target
    decoded = json.loads(target.read_text(encoding="utf-8"))
    assert decoded["title"] == "Красный угол"
    assert decoded["asset"] == str(tmp_path / "video.mp4")
    assert decoded["event"]["event_id"] == "event-001"
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))

    atomic_write_json(target, {"replaced": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"replaced": True}


def test_persist_fighter_portrait_crops_to_private_run_relative_webp(
    tmp_path: Path,
) -> None:
    source = tmp_path / "private-original-name.png"
    Image.new("RGB", (480, 240), (220, 40, 40)).save(source)
    run_dir = tmp_path / "runs" / "job-001"
    run_dir.mkdir(parents=True)

    stored = persist_fighter_portrait(source, run_dir, "fighter_a")

    assert stored == "profiles/fighter_a.webp"
    destination = run_dir / stored
    assert destination.is_file()
    with Image.open(destination) as portrait:
        assert portrait.size == (256, 256)
        assert portrait.mode == "RGB"
    assert source.name not in stored
    assert not list((run_dir / "profiles").glob("*.tmp.webp"))


def test_persist_fighter_portrait_rejects_invalid_input_without_partial_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "not-an-image.txt"
    source.write_text("not an image", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="портрета"):
        persist_fighter_portrait(source, run_dir, "fighter_b")

    assert not (run_dir / "profiles" / "fighter_b.webp").exists()
    assert not list((run_dir / "profiles").glob("*.tmp.webp"))


@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe unavailable"
)
def test_finalize_h264_preserves_audio_and_accepts_silent_source(
    tmp_path: Path,
) -> None:
    annotated = tmp_path / "annotated-input.mp4"
    source_audio = tmp_path / "source-audio.mp4"
    source_silent = tmp_path / "source-silent.mp4"
    _make_video(annotated, duration=0.8, audio=False)
    _make_video(source_audio, duration=0.8, audio=True)
    _make_video(source_silent, duration=0.8, audio=False)

    with_audio = finalize_h264_video(
        annotated, source_audio, tmp_path / "with-audio.mp4"
    )
    silent = finalize_h264_video(annotated, source_silent, tmp_path / "silent.mp4")

    with_audio_streams = _streams(with_audio)
    silent_streams = _streams(silent)
    assert any(
        stream["codec_type"] == "video" and stream["codec_name"] == "h264"
        for stream in with_audio_streams
    )
    assert any(stream["codec_type"] == "audio" for stream in with_audio_streams)
    assert any(
        stream["codec_type"] == "video" and stream["codec_name"] == "h264"
        for stream in silent_streams
    )
    assert not any(stream["codec_type"] == "audio" for stream in silent_streams)


@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe unavailable"
)
def test_finalize_h264_does_not_truncate_video_when_audio_ends_early(
    tmp_path: Path,
) -> None:
    annotated = tmp_path / "annotated-3s.mp4"
    source = tmp_path / "short-audio.mp4"
    _make_video(annotated, duration=3.0, audio=False)
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=96x64:rate=10:duration=3",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=1",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        str(source),
    )

    output = finalize_h264_video(annotated, source, tmp_path / "final.mp4")

    assert 2.9 <= _duration(output) <= 3.1
    assert any(stream["codec_type"] == "audio" for stream in _streams(output))


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe unavailable")
@pytest.mark.parametrize("audio_duration", [0.3, 0.6, 1.2])
def test_mux_bounds_audio_to_rendered_frames_not_unbounded_padding(tmp_path, audio_duration):
    annotated, source = tmp_path / "picture.mp4", tmp_path / "audio.mp4"
    _ffmpeg("-f", "lavfi", "-i", "color=c=black:s=96x64:r=30:d=0.6",
            "-c:v", "libx264", str(annotated))
    _make_video(source, duration=audio_duration, audio=True)
    for index in range(2):
        output = finalize_h264_video(annotated, source, tmp_path / f"mux-{index}.mp4")
        streams = _probe_payload(output)["streams"]
        video = next(s for s in streams if s["codec_type"] == "video")
        audio = next(s for s in streams if s["codec_type"] == "audio")
        assert int(video["nb_frames"]) == 18
        assert abs(float(video["duration"]) - float(audio["duration"])) <= 1 / 30


@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe unavailable"
)
def test_exported_media_strips_private_source_metadata(tmp_path: Path) -> None:
    plain = tmp_path / "plain.mp4"
    tagged = tmp_path / "tagged.mp4"
    normalized = tmp_path / "normalized.mp4"
    annotated = tmp_path / "annotated.mp4"
    final = tmp_path / "final.mp4"
    _make_video(plain, duration=1.0, audio=True)
    _ffmpeg(
        "-i",
        str(plain),
        "-c",
        "copy",
        "-metadata",
        "title=BOXING_PRIVATE_TITLE_94C7",
        "-metadata",
        "comment=BOXING_PRIVATE_GPS_40_177",
        "-movflags",
        "use_metadata_tags",
        str(tagged),
    )
    source_payload = json.dumps(_probe_payload(tagged), ensure_ascii=False)
    assert "BOXING_PRIVATE_TITLE_94C7" in source_payload
    assert "BOXING_PRIVATE_GPS_40_177" in source_payload

    normalize_video(tagged, normalized, output_fps=10)
    _make_video(annotated, duration=1.0, audio=False)
    finalize_h264_video(annotated, tagged, final)
    clip = extract_event_clips(tagged, [_event()], tmp_path / "clips")['event-001']

    for exported in (normalized, final, clip):
        payload = json.dumps(_probe_payload(exported), ensure_ascii=False)
        assert "BOXING_PRIVATE_TITLE_94C7" not in payload
        assert "BOXING_PRIVATE_GPS_40_177" not in payload


@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe unavailable"
)
def test_extract_event_clips_uses_two_second_padding_and_updates_event(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    _make_video(source, duration=5.0, audio=True)
    event = _event()

    paths = extract_event_clips(source, [event], tmp_path / "clips")

    clip = paths[event.event_id]
    assert clip.is_file()
    assert event.clip_path == f"clips/{clip.name}"
    assert 3.9 <= _duration(clip) <= 4.5
    streams = _streams(clip)
    assert any(
        stream["codec_type"] == "video" and stream["codec_name"] == "h264"
        for stream in streams
    )
    assert any(stream["codec_type"] == "audio" for stream in streams)
