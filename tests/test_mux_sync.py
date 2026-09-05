from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from boxing_vision import artifacts, video

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def test_audio_padding_has_explicit_end_independent_of_shortest_scheduler(tmp_path, monkeypatch):
    silent, source, output = (tmp_path / name for name in ("silent.mp4", "source.mp4", "output.mp4"))
    silent.write_bytes(b"fixture")
    source.write_bytes(b"fixture")
    probes, commands = [], []

    def probe(path, **_kwargs):
        probes.append(path)
        return SimpleNamespace(duration_s=.6)

    def encode(arguments, **_kwargs):
        commands.append([str(value) for value in arguments])
        Path(arguments[-1]).write_bytes(b"completed")

    monkeypatch.setattr(video, "probe_video", probe)
    monkeypatch.setattr(artifacts, "_run_ffmpeg", encode)
    artifacts.finalize_h264_video(silent, source, output)
    assert probes == [silent], "Use rendered video length, never the source audio/container length"
    command = commands[0]
    assert "-shortest" not in command
    assert command[command.index("-af") + 1] == "apad=whole_dur=0.600000000,atrim=end=0.600000000"
    assert command[command.index("-t") + 1] == "0.600000000"


@pytest.mark.parametrize("duration", [0, float("nan"), float("inf")])
def test_invalid_rendered_duration_preserves_existing_output(tmp_path, monkeypatch, duration):
    silent, source, output = (tmp_path / name for name in ("silent.mp4", "source.mp4", "output.mp4"))
    silent.write_bytes(b"fixture")
    source.write_bytes(b"fixture")
    output.write_bytes(b"previous-working-export")
    monkeypatch.setattr(video, "probe_video", lambda *_args, **_kwargs: SimpleNamespace(duration_s=duration))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Invalid duration must fail before encoding")

    monkeypatch.setattr(artifacts, "_run_ffmpeg", unexpected)
    with pytest.raises(ValueError, match="длительность"):
        artifacts.finalize_h264_video(silent, source, output)
    assert output.read_bytes() == b"previous-working-export"


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="ffmpeg/ffprobe required")
@pytest.mark.parametrize("audio_duration,rate", [(.15, 48000), (.6, 48000), (1.2, 48000), (.6, 44100)])
def test_repeated_mux_keeps_every_frame_and_audio_within_one_frame(tmp_path, audio_duration, rate):
    source, silent = tmp_path / "source.mp4", tmp_path / "silent.mp4"
    subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=0x455565:s=240x160:r=30:d=0.6",
                    "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={rate}:duration={audio_duration}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)], check=True)
    subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-map", "0:v:0", "-c:v", "copy", "-an", str(silent)], check=True)
    for attempt in range(6):
        output = artifacts.finalize_h264_video(silent, source, tmp_path / f"mux-{attempt}.mp4")
        completed = subprocess.run([str(FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(output)],
                                   check=True, capture_output=True, text=True)
        streams = json.loads(completed.stdout)["streams"]
        rendered = next(stream for stream in streams if stream["codec_type"] == "video")
        audio = next(stream for stream in streams if stream["codec_type"] == "audio")
        assert rendered["nb_frames"] == "18", f"Attempt {attempt} lost rendered frames"
        assert rendered["codec_name"] == "h264" and audio["codec_name"] == "aac"
        assert abs(float(rendered["duration"]) - .6) < 1e-6
        assert abs(float(rendered["duration"]) - float(audio["duration"])) <= 1 / 30, (attempt, rendered["duration"], audio["duration"])
        assert abs(float(rendered["start_time"]) - float(audio["start_time"])) <= 1 / 30
