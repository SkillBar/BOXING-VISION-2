from __future__ import annotations

import io
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from boxing_vision import artifacts, media_process, previews, video


class BlockingPipe:
    """An anonymous-pipe stand-in: deliberately has no selectable descriptor."""

    def __init__(self):
        self.read_started = threading.Event()
        self.released = threading.Event()
        self.closed = False

    def readline(self, _limit=-1):
        self.read_started.set()
        assert self.released.wait(timeout=5), "reader was not released by child exit"
        return ""

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, output="", *, running=False, stubborn=False):
        self.stdout = output if not isinstance(output, str) else io.StringIO(output)
        self.returncode = None if running else 0
        self.stubborn = stubborn
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def _finish(self, code):
        self.returncode = code
        if isinstance(self.stdout, BlockingPipe):
            self.stdout.released.set()

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self._finish(-15)

    def kill(self):
        self.killed = True
        self._finish(-9)

    def wait(self, timeout=None):
        self.waited = True
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake ffmpeg", timeout)
        return self.returncode

    def communicate(self, timeout=None):
        self.wait(timeout)
        return "", ""


def setup_normalizer(monkeypatch, tmp_path, process):
    source = tmp_path / "исходное видео.mp4"
    source.write_bytes(b"source")
    metadata = video.VideoMetadata(
        source, 2.0, 640, 360, 30, 60, "h264", False, None, 0, 6
    )
    monkeypatch.setattr(video, "validate_video", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(video.shutil, "which", lambda executable: str(executable))
    launches = []

    def launch(command, **kwargs):
        launches.append(kwargs)
        Path(command[-1]).write_bytes(b"normalized")
        return process

    monkeypatch.setattr(video.subprocess, "Popen", launch)
    return source, tmp_path / "result.mp4", launches


def test_windows_normalization_reads_nonselectable_pipe_and_drains_progress(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(media_process, "sys", SimpleNamespace(platform="win32"))
    process = FakeProcess("out_time_us=500000\nout_time_ms=1000000\nprogress=end\n")
    source, destination, launches = setup_normalizer(monkeypatch, tmp_path, process)
    progress = []
    assert (
        video.normalize_video(source, destination, progress_callback=progress.append)
        == destination
    )
    assert progress == [0.25, 0.5, 1.0]
    assert process.stdout.closed and process.waited
    assert launches[0]["creationflags"] == 0x08000000
    assert launches[0]["stdin"] == subprocess.DEVNULL
    assert destination.read_bytes() == b"normalized"


@pytest.mark.parametrize("stubborn", [False, True])
def test_cancellation_reaps_silent_pipe_reader_and_removes_partial_output(
    monkeypatch, tmp_path, stubborn
):
    pipe = BlockingPipe()
    process = FakeProcess(pipe, running=True, stubborn=stubborn)
    source, destination, _ = setup_normalizer(monkeypatch, tmp_path, process)
    started = time.monotonic()
    with pytest.raises(video.VideoNormalizationError, match="отменена"):
        video.normalize_video(
            source, destination, cancel_callback=pipe.read_started.is_set
        )
    assert time.monotonic() - started < 2
    assert process.terminated and process.waited and process.poll() is not None
    assert process.killed is stubborn
    assert pipe.closed and not destination.exists()
    assert not destination.with_name(".result.part.mp4").exists()
    assert not any(
        t.name == "boxing-ffmpeg-progress" and t.is_alive()
        for t in threading.enumerate()
    )


def test_closed_progress_pipe_does_not_prevent_cancelling_still_running_child(
    monkeypatch, tmp_path
):
    process = FakeProcess("", running=True)
    source, destination, _ = setup_normalizer(monkeypatch, tmp_path, process)
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(video.VideoNormalizationError, match="отменена"):
        video.normalize_video(source, destination, cancel_callback=cancel)
    assert process.terminated and process.stdout.closed


def test_progress_callback_exception_stops_child_and_bounded_queue_reader(
    monkeypatch, tmp_path
):
    process = FakeProcess("out_time_us=10000\n" * 5000, running=True)
    source, destination, _ = setup_normalizer(monkeypatch, tmp_path, process)

    def fail(_value):
        raise ValueError("callback failure")

    with pytest.raises(ValueError, match="callback failure"):
        video.normalize_video(source, destination, progress_callback=fail)
    assert process.terminated and process.stdout.closed
    assert not destination.with_name(".result.part.mp4").exists()


def test_failed_encoder_retains_last_diagnostic_after_large_output(
    monkeypatch, tmp_path
):
    process = FakeProcess("diagnostic\n" * 500 + "final codec failure\n")
    process.returncode = 1
    source, destination, _ = setup_normalizer(monkeypatch, tmp_path, process)
    with pytest.raises(video.VideoNormalizationError, match="final codec failure"):
        video.normalize_video(source, destination)
    assert process.stdout.closed and not destination.exists()


def test_pipe_read_error_stops_encoder_and_propagates_diagnostic(monkeypatch, tmp_path):
    class BrokenPipe:
        closed = False

        def readline(self, _limit=-1):
            raise OSError("broken progress pipe")

        def close(self):
            self.closed = True

    process = FakeProcess(BrokenPipe(), running=True)
    source, destination, _ = setup_normalizer(monkeypatch, tmp_path, process)
    with pytest.raises(video.VideoNormalizationError, match="broken progress pipe"):
        video.normalize_video(source, destination)
    assert process.terminated and process.stdout.closed


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_non_windows_does_not_set_windows_creationflags(monkeypatch, platform):
    monkeypatch.setattr(media_process, "sys", SimpleNamespace(platform=platform))
    assert media_process.hidden_process_kwargs() == {}


def test_probe_and_export_use_same_windows_console_suppression(monkeypatch, tmp_path):
    monkeypatch.setattr(media_process, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(video.shutil, "which", lambda executable: str(executable))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 640,
                            "height": 360,
                            "duration": "2",
                            "avg_frame_rate": "30/1",
                        }
                    ]
                }
            ),
            "",
        )

    monkeypatch.setattr(video.subprocess, "run", run)
    assert video.probe_video(source).width == 640
    monkeypatch.setattr(artifacts, "_resolve_executable", lambda value: str(value))

    def launch(command, **kwargs):
        calls.append(kwargs)
        return FakeProcess()

    monkeypatch.setattr(artifacts.subprocess, "Popen", launch)
    assert artifacts._run_ffmpeg(["-version"]).returncode == 0
    assert len(calls) == 2 and all(
        call["creationflags"] == 0x08000000 for call in calls
    )


def test_preview_generation_uses_windows_console_suppression(monkeypatch, tmp_path):
    monkeypatch.setattr(media_process, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(previews.shutil, "which", lambda executable: str(executable))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = []

    def launch(command, **kwargs):
        calls.append(kwargs)
        process = FakeProcess(io.BytesIO(bytes(12)))
        process.stderr = io.BytesIO(b"")
        return process

    monkeypatch.setattr(previews.subprocess, "Popen", launch)
    result = previews.generate_hover_previews(
        source, tmp_path / "run", tile_width=2, tile_height=2
    )
    assert json.loads(result.read_text())["frames"][0]["time_ms"] == 0
    assert calls[0]["creationflags"] == 0x08000000
