from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from boxing_vision.config import AnalysisConfig
from boxing_vision.pipeline import AnalysisCancelledError, analyze_video
from boxing_vision.video import VideoValidationError

FFMPEG = shutil.which("ffmpeg")


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg unavailable")
def test_pipeline_backend_none_creates_complete_browser_artifacts(tmp_path: Path) -> None:
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
    )

    result = analyze_video(source, config, runs_dir=tmp_path / "runs")

    assert result.annotated_video.is_file()
    assert result.events_path.is_file()
    assert result.summary_path.is_file()
    assert result.log_path.is_file()
    assert result.events == []
    assert result.summary["metadata"]["backend"] == "unavailable"


def test_pipeline_honors_immediate_cancellation_before_creating_job(tmp_path: Path) -> None:
    with pytest.raises(AnalysisCancelledError):
        analyze_video(
            tmp_path / "missing.mp4",
            AnalysisConfig(),
            runs_dir=tmp_path / "runs",
            cancel_callback=lambda: True,
        )
    assert not (tmp_path / "runs").exists()


def test_pipeline_failure_removes_large_work_files_but_keeps_log(tmp_path: Path) -> None:
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
