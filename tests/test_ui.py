from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from boxing_vision.ui import (
    _apply_anchor_selection,
    _confirmed_anchors,
    _empty_anchor_state,
    _parse_round_list,
    _prepare_confirmation,
    build_app,
)
from boxing_vision.video import normalize_video

FFMPEG = shutil.which("ffmpeg")


def test_confirmation_selects_two_normalized_fighter_anchors() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    state = _empty_anchor_state()
    state["base_image"] = image.copy()

    rendered, state, _ = _apply_anchor_selection(image, state, (40, 50))
    rendered, state, message = _apply_anchor_selection(rendered, state, (160, 45))
    fighter_a, fighter_b = _confirmed_anchors(state)

    assert rendered.shape == image.shape
    assert fighter_a == (40 / 199, 50 / 99)
    assert fighter_b == (160 / 199, 45 / 99)
    assert "подтверждены" in message


def test_knockdown_round_parser_supports_repeated_rounds() -> None:
    assert _parse_round_list("2, 3; 3", 4) == (2, 3, 3)


def test_gradio_app_builds_with_private_analytics_disabled() -> None:
    app = build_app()
    assert app.analytics_enabled is False
    fields = {item["props"].get("label"): item["props"] for item in app.config["components"]}
    assert fields["Где искать бойцов"]["value"] == "none"
    assert fields["Режим времени"]["value"] == "continuous"
    assert fields["Что исправить кликом"]["choices"] == [("Красный A", "fighter_a"), ("Синий B", "fighter_b")]
    assert "Исправление только выбранного сегмента" in fields


@pytest.mark.parametrize("rotation", [90, 180, 270])
@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg unavailable")
def test_confirmation_orientation_matches_normalized_video(
    tmp_path: Path,
    rotation: int,
) -> None:
    base = tmp_path / "base.mp4"
    rotated = tmp_path / f"rotated-{rotation}.mp4"
    normalized = tmp_path / "normalized.mp4"
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
            "testsrc2=size=160x96:rate=5:duration=0.8",
            "-c:v",
            "mpeg4",
            str(base),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-display_rotation",
            str(rotation),
            "-i",
            str(base),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )

    _, preview, _, _ = _prepare_confirmation(str(rotated), 0)
    normalize_video(rotated, normalized, output_fps=5)
    capture = cv2.VideoCapture(str(normalized))
    ok, normalized_bgr = capture.read()
    capture.release()
    assert ok and preview is not None
    normalized_rgb = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)

    assert preview.shape == normalized_rgb.shape
    assert np.mean(np.abs(preview.astype(float) - normalized_rgb.astype(float))) < 3.0
