from __future__ import annotations

from pathlib import Path

import pytest

from boxing_vision.config import AnalysisConfig


def test_analysis_config_defaults_to_compact_hud() -> None:
    config = AnalysisConfig()

    assert config.hud_mode == "compact"
    config.validate()


@pytest.mark.parametrize("hud_mode", ["compact", "technical", "none"])
def test_analysis_config_accepts_supported_hud_modes(hud_mode: str) -> None:
    AnalysisConfig(hud_mode=hud_mode).validate()


def test_analysis_config_rejects_unknown_hud_mode() -> None:
    with pytest.raises(ValueError, match="compact, technical или none"):
        AnalysisConfig(hud_mode="cinematic").validate()


def test_config_serialization_never_exposes_portrait_upload_paths(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "sensitive-a.png"
    source_b = tmp_path / "sensitive-b.png"
    payload = AnalysisConfig(
        fighter_a_portrait_path=source_a,
        fighter_b_portrait_path=str(source_b),
    ).to_dict()

    assert "fighter_a_portrait_path" not in payload
    assert "fighter_b_portrait_path" not in payload
    assert str(tmp_path) not in repr(payload)


def test_config_serializes_adaptive_identity_policy() -> None:
    payload = AnalysisConfig(
        adaptive_identity_confidence_min=0.84,
        adaptive_identity_margin_min=0.26,
    ).to_dict()

    assert payload["adaptive_identity_confidence_min"] == 0.84
    assert payload["adaptive_identity_margin_min"] == 0.26
