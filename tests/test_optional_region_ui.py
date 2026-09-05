from pathlib import Path

import gradio as gr
import numpy as np
import pytest

from boxing_vision import ui
from boxing_vision.config import AnalysisConfig

POLYGON = ((.05, .1), (.95, .1), (.95, .95), (.05, .95))


def state_for(source: Path, mode="none"):
    return {
        "source": str(source.resolve()), "start_s": 0, "active_index": 0,
        "region_mode": mode, "ring_points": [], "confirmed": False,
        "views": [{"image": np.zeros((100, 200, 3), np.uint8), "time_s": time,
                   "boxes": [[10, 5, 80, 95], [110, 5, 180, 95]],
                   "selection": {"fighter_a": 0, "fighter_b": 1}}
                  for time in (0, 3, 6)],
    }


def test_legacy_config_keeps_scheduled_timing_and_infers_existing_polygon():
    empty = AnalysisConfig()
    assert empty.timing_mode == "scheduled"
    assert empty.region_mode == "auto"
    assert empty.effective_region_mode == "none"
    assert empty.effective_ring_rois == ()
    selected = AnalysisConfig(ring_rois=POLYGON)
    selected.validate()
    assert selected.effective_region_mode == "manual"
    assert selected.effective_ring_rois == POLYGON
    assert selected.tracking_overlay_style == "full"
    assert selected.display_prediction_ms == 1000


def test_explicit_unrestricted_config_does_not_apply_old_polygon():
    config = AnalysisConfig(region_mode="none", ring_rois=POLYGON, timing_mode="continuous")
    config.validate()
    assert config.effective_ring_rois == ()
    assert config.to_dict()["timing_mode"] == "continuous"


def test_segment_overrides_are_separate_from_legacy_source_overrides():
    config = AnalysisConfig(segment_identity_overrides={"shot-2-track-9-segment-1": "FIGHTER_B"})
    config.validate()
    assert config.identity_overrides == {}
    assert config.to_dict()["segment_identity_overrides"] == {"shot-2-track-9-segment-1": "FIGHTER_B"}
    with pytest.raises(ValueError, match="сегмента"):
        AnalysisConfig(segment_identity_overrides=None).validate()


@pytest.mark.parametrize("points", [(), ((0, 0),) * 4,
                                      ((0, 0), (1, 1), (1, 0), (0, 1)),
                                      ((0, 0), (1, 0), (float("nan"), 1), (0, 1))])
def test_manual_region_must_be_four_finite_convex_points(points):
    with pytest.raises(ValueError):
        AnalysisConfig(region_mode="manual", ring_rois=points).validate()


@pytest.mark.parametrize("field,value", [("region_mode", "unexpected"), ("timing_mode", "random"),
                                        ("tracking_overlay_style", "glow"),
                                        ("display_prediction_ms", -1), ("display_prediction_ms", 1001),
                                        ("display_prediction_ms", float("nan")), ("display_prediction_ms", True)])
def test_config_rejects_unsupported_display_and_timing_options(field, value):
    with pytest.raises(ValueError):
        AnalysisConfig(**{field: value}).validate()


def test_new_enrollment_can_confirm_without_any_polygon(tmp_path):
    source = tmp_path / "fight.mp4"
    _, confirmed, _ = ui._confirm_enrollment_ui(state_for(source))
    fields = ui._enrollment_config_fields(confirmed, str(source), 0)
    assert fields["region_mode"] == "none"
    assert fields["ring_rois"] == ()
    assert len(fields["enrollment_samples"]) == 3
    assert ui._empty_enrollment_state()["region_mode"] == "none"


def test_manual_enrollment_still_requires_polygon_and_preserves_it(tmp_path):
    source = tmp_path / "fight.mp4"
    state = state_for(source, "manual")
    with pytest.raises(gr.Error, match="четырьмя"):
        ui._confirm_enrollment_ui(state)
    state["ring_points"] = POLYGON
    _, confirmed, _ = ui._confirm_enrollment_ui(state)
    fields = ui._enrollment_config_fields(confirmed, str(source), 0)
    assert fields["region_mode"] == "manual"
    assert fields["ring_rois"] == POLYGON


def test_switching_region_invalidates_confirmation_without_rerunning_ml(tmp_path):
    source = tmp_path / "fight.mp4"
    state = state_for(source, "manual")
    state["ring_points"] = POLYGON
    _, confirmed, _ = ui._confirm_enrollment_ui(state)
    _, changed, _, role = ui._change_working_region(confirmed, "none")
    assert not changed["confirmed"]
    assert changed["ring_points"] == []
    assert "enrollment_samples" not in changed
    assert "ring" not in [value for _, value in role["choices"]]
    assert confirmed["ring_points"] == POLYGON


def test_unrestricted_enrollment_cannot_accidentally_collect_ring_points(tmp_path):
    with pytest.raises(ValueError, match="Рабочая область"):
        ui._apply_enrollment_selection(state_for(tmp_path / "f.mp4"), "ring", (20, 20))


def test_motion_candidates_are_not_labelled_as_confirmed_hits():
    summary = {"quality": {"motion_proposals": 80, "unresolved_motion_proposals": 29, "event_candidates": 51},
               "fighters": {"fighter_a": {"likely_landed": 1}, "fighter_b": {"likely_landed": 0}}}
    rendered = ui._overview(summary)
    assert "Движения на проверку: 80" in rendered
    assert "без подтверждённой личности: 29" in rendered
    assert "Кандидаты ударов: 51" in rendered
    assert "вероятные попадания: 1" in rendered
