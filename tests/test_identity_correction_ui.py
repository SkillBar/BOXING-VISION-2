from types import SimpleNamespace

import gradio as gr
import numpy as np
import pytest

from boxing_vision import cache_review, ui


@pytest.fixture
def correction_fixture(monkeypatch, tmp_path):
    payload = {"run_dir": str(tmp_path), "events": [], "summary": {}, "duration_s": 12,
               "events_path": str(tmp_path / "events.json"), "summary_path": str(tmp_path / "summary.json")}
    context = {"run_dir": str(tmp_path), "timestamp_ms": 1200, "shot_id": 2, "width": 200, "height": 100,
               "candidates": [{"source_track_id": 9, "segment_id": "shot-2-track-9-segment-1",
                               "bbox": {"x1": 10, "y1": 5, "x2": 80, "y2": 95}},
                              {"source_track_id": 11, "segment_id": "shot-2-track-11-segment-1",
                               "bbox": [110, 5, 180, 95]}]}
    calls = []
    monkeypatch.setattr(cache_review, "correct_identity_at", lambda run, **kwargs: calls.append((run, kwargs)), raising=False)
    monkeypatch.setattr(ui, "_load_existing_run", lambda _: payload)
    monkeypatch.setattr(ui, "_workspace", lambda *args: "workspace")
    monkeypatch.setattr(ui, "_overview", lambda *args: "overview")
    monkeypatch.setattr(ui, "_event_rows", lambda *args: [])
    return payload, context, calls


def test_bbox_click_targets_exact_segment_and_moment_not_whole_source(correction_fixture):
    payload, context, calls = correction_fixture
    output = ui._select_identity_correction(context, "FIGHTER_B", payload,
                                            SimpleNamespace(index=(30, 30)), lambda *args, **kwargs: None)
    assert len(calls) == 1
    assert calls[0][1] == {"timestamp_ms": 1200, "source_track_id": 9, "identity_state": "FIGHTER_B",
                           "segment_id": "shot-2-track-9-segment-1"}
    assert output[4]["render_stale"] is True
    assert "identity_overrides" not in output[4]
    assert output[5] is None


@pytest.mark.parametrize("index,action", [((199, 99), "FIGHTER_A"), ((-1, 10), "FIGHTER_A"),
                                         ((float("nan"), 10), "FIGHTER_A"), ((20, 20), "OTHER")])
def test_invalid_click_never_calls_correction_backend(correction_fixture, index, action):
    payload, context, calls = correction_fixture
    with pytest.raises(gr.Error):
        ui._select_identity_correction(context, action, payload, SimpleNamespace(index=index))
    assert calls == []


def test_overlapping_detections_require_unambiguous_click(correction_fixture):
    payload, context, calls = correction_fixture
    context["candidates"][1]["bbox"] = [20, 10, 100, 95]
    with pytest.raises(gr.Error, match="перекрываются"):
        ui._select_identity_correction(context, "FIGHTER_A", payload, SimpleNamespace(index=(30, 30)))
    assert calls == []


def test_frame_from_another_run_cannot_be_applied(correction_fixture):
    payload, context, calls = correction_fixture
    context["run_dir"] += "-other"
    with pytest.raises(gr.Error, match="изменился"):
        ui._select_identity_correction(context, "FIGHTER_A", payload, SimpleNamespace(index=(30, 30)))
    assert calls == []


def test_frame_preview_uses_cached_geometry_without_inference(monkeypatch, correction_fixture):
    payload, context, _ = correction_fixture
    requested, captured = [], []
    monkeypatch.setattr(cache_review, "get_identity_correction_frame",
                        lambda run, stamp: requested.append(stamp) or context, raising=False)

    class Capture:
        def __init__(self, path):
            captured.append(path)

        def set(self, key, value):
            captured.append(value)

        def read(self):
            return True, np.zeros((100, 200, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(ui.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(ui, "calibration_backend", lambda: pytest.fail("No model should run for a correction preview"))
    image, state, note = ui._load_identity_correction_frame(payload, 1.23)
    assert requested == [1230]
    assert captured[1] == 1200  # Actual cached timestamp, not requested time.
    assert image.shape == (100, 200, 3)
    assert image.any()
    assert state["candidates"] == context["candidates"]
    assert "не ко всему ID" in note
