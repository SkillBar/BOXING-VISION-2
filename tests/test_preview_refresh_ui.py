"""Preview refresh stays available while review/export gates remain closed."""
import json
from copy import deepcopy
from pathlib import Path

import gradio as gr
import pytest
from gradio.processing_utils import save_file_to_cache

from boxing_vision import pipeline, ui


@pytest.fixture
def pending_preview(monkeypatch, tmp_path):
    summary = {"metadata": {"export_stale": True, "tracking_preview_stale": True},
               "quality": {"required_review_count": 6}}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    payload = {"run_dir": str(tmp_path), "duration_s": 12, "events": [], "summary": summary,
               "render_stale": True, "tracking_preview_stale": True,
               "summary_path": str(summary_path), "workspace_video": str(tmp_path / "workspace-preview.mp4")}
    monkeypatch.setattr(ui, "_workspace", lambda *args: "workspace")
    monkeypatch.setattr(ui, "_overview", lambda *args: "overview")
    monkeypatch.setattr(ui, "_event_rows", lambda *args: [])
    monkeypatch.setattr(ui, "rebuild_from_cache", lambda *a, **kw: pytest.fail("Final export must not run"))
    monkeypatch.setattr(pipeline, "rebuild_from_cache", lambda *a, **kw: pytest.fail("Final export must not run"))
    monkeypatch.setattr(pipeline, "create_pose_backend", lambda *a, **kw: pytest.fail("ML must not run"))
    return payload


def test_preview_only_refresh_is_allowed_with_required_reviews(monkeypatch, pending_preview):
    payload = pending_preview
    calls = []
    refreshed = deepcopy(payload)
    refreshed["summary"]["metadata"]["tracking_preview_stale"] = False
    refreshed["tracking_preview_stale"] = False

    def rebuild(run, progress_callback, *, cancel_callback):
        calls.append((run, cancel_callback()))
        progress_callback(0.5, "Render only")
        return run / "workspace-preview.mp4"

    monkeypatch.setattr(pipeline, "rebuild_tracking_preview_from_cache", rebuild)
    monkeypatch.setattr(ui, "_load_existing_run", lambda _: refreshed)
    ui._CANCEL_EVENT.set()  # A previous cancelled job must not cancel this one.
    output = ui._refresh_tracking_player(payload, lambda *a, **kw: None)
    assert calls == [(Path(payload["run_dir"]), False)]
    assert output[1] == payload["workspace_video"]
    assert output[5]["tracking_preview_stale"] is False
    assert output[5]["render_stale"] is True
    assert output[5]["summary"]["metadata"]["export_stale"] is True
    assert output[5]["summary"]["quality"]["required_review_count"] == 6
    assert output[6] is None  # No ZIP download.
    assert "финальный MP4 и ZIP пока не пересобраны" in output[0]
    # Caller-owned state is not changed in place before Gradio commits outputs.
    assert payload["tracking_preview_stale"] is True


def test_preview_failure_keeps_saved_correction_and_export_stale(monkeypatch, pending_preview):
    payload = pending_preview
    original = deepcopy(payload)
    summary_path = Path(payload["summary_path"])
    before = summary_path.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(pipeline, "rebuild_tracking_preview_from_cache", fail)
    monkeypatch.setattr(ui, "_load_existing_run", lambda _: pytest.fail("Must not load a failed render as current"))
    with pytest.raises(gr.Error, match="encoder unavailable"):
        ui._refresh_tracking_player(payload, lambda *a, **kw: None)
    assert payload == original
    assert summary_path.read_bytes() == before
    assert payload["summary"]["metadata"]["export_stale"] is True
    assert payload["tracking_preview_stale"] is True


def test_preview_refresh_without_result_never_invokes_backend(monkeypatch):
    monkeypatch.setattr(pipeline, "rebuild_tracking_preview_from_cache", lambda *a, **kw: pytest.fail("No run"))
    with pytest.raises(gr.Error, match="Сначала откройте"):
        ui._refresh_tracking_player(None)


def test_preview_status_distinguishes_fresh_player_from_stale_export(pending_preview):
    note, update = ui._tracking_preview_status(pending_preview)
    assert "прежние рамки" in note and update["interactive"] is True
    pending_preview["tracking_preview_stale"] = False
    note, update = ui._tracking_preview_status(pending_preview)
    assert "плеере актуален" in note and update["interactive"] is True
    assert pending_preview["summary"]["metadata"]["export_stale"] is True
    _, update = ui._tracking_preview_status(None)
    assert update["interactive"] is False


def test_gradio_content_cache_changes_video_url_for_same_run_filename(tmp_path):
    source = tmp_path / "workspace-preview.mp4"
    cache = tmp_path / "gradio-cache"
    source.write_bytes(b"old rendered frames")
    old_path = save_file_to_cache(source, str(cache))
    source.write_bytes(b"new corrected tracking frames")
    new_path = save_file_to_cache(source, str(cache))
    assert old_path != new_path
    assert Path(old_path).read_bytes() == b"old rendered frames"
    assert Path(new_path).read_bytes() == b"new corrected tracking frames"


def test_refresh_button_updates_actual_gradio_video_not_only_workspace():
    app = ui.build_app()
    components = app.config["components"]
    button = next(item for item in components if item["type"] == "button"
                  and item["props"].get("value") == "Обновить трекинг в плеере")
    video = next(item for item in components if item["props"].get("elem_id") == "annotated-video")
    callback = next(item for item in app.config["dependencies"]
                    if [button["id"], "click"] in [list(target) for target in item["targets"]])
    assert callback["outputs"][1] == video["id"]
