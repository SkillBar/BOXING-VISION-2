import json
from pathlib import Path

import pytest

from boxing_vision import desktop, ui
from boxing_vision.demo_bundle import prepare_demo, stage_demo, verify_demo


def source_run(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "workspace-preview.mp4").write_bytes(b"test media fixture, not playback proof")
    (root / "annotated.mp4").write_bytes(b"stale must not ship")
    (root / "events.json").write_text(json.dumps([{"event_id": "event", "peak_ms": 49933, "clip_path": "/private/clip.mp4"}]))
    (root / "summary.json").write_text(json.dumps({
        "metadata": {"duration_s": 188, "tracking_preview_stale": False, "export_stale": True},
        "quality": {"winner_visible": False, "required_review_count": 121}, "fighters": {},
    }))
    (root / ".render_cache").mkdir()
    (root / ".render_cache" / "private.txt").write_text("not bundled")
    (root / "previews").mkdir()
    (root / "previews" / "sprite.webp").write_bytes(b"fixture")
    (root / "preview_manifest.json").write_text(json.dumps({"sheets": ["previews/sprite.webp"], "frames": []}))
    return root


def test_demo_is_portable_read_only_and_keeps_quality_limits(tmp_path):
    source = source_run(tmp_path)
    before = (source / "summary.json").read_bytes()
    demo = prepare_demo(source, tmp_path / "demo")
    manifest = verify_demo(demo)
    assert {r["path"] for r in manifest["files"]} == {
        "workspace-preview.mp4", "events.json", "summary.json", "preview_manifest.json", "previews/sprite.webp"}
    summary = json.loads((demo / "summary.json").read_text())
    assert summary["quality"] == {"winner_visible": False, "required_review_count": 121}
    assert summary["metadata"]["export_stale"] is True
    assert summary["metadata"]["demo_start_ms"] == 49583
    assert summary["metadata"]["demo_selected_event_id"] == "event"
    assert "clip_path" not in json.loads((demo / "events.json").read_text())[0]
    assert (source / "summary.json").read_bytes() == before
    state = ui._load_existing_run(demo)
    assert state["duration_s"] == 188
    assert Path(state["workspace_video"]).name == "workspace-preview.mp4"
    assert not ui._available_export_files(state)[0]["visible"]
    with pytest.raises(Exception, match="демо-разбор"):
        ui._require_mutable_result(state)


def test_demo_staging_and_first_launch_need_no_model_or_cache(tmp_path, monkeypatch):
    demo = prepare_demo(source_run(tmp_path), tmp_path / "prepared")
    bundle = tmp_path / "bundle"
    record = stage_demo(demo, bundle)
    paths = desktop.DesktopPaths(bundle, tmp_path / "data", bundle / "fonts")
    monkeypatch.setattr(desktop.os, "environ", {})
    root = desktop.prepare_bundled_demo(paths, {"files": [record]})
    assert root == bundle / "demo"
    assert desktop.os.environ["BOXING_VISION_DEMO_RUN"] == str(root)
    assert not paths.data.exists()


@pytest.mark.parametrize("fault", ["stale", "outside-preview", "symlink", "bad-time", "existing-output"])
def test_invalid_demo_never_overwrites_source(tmp_path, fault):
    source = source_run(tmp_path)
    output = tmp_path / "demo"
    if fault == "stale":
        summary = json.loads((source / "summary.json").read_text())
        summary["metadata"]["tracking_preview_stale"] = True
        (source / "summary.json").write_text(json.dumps(summary))
    if fault == "outside-preview":
        (source / "preview_manifest.json").write_text(json.dumps({"sheets": ["../secret.webp"]}))
    if fault == "symlink":
        (source / "previews" / "sprite.webp").unlink()
        (source / "previews" / "sprite.webp").symlink_to(source / "annotated.mp4")
    if fault == "existing-output":
        output.mkdir()
        (output / "keep.txt").write_text("keep")
    with pytest.raises(desktop.DesktopSetupError):
        prepare_demo(source, output, start_ms=999999 if fault == "bad-time" else 49583)
    assert (source / "annotated.mp4").read_bytes() == b"stale must not ship"
    if fault == "existing-output":
        assert (output / "keep.txt").read_text() == "keep"


def test_tampered_demo_fails_integrity_check(tmp_path):
    demo = prepare_demo(source_run(tmp_path), tmp_path / "demo")
    (demo / "events.json").write_text("[]")
    with pytest.raises(desktop.DesktopSetupError, match="повреждён"):
        verify_demo(demo)
