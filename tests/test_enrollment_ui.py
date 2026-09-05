from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import numpy as np
import pytest

from boxing_vision import ui


def enrollment_state(source: Path) -> dict:
    return {
        "source": str(source.resolve()), "start_s": 10.0, "active_index": 0,
        "ring_points": [(0.02, 0.02), (0.98, 0.02), (0.98, 0.98), (0.02, 0.98)],
        "confirmed": False,
        "views": [
            {"image": np.zeros((100, 200, 3), np.uint8), "time_s": t,
             "boxes": [[10, 5, 80, 95], [110, 5, 180, 95], [85, 10, 105, 80]],
             "selection": {"fighter_a": 0, "fighter_b": 1}, "quality": .9}
            for t in (10.0, 14.0, 18.0)
        ],
    }


def test_auto_proposals_are_not_confirmation(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "fight.mp4"
    monkeypatch.setattr(ui, "validate_video", lambda _: SimpleNamespace(duration_s=30))
    monkeypatch.setattr(ui, "_inspect_video", lambda _: "metadata")
    observed = []

    def sample(_, time):
        observed.append(time)
        result = deepcopy(enrollment_state(source)["views"][0])
        result["time_s"] = time
        return result

    monkeypatch.setattr(ui, "_enrollment_sample", sample)
    _, image, state, _, _, _ = ui._prepare_enrollment(str(source), 10, lambda *args, **kwargs: None)
    assert len(observed) == 9
    assert len(state["views"]) == 3
    assert not state["confirmed"]
    assert "enrollment_samples" not in state
    assert image is not None
    with pytest.raises(ValueError, match="Подтвердить"):
        ui._enrollment_config_fields(state, str(source), 10)


def test_confirmed_three_views_keep_actual_normalized_detector_boxes(tmp_path: Path) -> None:
    source = tmp_path / "fight.mp4"
    _, state, _ = ui._confirm_enrollment_ui(enrollment_state(source))
    fields = ui._enrollment_config_fields(state, str(source), 10)
    assert fields["enrollment_confirmed"] is True
    assert fields["enrollment_frames"] == (0.0, 4.0, 8.0)
    assert fields["enrollment_samples"][0]["fighter_a"] == [0.05, 0.05, 0.4, 0.95]
    assert fields["enrollment_samples"][0]["fighter_b"] == [0.55, 0.05, 0.9, 0.95]
    assert len(fields["ring_rois"]) == 4


def test_click_selects_real_detection_and_invalidates_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "fight.mp4"
    original = enrollment_state(source)
    _, confirmed, _ = ui._confirm_enrollment_ui(original)
    _, state, _, next_role = ui._apply_enrollment_selection(confirmed, "fighter_a", (90, 50))
    assert state["views"][0]["selection"]["fighter_a"] == 2
    assert not state["confirmed"]
    assert "enrollment_samples" not in state
    assert confirmed["views"][0]["selection"]["fighter_a"] == 0
    assert next_role["value"] == "fighter_b"


def test_one_detection_cannot_be_both_fighters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Один человек"):
        ui._apply_enrollment_selection(enrollment_state(tmp_path / "f.mp4"), "fighter_a", (120, 50))


def test_a_swapped_pair_can_be_corrected_without_selecting_a_third_person(tmp_path: Path) -> None:
    initial = enrollment_state(tmp_path / "f.mp4")
    _, state, _, _ = ui._clear_enrollment_selection(initial, "fighter_a")
    _, state, _, _ = ui._apply_enrollment_selection(state, "fighter_a", (120, 50))
    _, state, _, _ = ui._apply_enrollment_selection(state, "fighter_b", (40, 50))
    assert state["views"][0]["selection"] == {"fighter_a": 1, "fighter_b": 0}
    assert not state["confirmed"]


def test_click_outside_detections_does_not_create_an_anchor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="не обнаружен"):
        ui._apply_enrollment_selection(enrollment_state(tmp_path / "f.mp4"), "fighter_a", (199, 99))


@pytest.mark.parametrize("points", [[], [(0, 0)] * 4, [(0, 0), (1, 1), (1, 0), (0, 1)]])
def test_ring_requires_a_non_intersecting_quadrilateral(tmp_path: Path, points) -> None:
    state = enrollment_state(tmp_path / "f.mp4")
    state["ring_points"] = points
    with pytest.raises(gr.Error):
        ui._confirm_enrollment_ui(state)


def test_frame_navigation_preserves_confirmation_and_ring(tmp_path: Path) -> None:
    _, confirmed, _ = ui._confirm_enrollment_ui(enrollment_state(tmp_path / "f.mp4"))
    _, state, _, time = ui._show_enrollment_frame(confirmed, 2)
    assert state["confirmed"]
    assert time == 18
    assert state["ring_points"] == confirmed["ring_points"]
    with pytest.raises(ValueError, match="первом кадре"):
        ui._apply_enrollment_selection(state, "ring", (10, 10))


def test_source_or_fight_start_change_requires_new_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "fight.mp4"
    _, state, _ = ui._confirm_enrollment_ui(enrollment_state(source))
    with pytest.raises(ValueError, match="изменилось"):
        ui._enrollment_config_fields(state, str(tmp_path / "other.mp4"), 10)
    with pytest.raises(ValueError, match="изменилось"):
        ui._enrollment_config_fields(state, str(source), 11)


def test_duplicate_calibration_times_are_rejected(tmp_path: Path) -> None:
    state = enrollment_state(tmp_path / "f.mp4")
    state["views"][1]["time_s"] = state["views"][0]["time_s"]
    with pytest.raises(gr.Error, match="разных кадра"):
        ui._confirm_enrollment_ui(state)


def test_confirmation_explains_floor_vs_rope_polygon_when_both_supports_are_visible(tmp_path: Path) -> None:
    state = enrollment_state(tmp_path / "f.mp4")
    state["ring_points"] = [(0, .3), (1, .3), (1, .45), (0, .45)]
    state["views"][0]["support_keypoints"] = [
        {"left_ankle": {"x": .2, "y": .9, "score": .95}},
        {"left_ankle": {"x": .7, "y": .9, "score": .95}},
        {},
    ]
    with pytest.raises(gr.Error, match="полоса канатов"):
        ui._confirm_enrollment_ui(state)
    assert not state["confirmed"]


def test_enrollment_sample_retains_existing_ankles_without_another_inference(monkeypatch) -> None:
    from boxing_vision.contracts import BBox, Keypoint
    from boxing_vision.pose import RawPose

    calls = []
    pose = RawPose(BBox(10, 5, 80, 95), {"left_ankle": Keypoint(40, 90, .95)})
    monkeypatch.setattr(ui, "_prepare_confirmation", lambda *args: ("", np.zeros((100, 200, 3), np.uint8), {"frame_time_s": 2}, ""))
    monkeypatch.setattr(ui, "calibration_backend", lambda: SimpleNamespace(infer=lambda frame: calls.append(frame) or [pose]))
    sample = ui._enrollment_sample("unused.mp4", 2)
    assert len(calls) == 1
    assert sample["support_keypoints"] == [{"left_ankle": {"x": .2, "y": .9, "score": .95}}]


def write_review(tmp_path: Path) -> dict:
    items = [
        {"kind": "tracklet", "tracklet_id": "shot-2-track-9", "shot_id": 2, "source_track_id": 9,
         "start_ms": 3000, "end_ms": 6000, "identity_state": "UNKNOWN"},
        {"kind": "scene", "review_id": "scene-2", "shot_id": 2, "start_ms": 3000,
         "end_ms": 8000, "scene_state": "UNCERTAIN"},
        {"kind": "scene", "shot_id": 4},
        {"kind": "enrollment", "review_id": "identity-enrollment"},
    ]
    (tmp_path / "review.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    return {"run_dir": str(tmp_path), "events": [], "summary": {}, "duration_s": 10}


def test_review_choices_use_only_real_supported_ids(tmp_path: Path) -> None:
    payload = write_review(tmp_path)
    choices, _, _, _ = ui._refresh_identity_review(payload)
    assert [value for _, value in choices["choices"]] == ["shot-2-track-9", "scene-2"]


@pytest.mark.parametrize("item_id,action", [
    ("invented-id", "FIGHTER_A"), ("scene-2", "FIGHTER_A"),
    ("shot-2-track-9", "REPLAY"), ("scene-2", None),
])
def test_review_rejects_unknown_ids_and_actions_before_any_decode(tmp_path: Path, item_id, action) -> None:
    with pytest.raises(gr.Error):
        ui._apply_identity_review(item_id, action, write_review(tmp_path), lambda *args, **kwargs: None)


@pytest.mark.parametrize("item_id,action,expected_correction,expected_scenes", [
    ("shot-2-track-9", "OTHER", {"timestamp_ms": 4500, "source_track_id": 9,
                                   "identity_state": "OTHER", "segment_id": None}, {}),
    ("scene-2", "BREAK", None, {"2": "BREAK"}),
])
def test_review_only_redecodes_cache_and_invalidates_export(
    monkeypatch, tmp_path: Path, item_id, action, expected_correction, expected_scenes,
) -> None:
    from boxing_vision import cache_review, pipeline

    payload = write_review(tmp_path)
    decoded, corrected = [], []
    monkeypatch.setattr(pipeline, "redecode_from_cache", lambda run, **kwargs: decoded.append((run, kwargs)))
    monkeypatch.setattr(cache_review, "correct_identity_at", lambda run, **kwargs: corrected.append((run, kwargs)), raising=False)
    refreshed = dict(payload, events_path=str(tmp_path / "events.json"), summary_path=str(tmp_path / "summary.json"))
    monkeypatch.setattr(ui, "_load_existing_run", lambda _: refreshed)
    monkeypatch.setattr(ui, "_workspace", lambda *args: "workspace")
    monkeypatch.setattr(ui, "_overview", lambda _: "overview")
    monkeypatch.setattr(ui, "_event_rows", lambda *args: [])
    outputs = ui._apply_identity_review(item_id, action, payload, lambda *args, **kwargs: None)
    assert corrected == ([(tmp_path, expected_correction)] if expected_correction else [])
    assert decoded == ([] if expected_correction else [(tmp_path, {"identity_overrides": {}, "scene_overrides": expected_scenes})])
    assert outputs[4]["render_stale"] is True
    assert outputs[4]["identity_overrides"] == {}
    assert outputs[4]["scene_overrides"] == expected_scenes
    assert outputs[5] is None


def test_demo_portraits_are_presentational_and_only_for_explicit_demo(monkeypatch, tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    upload = tmp_path / "upload"
    summary = {"metadata": {"duration_s": 10}}
    captured = []
    monkeypatch.setenv("BOXING_VISION_DEMO_RUN", str(demo))
    monkeypatch.setattr(ui, "build_presentation_payload", lambda events, data, *args, **kwargs: captured.append(data) or {})
    monkeypatch.setattr(ui, "render_workspace_shell", lambda _: "workspace")
    ui._workspace([], summary, 10, demo)
    ui._workspace([], summary, 10, upload)
    ui._workspace([], summary, 10)
    assert [data["metadata"]["demo_portraits"] for data in captured] == [True, False, False]
    assert "demo_portraits" not in summary["metadata"]


@pytest.mark.parametrize("pending,stale,expected_visible", [(0, False, True), (1, False, False), (0, True, False)])
def test_final_exports_are_hidden_until_review_and_render_are_current(tmp_path: Path, pending, stale, expected_visible) -> None:
    video = tmp_path / "annotated.mp4"
    video.touch()
    (tmp_path / "boxing-vision-result.zip").touch()
    state = {"run_dir": str(tmp_path), "annotated_video": str(video),
             "summary": {"quality": {"required_review_count": pending}, "metadata": {"export_stale": stale}}}
    mp4, archive = ui._available_export_files(state)
    assert mp4["visible"] == archive["visible"] == expected_visible
    if not expected_visible:
        assert mp4["value"] is None and archive["value"] is None


def test_required_review_blocks_render_before_it_starts(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps({"quality": {"required_review_count": 1}}))
    renders = []
    monkeypatch.setattr(ui, "rebuild_from_cache", lambda *args, **kwargs: renders.append(args))
    with pytest.raises(gr.Error, match="обязательные эпизоды"):
        ui._rebuild_reviewed_video({"run_dir": str(tmp_path)}, lambda *args, **kwargs: None)
    assert not renders
