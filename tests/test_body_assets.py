from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from boxing_vision.ui_presenters import STATIC_DIR, build_presentation_payload

ASSETS = STATIC_DIR / "assets"


def test_local_typography_and_portrait_caption_contract() -> None:
    css = (STATIC_DIR / "boxing_vision.css").read_text()
    js = (STATIC_DIR / "boxing_vision.js").read_text()
    for face in ("SFProText-Regular", "SFProText-Medium", "SFProText-Semibold", "SFProText-Bold", "DrukCyr-Medium"):
        assert f'local("{face}")' in css
    assert '"bv-fighter-record", "Демопортрет"' not in js
    assert "avatar.title" in js  # Source attribution remains accessible.
    assert "font-kerning: normal" in css
    assert "font-variant-numeric: tabular-nums" in css


@pytest.fixture(scope="module")
def atlas() -> dict[str, np.ndarray]:
    layers = {}
    for name in ("base", "head", "body"):
        with Image.open(ASSETS / f"body-map-v3-{name}.png") as image:
            assert image.mode == "RGBA"
            assert image.size == (1024, 1536)
            layers[name] = np.array(image)
    return layers


@pytest.mark.parametrize("layer", ["base", "head", "body"])
def test_v3_atlas_layers_have_actual_transparency_and_opaque_content(atlas, layer) -> None:
    alpha = atlas[layer][:, :, 3]
    assert alpha.min() == 0
    assert alpha.max() == 255
    assert np.count_nonzero(alpha) > 1000


def test_zone_masks_remain_inside_base_and_do_not_overlap(atlas) -> None:
    base, head, body = (atlas[name][:, :, 3] for name in ("base", "head", "body"))
    assert np.all(head <= base)
    assert np.all(body <= base)
    assert not np.any((head > 0) & (body > 0))


def test_manifest_and_payload_use_one_registered_v3_atlas() -> None:
    manifest = json.loads((ASSETS / "body-map-v3-manifest.json").read_text())
    payload = build_presentation_payload([], {}, 30)["assets"]
    assert manifest["canvas_size"] == [1024, 1536]
    assert manifest["registration_error_px"] <= 1
    for field, manifest_key in (("body_map_base", "base"), ("body_map_head_mask", "head_mask"), ("body_map_body_mask", "body_mask")):
        assert payload[field].endswith("/" + manifest[manifest_key])
        assert (ASSETS / manifest[manifest_key]).is_file()
    assert payload["body_map"] == payload["body_map_base"]
    assert payload["body_map_atlas"]["canvas_size"] == manifest["canvas_size"]
    assert payload["body_map_atlas"]["viewport_crop"] == manifest["viewport_crop"]


def contact_event(**fields) -> dict:
    return {"event_id": "contact-1", "start_ms": 900, "peak_ms": 1000, "end_ms": 1200,
            "attacker_id": "fighter_a", "defender_id": "fighter_b", "target": "body",
            "outcome": "likely_landed", "confidence": .9, **fields}


def compact_contact(**fields) -> dict:
    return build_presentation_payload([contact_event(**fields)], {}, 30)["events"][0]


@pytest.mark.parametrize("legacy", [
    {"target_point_norm": {"x": .5, "y": .3}},
    {"target_point_x_norm": .5, "target_point_y_norm": .3},
    {"target_point_norm": {"x": .5, "y": .3}, "target_point_space": "defender_front_canonical_v1"},
    {"target_point_norm": {"x": .5, "y": .3}, "target_point_source": "bbox_projection",
     "target_point_space": "defender_front_canonical_v1"},
])
def test_legacy_coordinates_are_never_upgraded_to_proven_canonical_contacts(legacy) -> None:
    event = compact_contact(**legacy)
    assert event.get("target_point_source") != "canonical_contact_v1"


def test_verified_canonical_contact_keeps_its_original_geometry() -> None:
    event = compact_contact(target_point_norm={"x": .43, "y": .315},
        target_point_source="canonical_contact_v1", target_point_space="defender_front_canonical_v1",
        target_point_confidence=.94, target_uncertainty_radius=.06)
    assert event["target_point_norm"] == {"x": .43, "y": .315}
    assert event["target_point_confidence"] == .94
    assert event["target_uncertainty_radius"] == .06


@pytest.mark.parametrize("point", [
    {"x": float("nan"), "y": .3}, {"x": float("inf"), "y": .3},
    {"x": None, "y": .3}, {"x": "bad", "y": .3}, {"x": True, "y": .3},
    {"x": -.01, "y": .3}, {"x": .5, "y": 1.01}, {},
])
def test_invalid_canonical_coordinates_are_not_repaired_into_plausible_hits(point) -> None:
    event = compact_contact(target_point_norm=point, target_point_source="canonical_contact_v1",
        target_point_space="defender_front_canonical_v1", target_point_confidence=.94)
    assert "target_point_norm" not in event


@pytest.mark.parametrize("flag", [False, None, "true", "false", 1, 0])
def test_real_boxer_demo_portraits_require_explicit_boolean_opt_in(flag) -> None:
    payload = build_presentation_payload([], {"metadata": {"demo_portraits": flag}})
    assert all("portrait_url" not in fighter for fighter in payload["fighters"].values())


def test_demo_portraits_include_labels_and_attribution() -> None:
    payload = build_presentation_payload([], {"metadata": {"demo_portraits": True}})
    for role, filename in (("fighter_a", "demo-bivol.jpg"), ("fighter_b", "demo-usyk.jpg")):
        fighter = payload["fighters"][role]
        assert fighter["portrait_url"].endswith("/" + filename)
        assert (ASSETS / filename).is_file()
        assert fighter["demo_portrait"] is True
        assert "не идентификация" in fighter["portrait_label"]
        assert "CC BY-SA" in fighter["portrait_attribution"]


def test_custom_run_portrait_wins_over_demo_fallback(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    Image.new("RGB", (2, 2)).save(profile_dir / "a.png")
    summary = {"metadata": {"demo_portraits": True},
               "fighters": {"fighter_a": {"portrait_filename": "profiles/a.png"}}}
    payload = build_presentation_payload([], summary, run_dir=tmp_path)
    assert payload["fighters"]["fighter_a"]["portrait_url"].endswith("/profiles/a.png")
    assert "demo_portrait" not in payload["fighters"]["fighter_a"]
    assert payload["fighters"]["fighter_b"]["demo_portrait"] is True


def test_explicit_portrait_url_wins_without_changing_other_fighter() -> None:
    payload = build_presentation_payload([], {"metadata": {"demo_portraits": True}},
        portrait_urls={"fighter_b": "/profiles/approved-b.png"})
    assert payload["fighters"]["fighter_b"]["portrait_url"] == "/profiles/approved-b.png"
    assert "demo_portrait" not in payload["fighters"]["fighter_b"]
    assert payload["fighters"]["fighter_a"]["demo_portrait"] is True
