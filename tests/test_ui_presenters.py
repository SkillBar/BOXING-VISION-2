from __future__ import annotations

import json
from pathlib import Path

import pytest

from boxing_vision.ui_presenters import (
    aggregate_fighter,
    build_presentation_payload,
    filter_events_for_scope,
    render_fighter_panel_mount,
    render_workspace_shell,
)


def _event(
    index: int,
    *,
    attacker: str = "fighter_a",
    target: str = "head",
    outcome: str = "likely_landed",
    review_status: str = "unreviewed",
    replay: bool = False,
) -> dict[str, object]:
    return {
        "event_id": f"evt_{index:05d}",
        "round": 1 if index < 10 else 2,
        "start_ms": index * 1000,
        "peak_ms": index * 1000 + 200,
        "end_ms": index * 1000 + 450,
        "attacker_id": attacker,
        "defender_id": "fighter_b" if attacker == "fighter_a" else "fighter_a",
        "hand": "left" if index % 2 else "right",
        "technique": "hook",
        "target": target,
        "outcome": outcome,
        "confidence": 0.86,
        "impact_proxy_0_100": 67,
        "is_replay": replay,
        "review_status": review_status,
    }


def test_scope_and_dealt_received_target_aggregations() -> None:
    events = [
        _event(0, target="head", outcome="likely_landed"),
        _event(1, target="body", outcome="blocked"),
        _event(2, attacker="fighter_b", target="body", outcome="likely_landed"),
        _event(3, target="unknown", outcome="unclear"),
        _event(4, target="head", outcome="likely_landed", replay=True),
        _event(5, target="body", outcome="missed", review_status="rejected"),
    ]

    dealt = aggregate_fighter(events, "fighter_a", mode="dealt")
    received = aggregate_fighter(events, "fighter_a", mode="received")
    to_time = aggregate_fighter(
        events,
        "fighter_a",
        mode="dealt",
        scope="to_time",
        current_time_ms=1500,
    )

    assert dealt["attempts"] == 3
    assert dealt["likely_landed"] == 1
    assert dealt["blocked"] == 1
    assert dealt["unclear"] == 1
    assert dealt["targets"]["head"] == {"landed": 1, "thrown": 1}
    assert dealt["targets"]["body"] == {"landed": 0, "thrown": 1}
    assert received["attempts"] == 1
    assert received["targets"]["body"] == {"landed": 1, "thrown": 1}
    assert to_time["attempts"] == 2


def test_round_scope_and_invalid_scope() -> None:
    events = [_event(1), _event(11)]
    assert [
        event["event_id"]
        for event in filter_events_for_scope(events, "round", round_number=2)
    ] == ["evt_00011"]
    with pytest.raises(ValueError):
        filter_events_for_scope(events, "tomorrow")  # type: ignore[arg-type]


@pytest.mark.parametrize("count", [0, 1, 500, 2000])
def test_payload_handles_timeline_cardinalities(count: int) -> None:
    payload = build_presentation_payload(
        (_event(index) for index in range(count)), {}, 120.0
    )
    assert len(payload["events"]) == count
    assert payload["duration_ms"] >= 120_000
    assert payload["metadata"]["round_length_s"] == 180
    assert payload["metadata"]["rest_length_s"] == 60
    assert payload["fighters"]["fighter_a"]["corner"] == "red"
    assert payload["fighters"]["fighter_b"]["corner"] == "blue"
    assert payload["assets"]["body_map_base"].endswith(
        "/body-map-v3-base.png"
    )
    assert payload["assets"]["body_map_head_mask"].endswith("/body-map-v3-head.png")
    assert payload["assets"]["body_map_body_mask"].endswith("/body-map-v3-body.png")


def test_optional_profile_fields_and_nested_portrait_are_safe(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    portrait = profile_dir / "fighter_a.webp"
    portrait.write_bytes(b"RIFFfake")
    summary = {
        "fighters": {
            "fighter_a": {
                "name": "Анна Иванова",
                "record": "14–2",
                "portrait_filename": "profiles/fighter_a.webp",
            },
            "fighter_b": {"name": "Боксёр B", "portrait_filename": "../secret.webp"},
        }
    }
    payload = build_presentation_payload([], summary, run_dir=tmp_path)

    assert payload["fighters"]["fighter_a"]["record"] == "14–2"
    assert payload["fighters"]["fighter_a"]["portrait_url"].endswith(
        "/profiles/fighter_a.webp"
    )
    assert "portrait_url" not in payload["fighters"]["fighter_b"]


def test_preview_manifest_resolves_sheet_urls_and_geometry(tmp_path: Path) -> None:
    preview_dir = tmp_path / "previews"
    preview_dir.mkdir()
    (preview_dir / "sprite-001.webp").write_bytes(b"RIFFfake")
    manifest = {
        "version": 1,
        "interval_ms": 2000,
        "tile_width": 160,
        "tile_height": 90,
        "sheet_width": 1622,
        "sheet_height": 922,
        "sheets": ["previews/sprite-001.webp"],
        "frames": [{"time_ms": 0, "sheet": 0, "x": 2, "y": 2}],
    }
    manifest_path = tmp_path / "preview_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    payload = build_presentation_payload(
        [],
        {"metadata": {"preview_manifest": "preview_manifest.json"}},
        run_dir=tmp_path,
    )

    resolved = payload["preview_manifest"]
    assert resolved["tile_width"] == 160
    assert resolved["frames"][0] == {"time_ms": 0, "sheet": 0, "x": 2, "y": 2}
    assert resolved["sheets"][0].endswith("/previews/sprite-001.webp")


def test_embedded_payload_escapes_untrusted_names_and_script_text() -> None:
    malicious = "</div><script>alert('x')</script>"
    payload = build_presentation_payload(
        [_event(1)],
        {"fighters": {"fighter_a": {"name": malicious}}},
        3.0,
    )
    html = render_workspace_shell(payload)

    assert malicious not in html
    assert "&lt;/div&gt;&lt;script&gt;" in html
    assert "data-bv-payload hidden" in html
    assert render_fighter_panel_mount("fighter_a").startswith(
        '<aside class="bv-fighter-panel'
    )


def test_workspace_shell_uses_russian_boxing_terms_and_theater_control() -> None:
    html = render_workspace_shell(build_presentation_payload([_event(1)], {}, 3.0))

    assert "На весь экран" in html
    assert "data-bv-theater" in html
    assert "Голова + корпус" in html
    assert "Попадание" in html
    assert "НЕОФИЦИАЛЬНАЯ AI-ОЦЕНКА" not in html
    assert "Экспериментальная аналитика" not in html
    assert "Другие фильтры" in html
    assert 'data-bv-workspace-version="2"' in html
    assert 'data-bv-a11y-listbox role="listbox"' in html
    assert 'aria-owns="bv-event-listbox"' in html
    assert "Предыдущий удар" in html
    assert "Следующий удар" in html
    assert "По ширине · до момента" in html


def test_payload_hides_winner_when_identity_quality_gate_fails() -> None:
    payload = build_presentation_payload(
        [_event(1)],
        {
            "winner": {
                "fighter_id": "fighter_a",
                "name": "Боксёр A",
                "confidence": 0.9,
            },
            "quality": {"winner_visible": False},
        },
        3.0,
    )

    assert payload["result"]["winner"] == {}


def test_payload_keeps_optional_canonical_target_point_fields() -> None:
    event = _event(1)
    event.update(
        {
            "target_point_norm": {"x": 0.52, "y": 0.31},
            "target_point_confidence": 0.91,
            "target_uncertainty_radius": 0.07,
            "target_point_space": "defender_front_canonical_v1",
            "proposal_confidence": 0.82,
            "classification_confidence": 0.78,
            "outcome_confidence": 0.73,
            "model_version": "pose-tcn-demo-1",
        }
    )

    payload = build_presentation_payload([event], {}, 3.0)
    compact = payload["events"][0]

    assert compact["target_point_norm"] == {"x": 0.52, "y": 0.31}
    assert compact["target_point_confidence"] == 0.91
    assert compact["target_uncertainty_radius"] == 0.07
    assert compact["target_point_space"] == "defender_front_canonical_v1"
    assert compact["proposal_confidence"] == 0.82
    assert compact["classification_confidence"] == 0.78
    assert compact["outcome_confidence"] == 0.73
    assert compact["model_version"] == "pose-tcn-demo-1"


def test_payload_accepts_legacy_target_point_array() -> None:
    event = _event(1)
    event["target_point_norm"] = [0.33, 0.44]

    compact = build_presentation_payload([event], {}, 3.0)["events"][0]

    assert compact["target_point_norm"] == {"x": 0.33, "y": 0.44}


def test_payload_does_not_invent_missing_impact_and_normalizes_legacy_accuracy() -> None:
    event = _event(1)
    event.pop("impact_proxy_0_100")
    summary = {
        "fighters": {
            "fighter_a": {
                "attempts": 10,
                "likely_landed": 5,
                "accuracy_pct": 50,
            }
        }
    }

    payload = build_presentation_payload([event], summary, 3.0)

    assert payload["events"][0]["impact_proxy_0_100"] is None
    assert payload["fighters"]["fighter_a"]["summary_totals"]["accuracy"] == 0.5
