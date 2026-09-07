"""Presentation-only helpers for the interactive Boxing Vision workspace.

The ML pipeline deliberately emits a stable, verbose JSON contract.  The web
workspace needs a smaller document that is safe to embed in ``gr.HTML`` and can
be recalculated in the browser while the viewer scrubs the video.  This module
is the boundary between those two concerns: it never mutates analysis results
and it has no dependency on Gradio.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from html import escape
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from .contracts import PunchEvent

STATIC_DIR = Path(__file__).with_name("static")
BODY_MAP_ASSET = STATIC_DIR / "assets" / "body-map-v3-base.png"
BODY_MAP_HEAD_MASK = STATIC_DIR / "assets" / "body-map-v3-head.png"
BODY_MAP_BODY_MASK = STATIC_DIR / "assets" / "body-map-v3-body.png"

FIGHTER_IDS = ("fighter_a", "fighter_b")
OUTCOMES = ("likely_landed", "blocked", "missed", "unclear")
TARGETS = ("head", "body", "unknown")
REJECTED_REVIEW_STATES = {"rejected", "deleted"}


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _integer(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _canonical_outcome(value: object) -> str:
    normalized = (
        str(value or "unclear").strip().lower().replace("-", "_").replace(" ", "_")
    )
    if normalized in {"landed", "likely_landed"}:
        return "likely_landed"
    if normalized in {"blocked", "block"}:
        return "blocked"
    if normalized in {"miss", "missed"}:
        return "missed"
    return "unclear"


def _canonical_target(value: object) -> str:
    normalized = str(value or "unknown").strip().lower().replace("-", "_")
    return normalized if normalized in TARGETS else "unknown"


def _event_mapping(event: Mapping[str, Any] | PunchEvent) -> dict[str, Any]:
    if isinstance(event, PunchEvent):
        return event.to_dict()
    return dict(event)


def _compact_event(event: Mapping[str, Any] | PunchEvent) -> dict[str, Any]:
    raw = _event_mapping(event)
    start_ms = max(0, _integer(raw.get("start_ms")))
    peak_ms = max(start_ms, _integer(raw.get("peak_ms"), start_ms))
    end_ms = max(peak_ms, _integer(raw.get("end_ms"), peak_ms))
    confidence = min(1.0, max(0.0, _number(raw.get("confidence"))))
    raw_impact = raw.get("impact_proxy_0_100")
    impact = (
        None if raw_impact is None else min(100, max(0, _integer(raw_impact)))
    )
    review_status = str(raw.get("review_status") or "unreviewed").strip().lower()
    evidence = _mapping(raw.get("evidence"))
    compact: dict[str, Any] = {
        "event_id": str(raw.get("event_id") or f"event-{peak_ms}"),
        "round": max(1, _integer(raw.get("round"), 1)),
        "start_ms": start_ms,
        "peak_ms": peak_ms,
        "end_ms": end_ms,
        "attacker_id": str(raw.get("attacker_id") or "unknown"),
        "defender_id": str(raw.get("defender_id") or "unknown"),
        "hand": "left"
        if str(raw.get("hand")).lower() == "left"
        else ("right" if str(raw.get("hand")).lower() == "right" else "unknown"),
        "technique": str(raw.get("technique") or "unknown").lower(),
        "target": _canonical_target(raw.get("target")),
        "outcome": _canonical_outcome(raw.get("outcome")),
        "confidence": round(confidence, 4),
        "impact_proxy_0_100": impact,
        "is_replay": bool(raw.get("is_replay")),
        "review_status": review_status,
    }
    if raw.get("clip_path"):
        compact["clip_path"] = str(raw["clip_path"])
    if raw.get("combo_id"):
        compact["combo_id"] = str(raw["combo_id"])
    if raw.get("is_counter"):
        compact["is_counter"] = True
    if evidence.get("possible_knockdown"):
        compact["possible_knockdown"] = True
    if raw.get("preview_url"):
        compact["preview_url"] = str(raw["preview_url"])
    # Newer analysis bundles may provide a contact point in the canonical
    # front-facing fighter atlas.  Keep these fields optional so cached runs
    # remain valid and the browser can fall back to a broad HEAD/BODY pulse.
    raw_point = raw.get("target_point_norm")
    point: dict[str, float] | None = None
    if isinstance(raw_point, Mapping):
        point = {
            "x": round(min(1.0, max(0.0, _number(raw_point.get("x"), 0.5))), 4),
            "y": round(min(1.0, max(0.0, _number(raw_point.get("y"), 0.5))), 4),
        }
    elif isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
        point = {
            "x": round(min(1.0, max(0.0, _number(raw_point[0], 0.5))), 4),
            "y": round(min(1.0, max(0.0, _number(raw_point[1], 0.5))), 4),
        }
    elif raw.get("target_point_x_norm") is not None and raw.get(
        "target_point_y_norm"
    ) is not None:
        point = {
            "x": round(
                min(1.0, max(0.0, _number(raw.get("target_point_x_norm"), 0.5))),
                4,
            ),
            "y": round(
                min(1.0, max(0.0, _number(raw.get("target_point_y_norm"), 0.5))),
                4,
            ),
        }
    if point is not None:
        # Never turn invalid or missing geometry into a credible centre point.
        if isinstance(raw_point, Mapping):
            original = [raw_point.get("x"), raw_point.get("y")]
        elif isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
            original = list(raw_point[:2])
        else:
            original = [raw.get("target_point_x_norm"), raw.get("target_point_y_norm")]
        try:
            import math
            valid = all(not isinstance(value, bool) and value is not None and math.isfinite(float(value))
                        and 0 <= float(value) <= 1 for value in original)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            point = None
    if point is not None:
        compact["target_point_norm"] = point
        compact["target_point_confidence"] = round(
            min(
                1.0,
                max(
                    0.0,
                    _number(
                        raw.get("target_point_confidence"),
                        0.0,
                    ),
                ),
            ),
            4,
        )
        compact["target_uncertainty_radius"] = round(
            min(1.0, max(0.0, _number(raw.get("target_uncertainty_radius"), 0.08))),
            4,
        )
        compact["target_point_space"] = str(
            raw.get("target_point_space") or "unspecified"
        )
        compact["target_point_source"] = raw.get("target_point_source")
    for field in (
        "proposal_confidence",
        "classification_confidence",
        "outcome_confidence",
    ):
        if raw.get(field) is not None:
            compact[field] = round(min(1.0, max(0.0, _number(raw.get(field)))), 4)
    if raw.get("model_version"):
        compact["model_version"] = str(raw["model_version"])
    return compact


def _is_countable(event: Mapping[str, Any]) -> bool:
    return (
        not bool(event.get("is_replay"))
        and str(event.get("review_status", "")).lower() not in REJECTED_REVIEW_STATES
    )


def filter_events_for_scope(
    events: Iterable[Mapping[str, Any] | PunchEvent],
    scope: Literal["to_time", "round", "fight"] = "fight",
    *,
    current_time_ms: int | None = None,
    round_number: int | None = None,
) -> list[dict[str, Any]]:
    """Return canonical, countable events for one UI metric scope.

    ``to_time`` is intentionally inclusive at ``peak_ms``.  This keeps the
    counters and the event feed in sync with a frame on the exact event peak.
    """

    canonical = [_compact_event(event) for event in events]
    selected = [event for event in canonical if _is_countable(event)]
    if scope == "to_time":
        limit = max(0, _integer(current_time_ms))
        selected = [
            event for event in selected if _integer(event.get("peak_ms")) <= limit
        ]
    elif scope == "round":
        selected_round = max(1, _integer(round_number, 1))
        selected = [
            event
            for event in selected
            if _integer(event.get("round"), 1) == selected_round
        ]
    elif scope != "fight":
        raise ValueError(f"Unsupported metric scope: {scope}")
    return sorted(
        selected,
        key=lambda event: (_integer(event.get("peak_ms")), str(event.get("event_id"))),
    )


def aggregate_fighter(
    events: Iterable[Mapping[str, Any] | PunchEvent],
    fighter_id: str,
    *,
    mode: Literal["dealt", "received"] = "dealt",
    scope: Literal["to_time", "round", "fight"] = "fight",
    current_time_ms: int | None = None,
    round_number: int | None = None,
) -> dict[str, Any]:
    """Aggregate headline and HEAD/BODY metrics for a fighter.

    The numerator of accuracy is likely-landed and the denominator is the set
    of classifiable attempts (landed + blocked + missed). ``unclear`` remains a
    visible counter but cannot silently depress the model's accuracy estimate.
    """

    if mode not in {"dealt", "received"}:
        raise ValueError(f"Unsupported body-map mode: {mode}")
    scoped = filter_events_for_scope(
        events,
        scope,
        current_time_ms=current_time_ms,
        round_number=round_number,
    )
    identity_field = "attacker_id" if mode == "dealt" else "defender_id"
    selected = [
        event for event in scoped if str(event.get(identity_field)) == fighter_id
    ]
    outcome_counts = {outcome: 0 for outcome in OUTCOMES}
    targets = {target: {"landed": 0, "thrown": 0} for target in TARGETS}
    impact_values: list[int] = []
    for event in selected:
        outcome = _canonical_outcome(event.get("outcome"))
        target = _canonical_target(event.get("target"))
        outcome_counts[outcome] += 1
        targets[target]["thrown"] += 1
        if outcome == "likely_landed":
            targets[target]["landed"] += 1
        if event.get("impact_proxy_0_100") is not None:
            impact_values.append(_integer(event.get("impact_proxy_0_100")))
    classified = sum(
        outcome_counts[key] for key in ("likely_landed", "blocked", "missed")
    )
    accuracy = outcome_counts["likely_landed"] / classified if classified else 0.0
    average_impact = sum(impact_values) / len(impact_values) if impact_values else 0.0
    return {
        "attempts": len(selected),
        "likely_landed": outcome_counts["likely_landed"],
        "blocked": outcome_counts["blocked"],
        "missed": outcome_counts["missed"],
        "unclear": outcome_counts["unclear"],
        "accuracy": round(accuracy, 4),
        "average_impact_proxy": round(average_impact, 1),
        "targets": targets,
    }


def gradio_file_url(path: str | Path) -> str:
    """Return the URL shape used by Gradio 5's local-file endpoint."""

    resolved = Path(path).expanduser().resolve()
    # Gradio URLs use forward slashes even when the filesystem is Windows.
    return "/gradio_api/file=" + quote(resolved.as_posix(), safe="/")


def _safe_run_asset(run_dir: Path | None, filename: object) -> str | None:
    if run_dir is None or not filename:
        return None
    # Presentation metadata stores run-relative filenames, not original user
    # paths.  Nested folders (for example ``profiles/fighter_a.webp``) are
    # supported, while absolute paths and traversal are rejected.
    relative = Path(str(filename))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = (run_dir / relative).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError:
        return None
    return gradio_file_url(candidate) if candidate.is_file() else None


def _preview_manifest_payload(
    value: Mapping[str, Any] | str | Path | None,
    *,
    run_dir: Path | None,
) -> dict[str, Any] | None:
    """Resolve a generated preview manifest to browser-loadable sheet URLs."""

    if value is None:
        return None
    manifest_root = run_dir
    if isinstance(value, (str, Path)):
        manifest_path = Path(value)
        if not manifest_path.is_absolute() and run_dir is not None:
            manifest_path = run_dir / manifest_path
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_file():
            return None
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(loaded, Mapping):
            return None
        raw = dict(loaded)
        manifest_root = manifest_path.parent
    else:
        raw = dict(value)
    if manifest_root is None:
        return None

    root = manifest_root.resolve()
    sheet_urls: list[str] = []
    sheet_index_map: dict[int, int] = {}
    for original_index, sheet in enumerate(raw.get("sheets", [])):
        if not isinstance(sheet, str) or not sheet:
            continue
        if sheet.startswith(("/gradio_api/file=", "http://", "https://")):
            sheet_index_map[original_index] = len(sheet_urls)
            sheet_urls.append(sheet)
            continue
        candidate = (root / sheet).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            sheet_index_map[original_index] = len(sheet_urls)
            sheet_urls.append(gradio_file_url(candidate))

    frames: list[dict[str, int]] = []
    for frame in raw.get("frames", []):
        if not isinstance(frame, Mapping):
            continue
        original_sheet_index = _integer(frame.get("sheet"), -1)
        sheet_index = sheet_index_map.get(original_sheet_index)
        if sheet_index is None:
            continue
        frames.append(
            {
                "time_ms": max(0, _integer(frame.get("time_ms"))),
                "sheet": sheet_index,
                "x": max(0, _integer(frame.get("x"))),
                "y": max(0, _integer(frame.get("y"))),
            }
        )
    if not sheet_urls or not frames:
        return None
    return {
        "version": max(1, _integer(raw.get("version"), 1)),
        "interval_ms": max(1, _integer(raw.get("interval_ms"), 2000)),
        "tile_width": max(1, _integer(raw.get("tile_width"), 160)),
        "tile_height": max(1, _integer(raw.get("tile_height"), 90)),
        "sheet_width": max(1, _integer(raw.get("sheet_width"), 160)),
        "sheet_height": max(1, _integer(raw.get("sheet_height"), 90)),
        "sheets": sheet_urls,
        "frames": frames,
    }


def _fighter_profile(
    summary: Mapping[str, Any],
    fighter_id: str,
    *,
    run_dir: Path | None,
    portrait_urls: Mapping[str, str] | None,
) -> dict[str, Any]:
    fighters = _mapping(summary.get("fighters"))
    fighter = _mapping(fighters.get(fighter_id))
    stats = {**fighter, **_mapping(fighter.get("stats"))}
    explicit_portrait = str((portrait_urls or {}).get(fighter_id) or "").strip() or None
    portrait_url = explicit_portrait or _safe_run_asset(
        run_dir, fighter.get("portrait_filename")
    )
    profile: dict[str, Any] = {
        "id": fighter_id,
        "corner": "red" if fighter_id == "fighter_a" else "blue",
        "name": str(
            fighter.get("name")
            or ("Боксёр A" if fighter_id == "fighter_a" else "Боксёр B")
        ),
    }
    record = str(fighter.get("record") or "").strip()
    if record:
        profile["record"] = record
    if portrait_url:
        profile["portrait_url"] = portrait_url
    elif _mapping(summary.get("metadata")).get("demo_portraits") is True:
        demo = "demo-bivol.jpg" if fighter_id == "fighter_a" else "demo-usyk.jpg"
        profile["portrait_url"] = gradio_file_url(STATIC_DIR / "assets" / demo)
        profile["demo_portrait"] = True
        profile["portrait_label"] = "Демопортрет, не идентификация спортсмена на видео"
        profile["portrait_attribution"] = (
            "Дмитрий Бивол · Вячеслав Евдокимов / ФК Зенит · CC BY-SA 3.0"
            if fighter_id == "fighter_a" else "Александр Усик · Gabriel Hutchinson / WikiPortraits · CC BY-SA 4.0"
        )
    # Full-fight fallback totals are retained for an old run with no event file.
    accuracy = _number(stats.get("accuracy", stats.get("accuracy_pct")))
    if accuracy > 1.0:
        accuracy /= 100.0
    profile["summary_totals"] = {
        "attempts": _integer(stats.get("attempts")),
        "likely_landed": _integer(stats.get("likely_landed")),
        "blocked": _integer(stats.get("blocked")),
        "missed": _integer(stats.get("missed")),
        "unclear": _integer(stats.get("unclear")),
        "accuracy": round(min(1.0, max(0.0, accuracy)), 4),
        "average_impact_proxy": round(
            _number(stats.get("average_impact_proxy", stats.get("avg_impact"))), 1
        ),
        "landed_targets": _mapping(stats.get("landed_targets")),
        "received_landed_targets": _mapping(stats.get("received_landed_targets")),
    }
    return profile


def build_presentation_payload(
    events: Iterable[Mapping[str, Any] | PunchEvent],
    summary: Mapping[str, Any] | None,
    duration_s: float | None = None,
    *,
    run_dir: str | Path | None = None,
    body_map_asset_url: str | None = None,
    body_map_head_mask_url: str | None = None,
    body_map_body_mask_url: str | None = None,
    portrait_urls: Mapping[str, str] | None = None,
    preview_sprite_url: str | None = None,
    preview_manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Create the compact, JSON-ready contract consumed by the JS workspace."""

    from .quality import apply_result_gate
    summary_map = _mapping(apply_result_gate(_mapping(summary)))
    metadata = _mapping(summary_map.get("metadata"))
    canonical_events = sorted(
        (_compact_event(event) for event in events),
        key=lambda event: (_integer(event.get("peak_ms")), str(event.get("event_id"))),
    )
    inferred_end_ms = max(
        (_integer(event.get("end_ms")) for event in canonical_events), default=0
    )
    duration_ms = max(
        inferred_end_ms,
        round(
            max(0.0, _number(duration_s, _number(metadata.get("duration_s")))) * 1000
        ),
    )
    resolved_run_dir = Path(run_dir).resolve() if run_dir else None
    fighters = {
        fighter_id: _fighter_profile(
            summary_map,
            fighter_id,
            run_dir=resolved_run_dir,
            portrait_urls=portrait_urls,
        )
        for fighter_id in FIGHTER_IDS
    }
    rounds = [
        dict(card)
        for card in summary_map.get("round_scores", [])
        if isinstance(card, Mapping)
    ]
    body_url = body_map_asset_url or gradio_file_url(BODY_MAP_ASSET)
    head_mask_url = body_map_head_mask_url or gradio_file_url(BODY_MAP_HEAD_MASK)
    body_mask_url = body_map_body_mask_url or gradio_file_url(BODY_MAP_BODY_MASK)
    quality = _mapping(summary_map.get("quality"))
    winner = _mapping(summary_map.get("winner")) or {
        "fighter_id": summary_map.get("winner_id"),
        "name": summary_map.get("winner_name"),
        "confidence": _number(summary_map.get("confidence")),
    }
    if quality.get("winner_visible") is False:
        winner = {}

    payload: dict[str, Any] = {
        "version": 1,
        "duration_ms": duration_ms,
        "fighters": fighters,
        "events": canonical_events,
        "rounds": rounds,
        "metadata": {
            "bundled_demo": metadata.get("bundled_demo") is True,
            "demo_read_only": metadata.get("demo_read_only") is True,
            "demo_start_ms": min(duration_ms, max(0, round(_number(metadata.get("demo_start_ms"))))),
            "demo_selected_event_id": str(metadata.get("demo_selected_event_id") or "")
            if any(event["event_id"] == metadata.get("demo_selected_event_id") for event in canonical_events) else None,
            "desktop_theater": metadata.get("desktop_theater") is True,
            "scheduled_rounds": max(
                1,
                _integer(
                    metadata.get("scheduled_rounds"),
                    len(rounds)
                    or max(
                        (_integer(event.get("round"), 1) for event in canonical_events),
                        default=1,
                    ),
                ),
            ),
            "round_length_s": max(0, _integer(metadata.get("round_length_s"), 180)),
            "rest_length_s": max(0, _integer(metadata.get("rest_length_s"), 60)),
            "fight_start_s": max(0.0, _number(metadata.get("fight_start_s"))),
            "fight_end_s": _number(metadata.get("fight_end_s"), duration_ms / 1000.0),
            "output_fps": max(1.0, _number(metadata.get("output_fps"), 30.0)),
        },
        "result": {
            "score_total": _mapping(summary_map.get("score_total")),
            "winner": winner,
            "quality": quality,
            "disclaimer": str(summary_map.get("disclaimer") or "Аналитика модели."),
        },
        "assets": {
            # ``body_map`` remains for older cached workspaces.
            "body_map": body_url,
            "body_map_base": body_url,
            "body_map_head_mask": head_mask_url,
            "body_map_body_mask": body_mask_url,
            "body_map_atlas": {
                "version": 3,
                "coordinate_space": "defender_front_canonical_v1",
                "canvas_size": [1024, 1536],
                "viewport_crop": {"x": 0.18, "y": 0.0, "width": 0.64, "height": 0.72},
            },
        },
    }
    if preview_sprite_url:
        payload["assets"]["preview_sprite"] = preview_sprite_url
    resolved_manifest = _preview_manifest_payload(
        preview_manifest or metadata.get("preview_manifest"),
        run_dir=resolved_run_dir,
    )
    if resolved_manifest:
        payload["preview_manifest"] = resolved_manifest
    return payload


def empty_presentation_payload() -> dict[str, Any]:
    return build_presentation_payload([], {}, 0.0)


def serialize_payload(payload: Mapping[str, Any]) -> str:
    """Compact JSON escaped for a text-only HTML node.

    The JavaScript reads this node through ``textContent``. Escaping therefore
    prevents markup/script injection while producing the original JSON string
    after the browser decodes HTML entities.
    """

    compact = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return escape(compact, quote=False)


def render_fighter_panel_mount(fighter_id: Literal["fighter_a", "fighter_b"]) -> str:
    if fighter_id not in FIGHTER_IDS:
        raise ValueError(f"Unsupported fighter id: {fighter_id}")
    role = "A" if fighter_id == "fighter_a" else "B"
    return (
        f'<aside class="bv-fighter-panel bv-panel-{role.lower()}" '
        f'data-fighter-id="{fighter_id}" aria-label="Панель бойца {role}">'
        '<div class="bv-panel-skeleton" aria-hidden="true">'
        "<span></span><span></span><span></span></div>"
        "</aside>"
    )


def render_workspace_shell(payload: Mapping[str, Any] | None = None) -> str:
    """Render timeline/toolbar DOM; fighter mounts stay beside ``gr.Video``.

    This fragment is safe to replace after analysis or review. The standalone
    panel mounts and Gradio video are not children of it, so they retain their
    state while the payload is refreshed.
    """

    data = serialize_payload(payload or empty_presentation_payload())
    return f"""
    <section class="bv-workspace-chrome" data-bv-workspace-version="2">
      <div class="bv-workspace-bar">
        <div class="bv-workspace-title">
          <span class="bv-live-dot" aria-hidden="true"></span>
          <span>Анализ боя</span>
          <small>События и статистика</small>
        </div>
        <div class="bv-mobile-fighter-switch" role="group" aria-label="Боец на мобильном экране">
          <button type="button" data-bv-mobile-fighter="fighter_a" aria-pressed="true">A</button>
          <button type="button" data-bv-mobile-fighter="fighter_b" aria-pressed="false">B</button>
        </div>
        <button class="bv-theater-button" type="button" data-bv-theater aria-label="Открыть рабочее место на весь экран">На весь экран</button>
      </div>

      <div class="bv-timeline-frame">
        <div class="bv-lane-labels" aria-hidden="true"><span>Раунд</span><span>A</span><span>B</span><span>Проверка</span></div>
        <div class="bv-canvas-scroller" data-bv-canvas-scroller>
          <canvas class="bv-timeline-canvas" data-bv-timeline tabindex="0" role="application" aria-controls="bv-event-listbox" aria-owns="bv-event-listbox" aria-label="Интерактивный таймлайн боя. Стрелки перемещают по кадрам, пробел запускает видео, стрелки вверх и вниз выбирают события."></canvas>
          <div class="bv-timeline-tooltip" data-bv-tooltip role="tooltip" hidden></div>
        </div>
      </div>
      <div class="bv-timeline-toolbar" aria-label="Управление таймлайном">
        <div class="bv-scope-control" role="group" aria-label="Объём метрик">
          <button type="button" data-bv-scope="to_time" aria-pressed="true">До момента</button>
          <button type="button" data-bv-scope="round" aria-pressed="false">Раунд</button>
          <button type="button" data-bv-scope="fight" aria-pressed="false">Весь бой</button>
        </div>
        <details class="bv-more-filters">
          <summary aria-label="Другие фильтры">Фильтры</summary>
          <div class="bv-more-filters-popover bv-filter-row">
            <label>Боец<select data-bv-filter="fighter"><option value="all">A + B</option><option value="fighter_a">A</option><option value="fighter_b">B</option></select></label>
            <label>Раунд<select data-bv-filter="round"><option value="all">Все</option></select></label>
            <label>Исход<select data-bv-filter="outcome"><option value="all">Все</option><option value="likely_landed">Попадание</option><option value="blocked">Блокировано</option><option value="missed">Промах</option><option value="unclear">Не определено</option></select></label>
            <label>Зона<select data-bv-filter="target"><option value="all">Голова + корпус</option><option value="head">Голова</option><option value="body">Корпус</option><option value="unknown">Не определена</option></select></label>
            <label>Техника<select data-bv-filter="technique"><option value="all">Все</option></select></label>
            <label>Рука<select data-bv-filter="hand"><option value="all">Левая + правая</option><option value="left">Левая</option><option value="right">Правая</option></select></label>
            <label>Уверенность<select data-bv-filter="confidence"><option value="0">Все</option><option value="0.6">≥ 60%</option><option value="0.75">≥ 75%</option><option value="0.9">≥ 90%</option></select></label>
            <label>Статус<select data-bv-filter="review"><option value="all">Все</option><option value="unreviewed">На проверку</option><option value="confirmed">Подтверждено</option><option value="rejected">Отклонено</option><option value="replay">Повтор</option></select></label>
          </div>
        </details>
        <div class="bv-time-readout"><span data-bv-time>00:00.000</span><span data-bv-range>По ширине · до момента</span></div>
        <div class="bv-zoom-control" role="group" aria-label="Масштаб таймлайна">
          <button type="button" data-bv-zoom="1" aria-pressed="true">По ширине</button>
          <button type="button" data-bv-zoom="2" aria-pressed="false">2×</button>
          <button type="button" data-bv-zoom="4" aria-pressed="false">4×</button>
          <button type="button" data-bv-zoom="8" aria-pressed="false">8×</button>
        </div>
      </div>

      <div class="bv-inspector" data-bv-inspector data-empty="true">
        <div class="bv-inspector-thumbnail" data-bv-inspector-thumbnail hidden aria-hidden="true"></div>
        <div class="bv-inspector-copy">
          <time class="bv-inspector-time" data-bv-inspector-time>Выберите событие</time>
          <strong data-bv-inspector-title>Нажмите на маркер удара в таймлайне</strong>
          <span class="bv-inspector-result" data-bv-inspector-detail>Здесь появятся исход, уверенность и интенсивность.</span>
          <dl class="bv-inspector-meta">
            <div><dt>Боец</dt><dd data-bv-inspector-fighter>—</dd></div>
            <div><dt>Раунд</dt><dd data-bv-inspector-round>—</dd></div>
            <div><dt>Проверка</dt><dd data-bv-inspector-review>—</dd></div>
          </dl>
        </div>
        <div class="bv-inspector-controls">
          <div class="bv-inspector-navigation">
            <button type="button" data-bv-event-nav="-1" aria-label="Предыдущий удар">Предыдущий</button>
            <button type="button" data-bv-event-nav="1" aria-label="Следующий удар">Следующий</button>
          </div>
          <details class="bv-inspector-review" data-bv-review-menu>
            <summary>Проверить</summary>
            <div class="bv-review-popover">
              <p data-bv-inspector-note>Сверьте событие с видео перед подтверждением.</p>
              <div class="bv-inspector-actions">
                <button type="button" data-bv-review="confirmed" aria-label="Подтвердить событие">Подтвердить</button>
                <button type="button" data-bv-review="rejected" aria-label="Отклонить событие">Отклонить</button>
                <button type="button" data-bv-review="real_not_replay" aria-label="Подтвердить, что событие не повтор">Это не повтор</button>
              </div>
            </div>
          </details>
        </div>
      </div>
      <div id="bv-event-listbox" class="bv-a11y-listbox" data-bv-a11y-listbox role="listbox" aria-label="События таймлайна"></div>
      <div class="bv-selected-event-sr" data-bv-live aria-live="polite" aria-atomic="true"></div>
      <div class="bv-payload" data-bv-payload hidden>{data}</div>
    </section>
    """


def render_workspace_fragments(
    events: Iterable[Mapping[str, Any] | PunchEvent],
    summary: Mapping[str, Any] | None,
    duration_s: float | None = None,
    **payload_kwargs: Any,
) -> tuple[str, str, str]:
    """Convenience API for the three Gradio HTML outputs."""

    payload = build_presentation_payload(events, summary, duration_s, **payload_kwargs)
    return (
        render_fighter_panel_mount("fighter_a"),
        render_fighter_panel_mount("fighter_b"),
        render_workspace_shell(payload),
    )


__all__ = [
    "BODY_MAP_ASSET",
    "BODY_MAP_BODY_MASK",
    "BODY_MAP_HEAD_MASK",
    "STATIC_DIR",
    "aggregate_fighter",
    "build_presentation_payload",
    "empty_presentation_payload",
    "filter_events_for_scope",
    "gradio_file_url",
    "render_fighter_panel_mount",
    "render_workspace_fragments",
    "render_workspace_shell",
    "serialize_payload",
]
