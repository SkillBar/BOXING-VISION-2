from __future__ import annotations

import gzip
import json
import os
import uuid
import zipfile
from html import escape
from pathlib import Path
from threading import Event

import cv2
import gradio as gr
import numpy as np

from .artifacts import atomic_write_json
from .calibration import (
    box_iou,
    calibration_backend,
    confirm_enrollment,
    enrollment_support_keypoints,
    propose_roles,
    render_enrollment_view,
)
from .config import AnalysisConfig, validate_working_region
from .contracts import PunchEvent
from .local_fonts import installed_display_font_assets
from .pipeline import (
    AnalysisCancelledError,
    RenderCacheUnavailableError,
    analyze_video,
    rebuild_from_cache,
)
from .scoring import build_fight_summary
from .ui_presenters import (
    STATIC_DIR,
    build_presentation_payload,
    render_fighter_panel_mount,
    render_workspace_shell,
)
from .video import validate_video

_GRADIO_PROGRESS = gr.Progress()
_CANCEL_EVENT = Event()
_STATIC_CSS = STATIC_DIR / "boxing_vision.css"
_STATIC_JS = STATIC_DIR / "boxing_vision.js"
_STATIC_CSS_SOURCE = _STATIC_CSS.read_text(encoding="utf-8")

CSS = r"""
:root {
  --arena-black: #090a0d;
  --arena-panel: #111319;
  --arena-panel-2: #171a21;
  --arena-line: rgba(255,255,255,.10);
  --arena-text: #f4f1ea;
  --arena-muted: #9da3af;
  --arena-red: #ed3f3f;
  --arena-blue: #38a6ff;
  --arena-gold: #efb84d;
}
body, .gradio-container {
  background: var(--arena-black) !important;
  color: var(--arena-text) !important;
}
.gradio-container {
  max-width: 1480px !important;
  padding: 0 28px 56px !important;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
#hero {
  position: relative;
  overflow: hidden;
  margin: 0 -28px 20px;
  min-height: 230px;
  padding: 48px 48px 38px;
  border-bottom: 1px solid var(--arena-line);
  background:
    radial-gradient(circle at 84% 10%, rgba(56,166,255,.16), transparent 31%),
    radial-gradient(circle at 67% 95%, rgba(237,63,63,.18), transparent 35%),
    linear-gradient(115deg, #0a0b0e 0%, #11131a 58%, #090a0d 100%);
}
#hero::after {
  content: "";
  position: absolute;
  inset: 0;
  opacity: .23;
  background-image: repeating-linear-gradient(90deg, transparent 0 79px, rgba(255,255,255,.05) 80px);
  pointer-events: none;
}
.hero-kicker { color: var(--arena-gold); font-size: 12px; letter-spacing: .24em; text-transform: uppercase; }
.hero-title { margin: 10px 0 6px; font-size: clamp(46px, 6.5vw, 92px); line-height: .9; letter-spacing: -.065em; font-weight: 850; }
.hero-title span:first-child { color: var(--arena-red); }
.hero-title span:last-child { color: var(--arena-blue); }
.hero-sub { max-width: 760px; color: #c8ccd5; font-size: 16px; line-height: 1.55; }
.hero-badge { display: inline-flex; margin-top: 18px; padding: 7px 10px; border: 1px solid rgba(239,184,77,.42); color: #f6d28b; background: rgba(239,184,77,.08); border-radius: 3px; font-size: 12px; }
#research-note {
  margin-bottom: 20px;
  padding: 12px 15px;
  border-left: 3px solid var(--arena-gold);
  color: #d7dae0;
  background: rgba(239,184,77,.06);
  font-size: 13px;
}
.section-label { margin: 4px 0 12px; color: var(--arena-muted); font-size: 11px; letter-spacing: .17em; text-transform: uppercase; }
.arena-card, #upload-card, #settings-card, #result-video-card, #stats-card, #timeline-card, #events-card, #downloads-card {
  border: 1px solid var(--arena-line) !important;
  border-radius: 5px !important;
  background: linear-gradient(180deg, rgba(24,27,34,.97), rgba(14,16,21,.98)) !important;
  box-shadow: 0 18px 54px rgba(0,0,0,.25) !important;
}
#upload-card, #settings-card { padding: 14px !important; }
#settings-card .form { gap: 10px; }
#analyze-button {
  min-height: 54px;
  border: 0 !important;
  border-radius: 3px !important;
  background: linear-gradient(90deg, #d82f36, #ef4e42 46%, #dc3745) !important;
  color: white !important;
  font-weight: 800 !important;
  letter-spacing: .04em;
  box-shadow: 0 10px 28px rgba(216,47,54,.25) !important;
}
#analyze-button:hover { filter: brightness(1.08); transform: translateY(-1px); }
#job-status { min-height: 0; }
.status-ready, .status-done { margin: 12px 0; padding: 12px 14px; border: 1px solid var(--arena-line); background: rgba(255,255,255,.03); color: #c8ccd5; }
.status-done { border-color: rgba(81,199,128,.34); color: #bde8cc; background: rgba(81,199,128,.07); }
.meta-grid { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 8px; margin-top: 8px; }
.meta-cell { padding: 9px 10px; border: 1px solid var(--arena-line); background: rgba(255,255,255,.025); }
.meta-cell b { display:block; color:#fff; font-size:14px; }
.meta-cell span { color:var(--arena-muted); font-size:10px; letter-spacing:.08em; text-transform:uppercase; }
.result-header { display:flex; align-items:end; justify-content:space-between; margin: 30px 0 12px; }
.result-header h2 { margin:0; font-size:32px; letter-spacing:-.035em; }
.result-header span { color:var(--arena-gold); font-size:11px; letter-spacing:.12em; text-transform:uppercase; }
.winner-card { position:relative; overflow:hidden; padding:20px; border:1px solid rgba(239,184,77,.25); background:linear-gradient(120deg,rgba(239,184,77,.10),rgba(255,255,255,.02)); }
.winner-eyebrow { color:var(--arena-gold); font-size:10px; letter-spacing:.16em; text-transform:uppercase; }
.winner-name { margin:5px 0 2px; font-size:28px; font-weight:800; letter-spacing:-.035em; }
.winner-score { color:#d7dae0; font-variant-numeric:tabular-nums; }
.quality-line { margin-top:10px; color:var(--arena-muted); font-size:12px; }
.fighter-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }
.fighter-card { padding:14px; border:1px solid var(--arena-line); background:rgba(255,255,255,.025); }
.fighter-card.red { border-top:3px solid var(--arena-red); }
.fighter-card.blue { border-top:3px solid var(--arena-blue); }
.fighter-name { font-size:17px; font-weight:750; }
.metric-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:7px; margin-top:10px; }
.metric { padding:8px; background:rgba(0,0,0,.22); }
.metric b { display:block; color:#fff; font-size:18px; }
.metric span { color:var(--arena-muted); font-size:9px; letter-spacing:.08em; text-transform:uppercase; }
.tech-line { margin-top:9px; color:#c8ccd5; font-size:11px; line-height:1.65; }
.timeline-shell { padding:16px 10px 8px; }
.timeline-track { position:relative; height:24px; margin:12px 10px 8px; border-top:2px solid rgba(255,255,255,.18); }
.timeline-dot { position:absolute; top:-6px; width:10px; height:10px; border-radius:50%; transform:translateX(-50%); box-shadow:0 0 0 3px rgba(9,10,13,.8); }
.timeline-dot.landed { background:#5cd38a; }
.timeline-dot.blocked { background:var(--arena-blue); }
.timeline-dot.missed { background:#f08c58; }
.timeline-dot.unclear { background:#8d94a3; }
.timeline-axis { display:flex; justify-content:space-between; color:var(--arena-muted); font-size:10px; }
.timeline-legend { display:flex; gap:14px; flex-wrap:wrap; color:#b9bec8; font-size:11px; }
.legend-dot { display:inline-block; width:7px; height:7px; margin-right:5px; border-radius:50%; }
#events-table { font-size: 12px; }
#events-table .table-wrap { border-radius: 3px !important; }
#downloads-card { padding:14px !important; }
.confirmation-note { margin:8px 0; padding:10px 12px; border:1px solid var(--arena-line); background:rgba(255,255,255,.025); color:#c8ccd5; font-size:12px; line-height:1.5; }
.confirmation-note.ready { border-color:rgba(81,199,128,.38); color:#bde8cc; background:rgba(81,199,128,.07); }
.round-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:7px; margin-top:10px; }
.round-card { padding:9px 10px; border:1px solid var(--arena-line); background:rgba(0,0,0,.20); }
.round-card b { display:block; color:#fff; font-size:14px; }
.round-card span { color:var(--arena-muted); font-size:10px; line-height:1.4; }
.timeline-controls { margin-top:8px; }
.legal-foot { margin-top:18px; color:#7f8591; font-size:11px; line-height:1.5; }
@media (max-width: 860px) {
  .gradio-container { padding:0 14px 32px !important; }
  #hero { margin:0 -14px 14px; padding:34px 22px 28px; }
  .meta-grid { grid-template-columns:1fr 1fr; }
  .fighter-grid { grid-template-columns:1fr; }
}
"""


TECHNIQUE_LABELS = {
    "jab": "джеб",
    "cross": "кросс",
    "straight": "прямой",
    "hook": "хук",
    "uppercut": "апперкот",
    "unknown": "неясно",
}
TARGET_LABELS = {"head": "голова", "body": "корпус", "unknown": "неясно"}
OUTCOME_LABELS = {
    "likely_landed": "вероятное попадание",
    "blocked": "блок",
    "missed": "промах",
    "unclear": "неясно",
}


def _video_path(value: object) -> str:
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    if not value:
        raise ValueError("Сначала загрузите видео")
    return str(value)


def _optional_file_path(value: object) -> str | None:
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    if hasattr(value, "path"):
        value = value.path
    return str(value) if value else None


def _empty_anchor_state() -> dict[str, object]:
    return {"fighter_a": None, "fighter_b": None, "base_image": None, "frame_time_s": 0.0}


def _prepare_confirmation(
    value: object,
    start_s: float = 0.0,
) -> tuple[str, np.ndarray | None, dict[str, object], str]:
    """Extract one lightweight still on which the user identifies both fighters."""

    state = _empty_anchor_state()
    if not value:
        return (
            _inspect_video(value),
            None,
            state,
            '<div class="confirmation-note">После загрузки укажите на кадре сначала красного, затем синего бойца.</div>',
        )
    try:
        source = _video_path(value)
        metadata = validate_video(source)
        requested_start = max(0.0, float(start_s or 0.0))
        frame_time_s = min(max(0.0, metadata.duration_s - 0.05), requested_start)
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError("OpenCV не смог открыть кадр подтверждения")
        try:
            if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            capture.set(cv2.CAP_PROP_POS_MSEC, frame_time_s * 1000.0)
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_MSEC, 0.0)
                ok, frame = capture.read()
                frame_time_s = 0.0
            if not ok or frame is None:
                raise RuntimeError("Не удалось декодировать кадр подтверждения")
        finally:
            capture.release()

        # Auto orientation is disabled above, so metadata is applied exactly
        # once even for square frames and 180-degree rotations.
        if metadata.rotation in {90, 270}:
            frame = cv2.rotate(
                frame,
                cv2.ROTATE_90_COUNTERCLOCKWISE
                if metadata.rotation == 90
                else cv2.ROTATE_90_CLOCKWISE,
            )
        elif metadata.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        height, width = frame.shape[:2]
        if width > 1100:
            scale = 1100.0 / width
            frame = cv2.resize(frame, (1100, max(2, round(height * scale))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        state["base_image"] = rgb.copy()
        state["frame_time_s"] = round(frame_time_s, 3)
        return (
            _inspect_video(value),
            rgb,
            state,
            (
                '<div class="confirmation-note">Кадр '
                f'{_format_time(round(frame_time_s * 1000))}: нажмите на корпус бойца красного угла, '
                "затем — синего. Это исключает рефери из первичной привязки ID.</div>"
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return (
            _inspect_video(value),
            None,
            state,
            f'<div class="confirmation-note">Не удалось подготовить кадр: {escape(str(exc))}</div>',
        )


def _apply_anchor_selection(
    image: np.ndarray,
    state: dict[str, object] | None,
    point: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object], str]:
    current = dict(state or _empty_anchor_state())
    base_value = current.get("base_image")
    base = np.asarray(base_value if isinstance(base_value, np.ndarray) else image).copy()
    if base.ndim != 3 or base.shape[2] < 3:
        raise ValueError("Кадр подтверждения недоступен")
    height, width = base.shape[:2]
    x = int(np.clip(point[0], 0, max(0, width - 1)))
    y = int(np.clip(point[1], 0, max(0, height - 1)))
    if current.get("fighter_a") is not None and current.get("fighter_b") is not None:
        current["fighter_a"] = None
        current["fighter_b"] = None
    fighter_id = "fighter_a" if current.get("fighter_a") is None else "fighter_b"
    current[fighter_id] = (
        x / max(1, width - 1),
        y / max(1, height - 1),
    )
    current["base_image"] = base.copy()

    rendered = base.copy()
    for key, color, label in (
        ("fighter_a", (237, 63, 63), "A · RED"),
        ("fighter_b", (56, 166, 255), "B · BLUE"),
    ):
        anchor = current.get(key)
        if not isinstance(anchor, (tuple, list)) or len(anchor) != 2:
            continue
        px = round(float(anchor[0]) * max(1, width - 1))
        py = round(float(anchor[1]) * max(1, height - 1))
        cv2.circle(rendered, (px, py), 17, color, 4, cv2.LINE_AA)
        cv2.putText(
            rendered,
            label,
            (min(max(4, px + 20), max(4, width - 120)), max(24, py - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
    if current.get("fighter_b") is None:
        message = '<div class="confirmation-note">Красный угол отмечен. Теперь нажмите на синего бойца.</div>'
    else:
        message = '<div class="confirmation-note ready">Оба бойца подтверждены: A — красный, B — синий. Можно запускать анализ.</div>'
    return rendered, current, message


def _select_fighter_anchor(
    image: np.ndarray,
    state: dict[str, object] | None,
    event: gr.SelectData,
) -> tuple[np.ndarray, dict[str, object], str]:
    index = event.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise gr.Error("Нажмите внутри кадра подтверждения")
    return _apply_anchor_selection(image, state, (int(index[0]), int(index[1])))


def _empty_enrollment_state() -> dict[str, object]:
    return {"views": [], "ring_points": [], "confirmed": False, "active_index": 0, "region_mode": "none"}


def _enrollment_region_mode(state: dict[str, object]) -> str:
    # States created before the optional-region UI required a manual polygon.
    mode = str(state.get("region_mode", "manual"))
    if mode not in {"none", "manual"}:
        raise ValueError("Выберите режим рабочей области")
    return mode


def _region_role_choices(mode: str):
    roles = [("Красный A", "fighter_a"), ("Синий B", "fighter_b")]
    return roles + ([("Рабочая область", "ring")] if mode == "manual" else [])


def _change_working_region(state: dict[str, object] | None, mode: str):
    current = dict(state or _empty_enrollment_state(), region_mode=mode, confirmed=False)
    _enrollment_region_mode(current)
    current.pop("enrollment_samples", None)
    if mode == "none":
        current["ring_points"] = []
    image = None
    if len(current.get("views", [])) == 3:
        image, _ = render_enrollment_view(current, int(current.get("active_index", 0)))
    note = ("Без ограничения области: подтвердите только A и B на трёх кадрах."
            if mode == "none" else "На первом кадре отметьте четыре точки пола вокруг бойцов, а не полосу канатов.")
    return image, current, _enrollment_note(note), gr.update(choices=_region_role_choices(mode), value="fighter_a")


def _enrollment_note(message: str, confirmed: bool = False) -> str:
    return f'<div class="confirmation-note{" ready" if confirmed else ""}">{escape(message)}</div>'


def _enrollment_sample(value: object, time_s: float) -> dict[str, object]:
    _, image, extracted, _ = _prepare_confirmation(value, time_s)
    if image is None:
        raise ValueError("Не удалось прочитать калибровочный кадр")
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    poses = calibration_backend().infer(frame)
    boxes = [[float(pose.bbox.x1), float(pose.bbox.y1), float(pose.bbox.x2), float(pose.bbox.y2)] for pose in poses]
    selection = propose_roles(frame, poses)
    selected = [selection.get("fighter_a"), selection.get("fighter_b")]
    if all(isinstance(index, int) and 0 <= index < len(poses) for index in selected) and selected[0] != selected[1]:
        pair = [poses[index] for index in selected]
        overlap = box_iou(pair[0].bbox, pair[1].bbox)
        quality = min(float(getattr(pose, "confidence", 0.0)) for pose in pair) - overlap
    else:
        quality = -1.0
    return {"image": image, "time_s": float(extracted["frame_time_s"]), "boxes": boxes,
            "selection": selection, "quality": quality,
            "support_keypoints": enrollment_support_keypoints(poses, image.shape[1], image.shape[0])}


def _prepare_enrollment(value: object, start_s: float = 0.0, progress: gr.Progress = _GRADIO_PROGRESS):
    state = _empty_enrollment_state()
    if not value:
        return (_inspect_video(value), None, state, _enrollment_note("Загрузите видео, чтобы предложить три кадра для подтверждения бойцов."), gr.update(value=0), 0.0)
    try:
        source = _video_path(value)
        metadata = validate_video(source)
        start = max(0.0, float(start_s or 0))
        if start >= metadata.duration_s - .1:
            raise ValueError("Начало боя должно находиться внутри видео")
        end = min(metadata.duration_s - .05, start + 15)
        samples = []
        for index, time_s in enumerate(np.linspace(start, end, 9)):
            progress(index / 9, desc=f"Ищем чистые кадры: {index + 1}/9")
            samples.append(_enrollment_sample(value, float(time_s)))
        # One view per temporal third; this does not assign identities by size
        # or screen position. Only the color proposal supplies tentative A/B.
        views = [max(samples[offset:offset + 3], key=lambda view: float(view["quality"])) for offset in (0, 3, 6)]
        state.update({"views": views, "start_s": start, "source": str(Path(source).resolve())})
        image, message = render_enrollment_view(state, 0)
        progress(1, desc="Проверьте три предложенных кадра")
        return (_inspect_video(value), image, state, _enrollment_note(message), gr.update(value=0), float(views[0]["time_s"]))
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        return (_inspect_video(value), None, state, _enrollment_note(f"Не удалось предложить кадры: {exc}"), gr.update(value=0), float(start_s or 0))


def _prepare_enrollment_for_region(value: object, start_s: float, region_mode: str,
                                   progress: gr.Progress = _GRADIO_PROGRESS):
    result = list(_prepare_enrollment(value, start_s, progress))
    result[2] = dict(result[2], region_mode=region_mode)
    _enrollment_region_mode(result[2])
    return tuple(result)


def _show_enrollment_frame(state: dict[str, object] | None, index: float):
    current = dict(state or _empty_enrollment_state())
    views = current.get("views", [])
    if len(views) != 3:
        return None, current, _enrollment_note("Сначала загрузите видео."), 0.0
    selected = max(0, min(2, int(index or 0)))
    current["active_index"] = selected
    image, message = render_enrollment_view(current, selected)
    return image, current, _enrollment_note(message, bool(current.get("confirmed"))), float(views[selected]["time_s"])


def _apply_enrollment_selection(state: dict[str, object], role: str, point: tuple[int, int]):
    current = dict(state)
    views = [dict(view, selection=dict(view["selection"])) for view in current.get("views", [])]
    if len(views) != 3:
        raise ValueError("Сначала предложите три калибровочных кадра")
    selected = int(current.get("active_index", 0))
    view = views[selected]
    height, width = np.asarray(view["image"]).shape[:2]
    x, y = float(point[0]), float(point[1])
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Нажмите внутри изображения")
    next_role = role
    if role == "ring":
        if _enrollment_region_mode(current) != "manual":
            raise ValueError("Сначала включите режим «Рабочая область»")
        if selected != 0:
            raise ValueError("Рабочая область отмечается на первом кадре")
        points = list(current.get("ring_points", []))
        if len(points) >= 4:
            points = []
        points.append((x / width, y / height))
        current["ring_points"] = points
    elif role in {"fighter_a", "fighter_b"}:
        candidates = [(index, box) for index, box in enumerate(view["boxes"]) if box[0] <= x <= box[2] and box[1] <= y <= box[3]]
        if not candidates:
            raise ValueError("Человек здесь не обнаружен. Нажмите внутри настоящей рамки или замените кадр.")
        candidate = min(candidates, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))[0]
        other = "fighter_b" if role == "fighter_a" else "fighter_a"
        if view["selection"].get(other) == candidate:
            raise ValueError("Один человек не может быть обоими бойцами. Сначала исправьте выбор другого угла.")
        view["selection"][role] = candidate
        next_role = other
    else:
        raise ValueError("Выберите бойца или область ринга")
    current.update({"views": views, "confirmed": False})
    current.pop("enrollment_samples", None)
    image, message = render_enrollment_view(current, selected)
    if role == "ring":
        message = f"Рабочая область: {len(current['ring_points'])}/4 точки. Отметьте пол вокруг бойцов по периметру."
    return image, current, _enrollment_note(message), gr.update(value=next_role)


def _select_enrollment_target(state: dict[str, object], role: str, event: gr.SelectData):
    try:
        if not isinstance(event.index, (tuple, list)) or len(event.index) < 2:
            raise ValueError("Нажмите внутри изображения")
        return _apply_enrollment_selection(state, role, (int(event.index[0]), int(event.index[1])))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def _clear_enrollment_selection(state: dict[str, object], role: str):
    current = dict(state)
    views = [dict(view, selection=dict(view["selection"])) for view in current.get("views", [])]
    if len(views) != 3:
        raise gr.Error("Сначала загрузите видео")
    index = int(current.get("active_index", 0))
    if role == "ring":
        current["ring_points"] = []
    else:
        views[index]["selection"] = {"fighter_a": None, "fighter_b": None}
    current.update({"views": views, "confirmed": False})
    current.pop("enrollment_samples", None)
    image, message = render_enrollment_view(current, index)
    return image, current, _enrollment_note(message), gr.update(value="ring" if role == "ring" else "fighter_a")


def _replace_enrollment_frame(value: object, state: dict[str, object], time_s: float):
    current = dict(state)
    views = list(current.get("views", []))
    if len(views) != 3:
        raise gr.Error("Сначала загрузите видео")
    metadata = validate_video(_video_path(value))
    if not float(current.get("start_s", 0)) <= float(time_s) < metadata.duration_s:
        raise gr.Error("Выберите время внутри выбранного боя")
    selected = int(current.get("active_index", 0))
    views[selected] = _enrollment_sample(value, float(time_s))
    current.update({"views": views, "confirmed": False})
    current.pop("enrollment_samples", None)
    if selected == 0:
        current["ring_points"] = []
    image, message = render_enrollment_view(current, selected)
    return image, current, _enrollment_note(message)


def _confirm_enrollment_ui(state: dict[str, object]):
    try:
        mode = _enrollment_region_mode(state)
        current = dict(state, region_mode=mode)
        if mode == "manual":
            validate_working_region(state.get("ring_points", []))
        else:
            current["ring_points"] = []
        times = [float(view["time_s"]) for view in state.get("views", [])]
        if len(times) != 3 or len(set(times)) != 3:
            raise ValueError("Подтвердите три разных кадра")
        current, _ = confirm_enrollment(current)
        image, message = render_enrollment_view(current, int(current.get("active_index", 0)))
        return image, current, _enrollment_note(message, True)
    except (ValueError, TypeError, IndexError) as exc:
        raise gr.Error(str(exc)) from exc


def _enrollment_config_fields(state: dict[str, object] | None, source: str, start_s: float) -> dict[str, object]:
    current = state or {}
    if not current.get("confirmed") or len(current.get("enrollment_samples", [])) != 3:
        raise ValueError("Проверьте бойцов, затем нажмите «Подтвердить три кадра»")
    if current.get("source") != str(Path(source).resolve()) or abs(float(current.get("start_s", 0)) - float(start_s)) > .001:
        raise ValueError("Видео или начало боя изменилось. Повторите подтверждение трёх кадров")
    samples = tuple(dict(sample) for sample in current["enrollment_samples"])
    mode = _enrollment_region_mode(current)
    points = tuple(tuple(point) for point in current.get("ring_points", [])) if mode == "manual" else ()
    if mode == "manual":
        validate_working_region(points)
    return {"enrollment_mode": "auto_confirm", "enrollment_samples": samples,
            "enrollment_frames": tuple(float(sample["time_s"]) for sample in samples),
            "enrollment_confirmed": True, "region_mode": mode, "ring_rois": points}


def _confirmed_anchors(state: dict[str, object] | None) -> tuple[tuple[float, float], tuple[float, float]]:
    current = state or {}
    fighter_a = current.get("fighter_a")
    fighter_b = current.get("fighter_b")
    if not isinstance(fighter_a, (tuple, list)) or not isinstance(fighter_b, (tuple, list)):
        raise TypeError("На кадре подтверждения укажите красного и синего бойца")
    return (
        (float(fighter_a[0]), float(fighter_a[1])),
        (float(fighter_b[0]), float(fighter_b[1])),
    )


def _parse_round_list(value: str, scheduled_rounds: int) -> tuple[int, ...]:
    raw = (value or "").replace(";", ",").replace(" ", ",")
    rounds: list[int] = []
    for token in raw.split(","):
        if not token.strip():
            continue
        try:
            round_number = int(token)
        except ValueError as exc:
            raise ValueError("Раунды нокдаунов указываются числами через запятую") from exc
        if not 1 <= round_number <= scheduled_rounds:
            raise ValueError(f"Раунд нокдауна должен быть от 1 до {scheduled_rounds}")
        rounds.append(round_number)
    return tuple(rounds)


def _request_cancel() -> str:
    _CANCEL_EVENT.set()
    return '<div class="status-ready">Запрошена отмена. Текущий этап будет безопасно остановлен.</div>'


def _format_time(milliseconds: int) -> str:
    seconds = max(0, milliseconds) / 1000.0
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds % 60:04.1f}"


def _inspect_video(value: object) -> str:
    if not value:
        return '<div class="status-ready">Загрузите локальный видеофайл. Он не покинет этот Mac.</div>'
    try:
        metadata = validate_video(_video_path(value))
    except (OSError, RuntimeError, ValueError) as exc:
        return f'<div class="status-ready">Не удалось проверить видео: {escape(str(exc))}</div>'
    return f"""
    <div class="meta-grid">
      <div class="meta-cell"><b>{metadata.duration_s / 60:.1f} мин</b><span>длительность</span></div>
      <div class="meta-cell"><b>{metadata.display_width}×{metadata.display_height}</b><span>разрешение</span></div>
      <div class="meta-cell"><b>{metadata.fps:.1f} fps</b><span>частота кадров</span></div>
      <div class="meta-cell"><b>{'есть' if metadata.has_audio else 'нет'}</b><span>звук</span></div>
    </div>
    <div class="legal-foot">A — красный угол, B — синий. Подтвердите личности на трёх кадрах; положение слева или справа не меняет угол бойца.</div>
    """


def _fighter_card(fighter: dict[str, object], corner: str) -> str:
    stats = fighter.get("stats") if isinstance(fighter.get("stats"), dict) else fighter
    assert isinstance(stats, dict)
    accuracy = float(stats.get("accuracy", stats.get("accuracy_pct", 0)) or 0)
    if accuracy <= 1:
        accuracy *= 100
    impact = float(stats.get("average_impact_proxy", stats.get("avg_impact", 0)) or 0)
    techniques = stats.get("techniques") if isinstance(stats.get("techniques"), dict) else {}
    tech_text = " · ".join(
        f"{TECHNIQUE_LABELS.get(str(key), str(key))}: {value}" for key, value in techniques.items()
    ) or "пока нет уверенных типов"
    return f"""
    <div class="fighter-card {corner}">
      <div class="fighter-name">{escape(str(fighter.get('name', 'Боксёр')))}</div>
      <div class="metric-grid">
        <div class="metric"><b>{int(stats.get('attempts', 0) or 0)}</b><span>кандидаты ударов</span></div>
        <div class="metric"><b>{int(stats.get('likely_landed', 0) or 0)}</b><span>вероятные попадания</span></div>
        <div class="metric"><b>{accuracy:.0f}%</b><span>оценка точности</span></div>
        <div class="metric"><b>{impact:.0f}</b><span>относительная интенсивность / 100</span></div>
      </div>
      <div class="tech-line">Блоки: {int(stats.get('blocked', 0) or 0)} · Промахи: {int(stats.get('missed', 0) or 0)} · Неясно: {int(stats.get('unclear', 0) or 0)}<br>{escape(tech_text)}</div>
    </div>
    """


def _overview(summary: dict[str, object]) -> str:
    from .quality import apply_result_gate, result_eligibility

    allowed, block_reasons = result_eligibility(summary)
    summary = apply_result_gate(summary)
    fighters = summary.get("fighters") if isinstance(summary.get("fighters"), dict) else {}
    assert isinstance(fighters, dict)
    fighter_a = fighters.get("fighter_a") if isinstance(fighters.get("fighter_a"), dict) else {}
    fighter_b = fighters.get("fighter_b") if isinstance(fighters.get("fighter_b"), dict) else {}
    assert isinstance(fighter_a, dict) and isinstance(fighter_b, dict)
    winner = summary.get("winner") if isinstance(summary.get("winner"), dict) else {}
    assert isinstance(winner, dict)
    score = summary.get("score_total") if isinstance(summary.get("score_total"), dict) else {}
    quality = summary.get("quality") if isinstance(summary.get("quality"), dict) else {}
    total_a = int(score.get("fighter_a", 0) or 0)
    total_b = int(score.get("fighter_b", 0) or 0)
    winner_name = escape(str(winner.get("name") or "Недостаточно данных"))
    winner_confidence = float(winner.get("confidence", 0) or 0)
    tracking = float(quality.get("tracking_confidence", 0) or 0)
    event_confidence = float(quality.get("event_confidence", 0) or 0)
    raw_round_scores = summary.get("round_scores")
    round_scores = raw_round_scores if isinstance(raw_round_scores, list) else []
    round_cards = "".join(
        (
            '<div class="round-card">'
            f'<b>Раунд {int(card.get("round", 1) or 1)} · '
            f'{int(card.get("fighter_a_points", 0) or 0)}—{int(card.get("fighter_b_points", 0) or 0)}</b>'
            f'<span>confidence {float(card.get("confidence", 0) or 0):.0%}<br>'
            f'{escape(str(card.get("reason", "Оценка модели")))}</span></div>'
        )
        for card in round_scores
        if isinstance(card, dict)
    )
    quality_warning = (
        " · низкая уверенность, результат требует ручной проверки"
        if tracking < 0.55 or event_confidence < 0.55
        else ""
    )
    motion_counts = ""
    if "motion_proposals" in quality:
        motions = max(0, int(quality.get("motion_proposals", 0) or 0))
        unresolved = max(0, int(quality.get("unresolved_motion_proposals", 0) or 0))
        candidates = max(0, int(quality.get("event_candidates", 0) or 0))
        landed = sum(max(0, int(fighter.get("likely_landed", 0) or 0)) for fighter in (fighter_a, fighter_b))
        motion_counts = (
            '<div class="quality-line">'
            f'Движения на проверку: {motions} · без подтверждённой личности: {unresolved}<br>'
            f'Кандидаты ударов: {candidates} · вероятные попадания: {landed}'
            '</div>'
        )
    if not allowed:
        return (
            '<div class="winner-card"><div class="winner-name">Итоги требуют проверки</div>'
            '<div class="quality-line">' + escape(" · ".join(block_reasons)) + '</div>' + motion_counts + '</div>'
            '<div class="fighter-grid">' + _fighter_card(fighter_a, 'red')
            + _fighter_card(fighter_b, 'blue') + '</div>'
        )
    return f"""
    <div class="winner-card">
      <div class="winner-eyebrow">Прогноз модели</div>
      <div class="winner-name">{winner_name}</div>
      <div class="winner-score">Сумма карточек: {total_a} — {total_b} · уверенность {winner_confidence:.0%}</div>
      <div class="quality-line">Качество трекинга {tracking:.0%} · средняя уверенность событий {event_confidence:.0%}{quality_warning}</div>
      {motion_counts}
    </div>
    <div class="fighter-grid">
      {_fighter_card(fighter_a, 'red')}
      {_fighter_card(fighter_b, 'blue')}
    </div>
    <div class="round-grid">{round_cards}</div>
    """


def _timeline(events: list[dict[str, object]], duration_s: float) -> str:
    duration_ms = max(1.0, duration_s * 1000.0)
    # Rendering thousands of DOM markers hurts the local UI. Keep the strongest
    # markers while the complete event table remains available below.
    visible = sorted(
        [
            event
            for event in events
            if str(event.get("review_status", "unreviewed")) not in {"rejected", "deleted"}
        ],
        key=lambda event: float(event.get("confidence", 0) or 0),
        reverse=True,
    )[:240]
    markers: list[str] = []
    for event in visible:
        position = min(100.0, max(0.0, float(event.get("peak_ms", 0) or 0) / duration_ms * 100.0))
        outcome = str(event.get("outcome", "unclear"))
        css_class = "landed" if outcome == "likely_landed" else outcome
        title = escape(
            f"{_format_time(int(event.get('peak_ms', 0) or 0))} · "
            f"{TECHNIQUE_LABELS.get(str(event.get('technique')), str(event.get('technique')))} · "
            f"{OUTCOME_LABELS.get(outcome, outcome)}"
        )
        markers.append(
            f'<span class="timeline-dot {css_class}" style="left:{position:.3f}%" title="{title}"></span>'
        )
    return f"""
    <div class="timeline-shell">
      <div class="timeline-legend">
        <span><i class="legend-dot" style="background:#5cd38a"></i>вероятное попадание</span>
        <span><i class="legend-dot" style="background:#38a6ff"></i>блок</span>
        <span><i class="legend-dot" style="background:#f08c58"></i>промах</span>
        <span><i class="legend-dot" style="background:#8d94a3"></i>неясно</span>
      </div>
      <div class="timeline-track">{''.join(markers)}</div>
      <div class="timeline-axis"><span>00:00</span><span>{_format_time(round(duration_ms))}</span></div>
    </div>
    """


def _workspace(
    events: list[dict[str, object]],
    summary: dict[str, object],
    duration_s: float,
    run_dir: str | os.PathLike[str] | None = None,
) -> str:
    resolved_run = Path(run_dir).expanduser().resolve() if run_dir else None
    presentation_summary = dict(summary)
    presentation_metadata = dict(summary.get("metadata") or {})
    # Demo portraits illustrate only the explicitly configured demo run. They
    # are not identities inferred from an uploaded fight and are not persisted.
    demo_run = os.environ.get("BOXING_VISION_DEMO_RUN")
    presentation_metadata["demo_portraits"] = bool(
        demo_run and resolved_run is not None
        and os.environ.get("BOXING_VISION_OPEN_SAVED_RUN") != "1"
        and Path(demo_run).expanduser().resolve() == resolved_run
    )
    presentation_metadata["desktop_theater"] = bool(
        presentation_metadata.get("bundled_demo") is True
        and os.environ.get("BOXING_VISION_DESKTOP") == "1"
    )
    presentation_summary["metadata"] = presentation_metadata
    preview_manifest = (
        resolved_run / "preview_manifest.json"
        if resolved_run is not None and (resolved_run / "preview_manifest.json").is_file()
        else None
    )
    payload = build_presentation_payload(
        events,
        presentation_summary,
        duration_s,
        run_dir=resolved_run,
        preview_manifest=preview_manifest,
    )
    return render_workspace_shell(payload)


def _workspace_video_path(run_dir: Path, annotated: Path) -> Path:
    """Keep the workspace clean while supporting pre-preview cached runs."""
    preview = run_dir / "workspace-preview.mp4"
    return preview if preview.is_file() and preview.stat().st_size > 0 else annotated


def _load_existing_run(run_dir: str | os.PathLike[str]) -> dict[str, object]:
    """Load a completed run for local QA without mutating analytical data."""

    root = Path(run_dir).expanduser().resolve()
    annotated = root / "annotated.mp4"
    events_path = root / "events.json"
    summary_path = root / "summary.json"
    log_path = root / "analysis.log"
    clips_dir = root / "clips"
    if not events_path.is_file() or not summary_path.is_file():
        raise ValueError("Готовый запуск не содержит annotated.mp4, events.json и summary.json")
    events_value = json.loads(events_path.read_text(encoding="utf-8"))
    summary_value = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(events_value, list) or not isinstance(summary_value, dict):
        raise TypeError("Контракт готового запуска повреждён")
    is_bundled_demo = _is_read_only_demo({"summary": summary_value}) and summary_value.get("metadata", {}).get("bundled_demo") is True
    if is_bundled_demo:
        annotated = root / "workspace-preview.mp4"
    if not annotated.is_file() or (is_bundled_demo and annotated.stat().st_size == 0):
        raise ValueError("В демо нет готового видео для просмотра" if is_bundled_demo else "Готовый запуск не содержит annotated.mp4, events.json и summary.json")
    events = [dict(event) for event in events_value if isinstance(event, dict)]
    metadata = summary_value.get("metadata")
    duration_s = (
        float(metadata.get("duration_s", 0) or 0)
        if isinstance(metadata, dict)
        else 0.0
    )
    if duration_s <= 0:
        duration_s = float(validate_video(annotated).duration_s)
    return {
        "events": events,
        "summary": summary_value,
        "duration_s": duration_s,
        "run_dir": str(root),
        "annotated_video": str(annotated),
        "workspace_video": str(_workspace_video_path(root, annotated)),
        "events_path": str(events_path),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "clips_dir": str(clips_dir),
        "render_stale": bool(metadata.get("export_stale", False)) if isinstance(metadata, dict) else False,
        "tracking_preview_stale": bool(metadata.get("tracking_preview_stale", metadata.get("export_stale", False))) if isinstance(metadata, dict) else False,
    }


def _pending_identity_review_count(summary: dict[str, object]) -> int:
    quality = summary.get("quality")
    return max(0, int(quality.get("required_review_count", 0) or 0)) if isinstance(quality, dict) else 0


_READ_ONLY_DEMO_MESSAGE = "Это предзагруженный демо-разбор только для просмотра. Чтобы анализировать или исправлять свой бой, нажмите «Новый анализ» и загрузите видео."


def _is_read_only_demo(payload: dict[str, object] | None) -> bool:
    state = payload or {}
    summary = state.get("summary")
    metadata = summary.get("metadata", {}) if isinstance(summary, dict) else {}
    return isinstance(metadata, dict) and metadata.get("demo_read_only") is True


def _require_mutable_result(payload: dict[str, object] | None) -> None:
    """Guard every write bridge, including calls bypassing hidden UI controls."""
    if _is_read_only_demo(payload):
        raise gr.Error(_READ_ONLY_DEMO_MESSAGE)
    state = payload or {}
    if state.get("run_dir"):
        summary_path = Path(str(state["run_dir"])) / "summary.json"
        if summary_path.is_file():
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            if _is_read_only_demo({"summary": saved}):
                raise gr.Error(_READ_ONLY_DEMO_MESSAGE)


def _result_mutation_controls(payload: dict[str, object] | None):
    mutable = not _is_read_only_demo(payload)
    return gr.update(visible=mutable), gr.update(visible=mutable, interactive=mutable), gr.update(visible=mutable)


def _available_export_files(payload: dict[str, object] | None):
    state = payload or {}
    summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    ready = bool(state.get("run_dir")) and not _is_read_only_demo(state) and not (
        _pending_identity_review_count(summary) or state.get("render_stale") or metadata.get("export_stale")
    )
    annotated = Path(str(state.get("annotated_video", "")))
    bundle = Path(str(state.get("run_dir", ""))) / "boxing-vision-result.zip"
    video_available = ready and annotated.is_file()
    bundle_available = ready and bundle.is_file()
    return (gr.update(value=str(annotated) if video_available else None, visible=video_available),
            gr.update(value=str(bundle) if bundle_available else None, visible=bundle_available))


def _event_rows(events: list[dict[str, object]], summary: dict[str, object]) -> list[list[object]]:
    fighters = summary.get("fighters") if isinstance(summary.get("fighters"), dict) else {}
    assert isinstance(fighters, dict)
    names = {
        fighter_id: str(data.get("name", fighter_id))
        for fighter_id, data in fighters.items()
        if isinstance(data, dict)
    }
    rows: list[list[object]] = []
    for event in events:
        outcome = str(event.get("outcome", "unclear"))
        evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
        context: list[str] = []
        if event.get("is_replay"):
            context.append("possible replay")
        if event.get("is_counter"):
            context.append("контратака")
        if event.get("combo_id"):
            context.append(str(event.get("combo_id")))
        if isinstance(evidence, dict) and evidence.get("possible_knockdown"):
            context.append("возможное падение")
        rows.append(
            [
                str(event.get("event_id", "—")),
                _format_time(int(event.get("peak_ms", 0) or 0)),
                (
                    f'{_format_time(int(event.get("start_ms", 0) or 0))}–'
                    f'{_format_time(int(event.get("end_ms", 0) or 0))}'
                ),
                int(event.get("round", 1) or 1),
                names.get(str(event.get("attacker_id")), str(event.get("attacker_id"))),
                "левая" if event.get("hand") == "left" else "правая",
                TECHNIQUE_LABELS.get(str(event.get("technique")), str(event.get("technique"))),
                TARGET_LABELS.get(str(event.get("target")), str(event.get("target"))),
                OUTCOME_LABELS.get(outcome, outcome),
                round(float(event.get("confidence", 0) or 0) * 100, 1),
                int(event.get("impact_proxy_0_100", 0) or 0),
                " · ".join(context) or "—",
                str(event.get("review_status", "unreviewed")),
            ]
        )
    return rows


def _filtered_events(
    filter_value: str,
    payload: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    state = payload or {}
    raw_events = state.get("events")
    raw_summary = state.get("summary")
    events = [event for event in raw_events if isinstance(event, dict)] if isinstance(raw_events, list) else []
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    if filter_value in {"likely_landed", "blocked", "missed", "unclear"}:
        events = [event for event in events if event.get("outcome") == filter_value]
    elif filter_value == "possible_replay":
        events = [event for event in events if bool(event.get("is_replay"))]
    elif filter_value == "possible_knockdown":
        events = [
            event
            for event in events
            if isinstance(event.get("evidence"), dict)
            and bool(event["evidence"].get("possible_knockdown"))
        ]
    return events, summary


def _filter_event_results(
    filter_value: str,
    payload: dict[str, object] | None,
) -> tuple[str, list[list[object]], dict[str, object]]:
    events, summary = _filtered_events(filter_value, payload)
    duration_s = float((payload or {}).get("duration_s", 0) or 0)
    return (
        _timeline(events, duration_s),
        _event_rows(events, summary),
        gr.update(choices=_seek_choices(events, summary), value=None),
    )


def _seek_choices(
    events: list[dict[str, object]],
    summary: dict[str, object],
) -> list[tuple[str, str]]:
    fighters = summary.get("fighters") if isinstance(summary.get("fighters"), dict) else {}
    names = {
        fighter_id: str(data.get("name", fighter_id))
        for fighter_id, data in fighters.items()
        if isinstance(data, dict)
    }
    choices: list[tuple[str, str]] = []
    for event in events:
        peak_ms = int(event.get("peak_ms", 0) or 0)
        outcome = str(event.get("outcome", "unclear"))
        label = (
            f'{_format_time(peak_ms)} · '
            f'{names.get(str(event.get("attacker_id")), str(event.get("attacker_id")))} · '
            f'{TECHNIQUE_LABELS.get(str(event.get("technique")), str(event.get("technique")))} · '
            f'{OUTCOME_LABELS.get(outcome, outcome)}'
        )
        choices.append((label, f'{event.get("event_id", "event")}|{peak_ms / 1000.0:.3f}'))
    return choices


def _bundle_paths(
    run_dir: Path,
    annotated: Path,
    events_path: Path,
    summary_path: Path,
    log_path: Path,
    clips_dir: Path,
) -> Path:
    bundle = run_dir / "boxing-vision-result.zip"
    temporary = bundle.with_name(f".{bundle.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in (annotated, events_path, summary_path, log_path):
                if path.is_file():
                    archive.write(path, path.name)
            for clip in sorted(clips_dir.glob("*.mp4")):
                archive.write(clip, f"clips/{clip.name}")
            preview_manifest = run_dir / "preview_manifest.json"
            if preview_manifest.is_file():
                archive.write(preview_manifest, preview_manifest.name)
            for preview in sorted((run_dir / "previews").glob("*.webp")):
                archive.write(preview, f"previews/{preview.name}")
            for profile in sorted((run_dir / "profiles").glob("*")):
                if profile.is_file():
                    archive.write(profile, f"profiles/{profile.name}")
            for filename in (
                "body_map_manifest.json",
                "model_manifest.json",
                "tracklets.jsonl",
                "tracking_diagnostics.jsonl",
                "scenes.json",
                "identity_profile.json",
                "review.json",
            ):
                artifact = run_dir / filename
                if artifact.is_file():
                    archive.write(artifact, artifact.name)
            for asset in sorted((run_dir / "assets").glob("*")):
                if asset.is_file():
                    archive.write(asset, f"assets/{asset.name}")
        os.replace(temporary, bundle)
    finally:
        temporary.unlink(missing_ok=True)
    return bundle


def _bundle_result(result: object) -> Path:
    return _bundle_paths(
        Path(result.run_dir),
        Path(result.annotated_video),
        Path(result.events_path),
        Path(result.summary_path),
        Path(result.log_path),
        Path(result.clips_dir),
    )


def _decorate_review_summary(
    events: list[dict[str, object]],
    previous: dict[str, object],
) -> dict[str, object]:
    previous_fighters = previous.get("fighters") if isinstance(previous.get("fighters"), dict) else {}
    fighter_names = {
        fighter_id: str(data.get("name", fighter_id))
        for fighter_id, data in previous_fighters.items()
        if isinstance(data, dict)
    }
    round_cards = previous.get("round_scores") if isinstance(previous.get("round_scores"), list) else []
    metadata = dict(previous.get("metadata")) if isinstance(previous.get("metadata"), dict) else {}
    scheduled_rounds = max(
        1,
        int(metadata.get("scheduled_rounds", len(round_cards) or 1) or 1),
    )
    raw_confirmed = metadata.get("confirmed_knockdowns_suffered")
    confirmed: dict[int, dict[str, int]] = {}
    if isinstance(raw_confirmed, dict):
        for round_number, counts in raw_confirmed.items():
            if isinstance(counts, dict):
                confirmed[int(round_number)] = {
                    str(fighter_id): int(count)
                    for fighter_id, count in counts.items()
                }
    punch_events = [PunchEvent(**event) for event in events]
    rebuilt = build_fight_summary(
        punch_events,
        fighter_names=fighter_names,
        scheduled_rounds=scheduled_rounds,
        confirmed_knockdowns_suffered=confirmed,
    )
    fighters = rebuilt.get("fighters")
    if isinstance(fighters, dict):
        for fighter_id, corner in (("fighter_a", "red"), ("fighter_b", "blue")):
            fighter = fighters.get(fighter_id)
            if isinstance(fighter, dict):
                previous_fighter = previous_fighters.get(fighter_id)
                fighter["id"] = fighter_id
                fighter["corner"] = corner
                if isinstance(previous_fighter, dict):
                    for profile_key in ("record", "portrait_filename"):
                        profile_value = previous_fighter.get(profile_key)
                        if profile_value:
                            fighter[profile_key] = profile_value
                fighter["stats"] = {
                    key: value
                    for key, value in fighter.items()
                    if key
                    not in {
                        "id",
                        "name",
                        "corner",
                        "record",
                        "portrait_filename",
                        "stats",
                    }
                }
    winner_id = rebuilt.get("winner_id")
    rebuilt["winner"] = {
        "fighter_id": winner_id,
        "name": str(rebuilt.get("winner_name") or "Недостаточно данных"),
        "confidence": float(rebuilt.get("confidence", 0) or 0),
        "label": "Прогноз модели" if winner_id else "Равный результат по модели",
    }
    quality = dict(previous.get("quality")) if isinstance(previous.get("quality"), dict) else {}
    eligible_confidence = [
        event.confidence
        for event in punch_events
        if not event.is_replay and event.review_status not in {"rejected", "deleted"}
    ]
    quality["event_confidence"] = (
        round(sum(eligible_confidence) / len(eligible_confidence), 4)
        if eligible_confidence
        else 0.0
    )
    quality["reviewed_events"] = sum(event.review_status != "unreviewed" for event in punch_events)
    rebuilt["quality"] = quality
    rebuilt["metadata"] = metadata
    from .quality import apply_result_gate
    return apply_result_gate(rebuilt)


def _review_event(
    seek_value: str | None,
    action: str,
    filter_value: str,
    payload: dict[str, object] | None,
) -> tuple[
    str,
    str,
    str,
    list[list[object]],
    dict[str, object],
    dict[str, object],
    str,
    str,
    str,
]:
    _require_mutable_result(payload)
    if not seek_value:
        raise gr.Error("Сначала выберите событие в списке перехода")
    state = dict(payload or {})
    raw_events = state.get("events")
    previous_summary = state.get("summary")
    if not isinstance(raw_events, list) or not isinstance(previous_summary, dict):
        raise gr.Error("Сначала завершите анализ")
    event_id = str(seek_value).split("|", 1)[0]
    events = [dict(event) for event in raw_events if isinstance(event, dict)]
    selected = next((event for event in events if event.get("event_id") == event_id), None)
    if selected is None:
        raise gr.Error("Событие не найдено")
    if action == "rejected":
        selected["review_status"] = "rejected"
    elif action == "real_not_replay":
        selected["is_replay"] = False
        selected["review_status"] = "confirmed"
    else:
        selected["review_status"] = "confirmed"

    summary = _decorate_review_summary(events, previous_summary)
    events_path = Path(str(state.get("events_path", "")))
    summary_path = Path(str(state.get("summary_path", "")))
    if not events_path.is_file() or not summary_path.is_file():
        raise gr.Error("Файлы текущего задания недоступны")
    atomic_write_json(events_path, events)
    atomic_write_json(summary_path, summary)
    bundle = _bundle_paths(
        Path(str(state["run_dir"])),
        Path(str(state["annotated_video"])),
        events_path,
        summary_path,
        Path(str(state["log_path"])),
        Path(str(state["clips_dir"])),
    )
    state.update({"events": events, "summary": summary})
    filtered, _ = _filtered_events(filter_value, state)
    status = (
        f'<div class="status-done">Review сохранён для {escape(event_id)}. '
        "Статистика и прогноз модели пересчитаны. Разметка внутри MP4 остаётся "
        "исходным автоматическим слоем.</div>"
    )
    duration_s = float(state.get("duration_s", 0) or 0)
    return (
        status,
        _overview(summary),
        _timeline(filtered, duration_s),
        _event_rows(filtered, summary),
        gr.update(choices=_seek_choices(filtered, summary), value=None),
        state,
        str(bundle),
        str(events_path),
        str(summary_path),
    )


_REVIEW_ACTIONS = {"confirmed", "rejected", "real_not_replay"}

_IDENTITY_REVIEW_CHOICES = [
    ("Красный A", "FIGHTER_A"), ("Синий B", "FIGHTER_B"),
    ("Рефери / другой человек", "OTHER"), ("Личность не определена", "UNKNOWN"),
]
_SCENE_REVIEW_CHOICES = [
    ("Активный бой", "ACTIVE_FIGHT"), ("Перерыв", "BREAK"), ("Повтор", "REPLAY"),
    ("Не бой", "NON_FIGHT"), ("Не удалось определить", "UNCERTAIN"),
]


def _identity_review_items(payload: dict[str, object] | None) -> list[dict[str, object]]:
    if not payload or not payload.get("run_dir"):
        return []
    source = Path(str(payload["run_dir"])) / "review.json"
    if not source.is_file():
        return []
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        raise TypeError("Файл проверки эпизодов повреждён")
    return [dict(item) for item in raw["items"] if isinstance(item, dict)
            and item.get("kind") in {"scene", "tracklet"}
            and (item.get("review_id") or item.get("tracklet_id"))]


def _identity_review_key(item: dict[str, object]) -> str:
    return str(item.get("review_id") or item.get("tracklet_id"))


def _refresh_identity_review(payload: dict[str, object] | None):
    if _is_read_only_demo(payload):
        return gr.update(choices=[], value=None), None, _enrollment_note(_READ_ONLY_DEMO_MESSAGE), gr.update(choices=[], value=None)
    items = _identity_review_items(payload)
    choices = [(f'{_format_time(int(item.get("start_ms", 0)))} · {"Сцена" if item["kind"] == "scene" else "Человек"} · {_identity_review_key(item)}',
                _identity_review_key(item)) for item in items]
    note = f"Эпизодов для проверки: {len(items)}. Выберите эпизод и подтвердите личность или тип сцены." if items else "Спорных сцен и последовательностей для проверки нет."
    return gr.update(choices=choices, value=None), None, _enrollment_note(note), gr.update(choices=[], value=None)


def _show_identity_review(item_id: str | None, payload: dict[str, object] | None):
    _require_mutable_result(payload)
    item = next((entry for entry in _identity_review_items(payload) if _identity_review_key(entry) == item_id), None)
    if item is None:
        return None, _enrollment_note("Выберите эпизод."), gr.update(choices=[], value=None)
    run_dir = Path(str(payload["run_dir"]))
    time_ms = int(item.get("start_ms", 0))
    end_ms = int(item.get("end_ms", time_ms))
    time_ms = (time_ms + end_ms) // 2
    cached_pose = None
    cached = run_dir / ".render_cache" / "detections.jsonl.gz"
    if item["kind"] == "tracklet" and cached.is_file():
        nearest = float("inf")
        with gzip.open(cached, "rt", encoding="utf-8") as handle:
            for line in handle:
                frame = json.loads(line)
                stamp = int(frame.get("timestamp_ms", 0))
                if stamp > end_ms:
                    break
                if frame.get("shot_id") != item.get("shot_id"):
                    continue
                for pose in frame.get("poses", []):
                    if str(pose.get("source_track_id")) == str(item.get("source_track_id")) and abs(stamp - time_ms) < nearest:
                        nearest = abs(stamp - time_ms)
                        cached_pose = (stamp, pose)
        if cached_pose:
            time_ms = cached_pose[0]
    capture = cv2.VideoCapture(str(run_dir / ".render_cache" / "normalized.mp4"))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ok, frame = capture.read()
    finally:
        capture.release()
    image = None
    if ok:
        if cached_pose:
            box = cached_pose[1].get("bbox", {})
            if isinstance(box, dict) and all(key in box for key in ("x1", "y1", "x2", "y2")):
                cv2.rectangle(frame, (int(box["x1"]), int(box["y1"])), (int(box["x2"]), int(box["y2"])), (93, 193, 244), 3)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    choices = _SCENE_REVIEW_CHOICES if item["kind"] == "scene" else _IDENTITY_REVIEW_CHOICES
    proposed = str(item.get("scene_state") if item["kind"] == "scene" else item.get("identity_state"))
    labels = {value: label for label, value in choices}
    confidence = item.get("identity_confidence")
    note = f"{_format_time(int(item.get('start_ms', 0)))}–{_format_time(end_ms)} · Предложение: {labels.get(proposed, 'не определено')}"
    if confidence is not None:
        note += f" · уверенность {float(confidence):.0%}"
    if not ok:
        note += ". Исходный кадр недоступен; проверьте наличие кэша видео."
    return image, _enrollment_note(note), gr.update(choices=choices, value=proposed if proposed in labels else None)


def _apply_identity_review(item_id: str | None, action: str | None, payload: dict[str, object] | None,
                           progress: gr.Progress = _GRADIO_PROGRESS):
    _require_mutable_result(payload)
    state = dict(payload or {})
    item = next((entry for entry in _identity_review_items(state) if _identity_review_key(entry) == item_id), None)
    if item is None:
        raise gr.Error("Эпизод отсутствует в текущем списке проверки")
    choices = _SCENE_REVIEW_CHOICES if item["kind"] == "scene" else _IDENTITY_REVIEW_CHOICES
    if action not in {value for _, value in choices}:
        raise gr.Error("Выберите допустимый результат проверки")
    from .pipeline import redecode_from_cache

    identity_overrides = dict(state.get("identity_overrides") or {})
    scene_overrides = dict(state.get("scene_overrides") or {})
    progress(.05, desc="Пересчитываем личности и события по сохранённым данным")
    run_dir = Path(str(state["run_dir"]))
    if item["kind"] == "scene":
        scene_overrides[str(item["shot_id"])] = action
        redecode_from_cache(run_dir, identity_overrides=identity_overrides, scene_overrides=scene_overrides)
    else:
        from .cache_review import correct_identity_at

        start_ms, end_ms = int(item.get("start_ms", 0)), int(item.get("end_ms", 0))
        time_ms = int(item.get("representative_timestamp_ms", (start_ms + end_ms) // 2))
        try:
            correct_identity_at(run_dir, timestamp_ms=time_ms, source_track_id=item.get("source_track_id"),
                                identity_state=action, segment_id=item.get("segment_id"))
        except (OSError, ValueError, TypeError) as exc:
            raise gr.Error(f"Не удалось исправить сегмент: {exc}. Можно выбрать точный кадр выше.") from exc
    refreshed = _load_existing_run(run_dir)
    state.update(refreshed)
    state.update({"identity_overrides": identity_overrides, "scene_overrides": scene_overrides, "render_stale": True})
    events = state["events"]
    summary = state["summary"]
    progress(1, desc="Результат проверки сохранён")
    state["tracking_preview_stale"] = True
    return (_enrollment_note("Проверка сохранена. Статистика пересчитана; в видео пока прежние рамки. Нажмите «Обновить трекинг в плеере» ниже.", True),
            _workspace(events, summary, float(state["duration_s"]), run_dir), _overview(summary),
            _event_rows(events, summary), state, None, str(state["events_path"]), str(state["summary_path"]))


def _correction_bbox(candidate: dict[str, object]) -> tuple[float, float, float, float]:
    raw = candidate.get("bbox", candidate.get("detector_bbox"))
    values = [raw.get(key) for key in ("x1", "y1", "x2", "y2")] if isinstance(raw, dict) else raw
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("В кэше отсутствует настоящая рамка человека")
    box = tuple(float(value) for value in values)
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Рамка в кэше повреждена")
    return box


def _load_identity_correction_frame(payload: dict[str, object] | None, time_s: float):
    """Read one normalized frame and cached detector boxes; never rerun ML."""
    _require_mutable_result(payload)
    from .cache_review import get_identity_correction_frame

    try:
        state = payload or {}
        if not state.get("run_dir"):
            raise ValueError("Сначала откройте результат анализа")
        if not np.isfinite(float(time_s)) or float(time_s) < 0:
            raise ValueError("Выберите время внутри видео")
        run_dir = Path(str(state["run_dir"])).resolve()
        context = dict(get_identity_correction_frame(run_dir, round(float(time_s) * 1000)))
        context["run_dir"] = str(run_dir)
        timestamp_ms = int(context["timestamp_ms"])
        capture = cv2.VideoCapture(str(run_dir / ".render_cache" / "normalized.mp4"))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise ValueError("Исходный кадр недоступен. Для исправления нужен сохранённый кэш видео")
        height, width = frame.shape[:2]
        if (int(context.get("width", width)), int(context.get("height", height))) != (width, height):
            raise ValueError("Размер кадра не совпадает с кэшем детектора")
        context.update({"width": width, "height": height})
        for candidate in context.get("candidates", []):
            x1, y1, x2, y2 = _correction_bbox(candidate)
            role = str(candidate.get("identity_state", "UNKNOWN"))
            color = (74, 81, 255) if role == "FIGHTER_A" else (255, 130, 86) if role == "FIGHTER_B" else (170, 170, 170)
            cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        note = (f"{_format_time(timestamp_ms)} · Выберите A или B и нажмите внутри его рамки. "
                "Исправление применяется только к непрерывному сегменту около этого момента, не ко всему ID.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), context, _enrollment_note(note)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc


def _select_identity_correction(frame_state: dict[str, object] | None, action: str,
                                payload: dict[str, object] | None, event: gr.SelectData,
                                progress: gr.Progress = _GRADIO_PROGRESS):
    """Click targets a real cached detection; the server revalidates its segment."""
    _require_mutable_result(payload)
    from .cache_review import correct_identity_at

    try:
        state = dict(payload or {})
        context = frame_state or {}
        if not state.get("run_dir") or not context.get("run_dir"):
            raise ValueError("Сначала покажите кадр для исправления")
        run_dir = Path(str(state["run_dir"])).resolve()
        if Path(str(context["run_dir"])).resolve() != run_dir:
            raise ValueError("Результат изменился. Снова покажите нужный кадр")
        if action not in {"FIGHTER_A", "FIGHTER_B"}:
            raise ValueError("Выберите красного A или синего B")
        if not isinstance(event.index, (tuple, list)) or len(event.index) != 2:
            raise ValueError("Нажмите внутри рамки бойца")
        x, y = (float(value) for value in event.index)
        if not np.isfinite((x, y)).all() or not (0 <= x < int(context["width"]) and 0 <= y < int(context["height"])):
            raise ValueError("Нажмите внутри кадра")
        candidates = []
        for candidate in context.get("candidates", []):
            x1, y1, x2, y2 = _correction_bbox(candidate)
            if x1 <= x <= x2 and y1 <= y <= y2:
                candidates.append(candidate)
        if not candidates:
            raise ValueError("Здесь нет обнаруженного человека. Нажмите внутри настоящей рамки")
        if len(candidates) != 1:
            raise ValueError("Рамки перекрываются. Нажмите на видимую часть нужного бойца вне пересечения или выберите другой кадр")
        selected = candidates[0]
        if selected.get("source_track_id") is None:
            raise ValueError("У выбранной рамки нет сохранённой траектории. Выберите соседний кадр")
        progress(.05, desc="Исправляем личность в выбранном сегменте")
        correct_identity_at(run_dir, timestamp_ms=int(context["timestamp_ms"]),
                            source_track_id=selected["source_track_id"], identity_state=action,
                            segment_id=selected.get("segment_id"))
        state.update(_load_existing_run(run_dir))
        state["render_stale"] = True
        state["tracking_preview_stale"] = True
        events, summary = state["events"], state["summary"]
        progress(1, desc="Исправление сохранено")
        return (_enrollment_note("Личность исправлена в выбранном сегменте. Статистика обновлена; в видео пока прежние рамки. Нажмите «Обновить трекинг в плеере» ниже.", True),
                _workspace(events, summary, float(state["duration_s"]), run_dir), _overview(summary),
                _event_rows(events, summary), state, None, str(state["events_path"]), str(state["summary_path"]))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc


def _review_workspace_event(
    command_value: str | None,
    payload: dict[str, object] | None,
) -> tuple[
    str,
    str,
    str,
    list[list[object]],
    dict[str, object],
    None,
    str,
    str,
]:
    """Validate and persist a review command emitted by the JS inspector."""

    _require_mutable_result(payload)
    try:
        command = json.loads(command_value or "{}")
    except (TypeError, ValueError) as exc:
        raise gr.Error("Некорректная команда ручной проверки") from exc
    if not isinstance(command, dict):
        raise gr.Error("Некорректная команда ручной проверки")
    event_id = str(command.get("event_id") or "").strip()
    action = str(command.get("action") or "").strip()
    if not event_id or len(event_id) > 128 or action not in _REVIEW_ACTIONS:
        raise gr.Error("Событие или действие review не разрешено")

    state = dict(payload or {})
    raw_events = state.get("events")
    previous_summary = state.get("summary")
    if not isinstance(raw_events, list) or not isinstance(previous_summary, dict):
        raise gr.Error("Сначала завершите анализ")
    events = [dict(event) for event in raw_events if isinstance(event, dict)]
    selected = next((event for event in events if str(event.get("event_id")) == event_id), None)
    if selected is None:
        raise gr.Error("Событие не найдено в текущем задании")
    if action == "rejected":
        selected["review_status"] = "rejected"
    elif action == "real_not_replay":
        selected["is_replay"] = False
        selected["review_status"] = "confirmed"
    else:
        selected["review_status"] = "confirmed"

    events_path = Path(str(state.get("events_path", "")))
    summary_path = Path(str(state.get("summary_path", "")))
    run_dir = Path(str(state.get("run_dir", "")))
    if (
        not events_path.is_file()
        or not summary_path.is_file()
        or not run_dir.is_dir()
        or events_path.parent.resolve() != run_dir.resolve()
        or summary_path.parent.resolve() != run_dir.resolve()
    ):
        raise gr.Error("Файлы текущего задания недоступны")

    summary = _decorate_review_summary(events, previous_summary)
    summary_metadata = summary.get("metadata")
    if isinstance(summary_metadata, dict):
        summary_metadata["export_stale"] = True
    atomic_write_json(events_path, events)
    atomic_write_json(summary_path, summary)
    duration_s = float(state.get("duration_s", 0) or 0)
    state.update(
        {
            "events": events,
            "summary": summary,
            "render_stale": True,
        }
    )
    status = (
        f'<div class="status-ready">Review сохранён для {escape(event_id)}. '
        "JSON и статистика пересчитаны; MP4 и полный ZIP помечены устаревшими. "
        "Нажмите «Пересобрать MP4» для render-only обновления.</div>"
    )
    return (
        status,
        _workspace(events, summary, duration_s, run_dir),
        _overview(summary),
        _event_rows(events, summary),
        state,
        None,
        str(events_path),
        str(summary_path),
    )


def _tracking_preview_status(payload: dict[str, object] | None):
    state = payload or {}
    if _is_read_only_demo(state):
        return _enrollment_note(_READ_ONLY_DEMO_MESSAGE), gr.update(interactive=False)
    if not state.get("run_dir"):
        return _enrollment_note("Сначала откройте результат анализа."), gr.update(interactive=False)
    summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    stale = state.get("tracking_preview_stale", metadata.get("tracking_preview_stale", metadata.get("export_stale", False)))
    note = (
        "В плеере пока прежние рамки. После всех исправлений обновите трекинг одной кнопкой. "
        "Это возможно и при оставшихся эпизодах на проверку; финальный экспорт остаётся отдельно."
        if stale else
        "Трекинг в плеере актуален. Эта кнопка обновляет только предпросмотр по сохранённым данным; финальный MP4 и ZIP не создаются."
    )
    return _enrollment_note(note, not stale), gr.update(interactive=True)


def _refresh_tracking_player(payload: dict[str, object] | None,
                             progress: gr.Progress = _GRADIO_PROGRESS):
    """Explicit preview-only render; unresolved review never blocks this path."""
    _require_mutable_result(payload)
    from .pipeline import rebuild_tracking_preview_from_cache

    state = dict(payload or {})
    if not state.get("run_dir"):
        raise gr.Error("Сначала откройте результат анализа")
    run_dir = Path(str(state["run_dir"])).expanduser().resolve()
    if not run_dir.is_dir():
        raise gr.Error("Каталог результата недоступен")
    _CANCEL_EVENT.clear()
    try:
        progress(.02, desc="Обновляем только трекинг в плеере, без повторного ML-анализа")
        preview = rebuild_tracking_preview_from_cache(
            run_dir, lambda value, description: progress(value, desc=description),
            cancel_callback=_CANCEL_EVENT.is_set,
        )
        state.update(_load_existing_run(run_dir))
    except Exception as exc:
        # No final-render or export fallback: an error must leave the old player
        # and stale export state intact, with the saved correction recoverable.
        raise gr.Error(f"Не удалось обновить плеер: {exc}. Исправления сохранены; прежнее видео и статус экспорта не изменены.") from exc
    state["workspace_video"] = str(preview)
    state["tracking_preview_stale"] = False
    state["render_stale"] = True
    events, summary = state["events"], state["summary"]
    progress(1, desc="Трекинг в плеере обновлён")
    # Gradio hashes the returned file contents into its served cache path, so
    # changed frames receive a new video URL even though the run filename stays.
    return (
        _enrollment_note("Трекинг в плеере обновлён по сохранённым данным. Проверка может продолжаться; финальный MP4 и ZIP пока не пересобраны.", True),
        str(preview), _workspace(events, summary, float(state["duration_s"]), run_dir),
        _overview(summary), _event_rows(events, summary), state, None, str(state["summary_path"]),
    )


def _rebuild_reviewed_video(
    payload: dict[str, object] | None,
    progress: gr.Progress = _GRADIO_PROGRESS,
) -> tuple[
    str,
    str,
    str,
    str,
    list[list[object]],
    dict[str, object],
    str,
    str,
]:
    _require_mutable_result(payload)
    state = dict(payload or {})
    run_dir = Path(str(state.get("run_dir", ""))).expanduser().resolve()
    if not run_dir.is_dir():
        raise gr.Error("Сначала завершите анализ")
    summary_file = run_dir / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
    if _pending_identity_review_count(summary):
        raise gr.Error("Сначала подтвердите обязательные эпизоды во вкладке «Проверка личностей и сцен». Видео доступно как предпросмотр; финальный экспорт пока закрыт.")
    try:
        annotated = rebuild_from_cache(
            run_dir,
            lambda value, description: progress(value, desc=description),
            cancel_callback=_CANCEL_EVENT.is_set,
        )
    except RenderCacheUnavailableError as exc:
        raise gr.Error(str(exc)) from exc
    except AnalysisCancelledError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Не удалось пересобрать MP4: {exc}") from exc

    events_path = run_dir / "events.json"
    summary_path = run_dir / "summary.json"
    events_value = json.loads(events_path.read_text(encoding="utf-8"))
    summary_value = json.loads(summary_path.read_text(encoding="utf-8"))
    events = [dict(event) for event in events_value if isinstance(event, dict)]
    if not isinstance(summary_value, dict):
        raise gr.Error("Пересобранная сводка повреждена")
    duration_s = float(state.get("duration_s", 0) or 0)
    state.update(
        {
            "events": events,
            "summary": summary_value,
            "annotated_video": str(annotated),
            "workspace_video": str(_workspace_video_path(run_dir, annotated)),
            "render_stale": False,
            "tracking_preview_stale": False,
        }
    )
    bundle = _bundle_paths(
        run_dir,
        annotated,
        events_path,
        summary_path,
        run_dir / "analysis.log",
        run_dir / "clips",
    )
    return (
        '<div class="status-done">MP4 пересобран по сохранённым наблюдениям без повторного ML-анализа. Экспорт актуален.</div>',
        str(_workspace_video_path(run_dir, annotated)),
        _workspace(events, summary_value, duration_s, run_dir),
        _overview(summary_value),
        _event_rows(events, summary_value),
        state,
        str(bundle),
        str(summary_path),
    )


def _run_analysis(
    video: object,
    anchor_state: dict[str, object] | None,
    fighter_a: str,
    fighter_b: str,
    fighter_a_record: str,
    fighter_b_record: str,
    fighter_a_portrait: object,
    fighter_b_portrait: object,
    stance_a: str,
    stance_b: str,
    hud_mode: str,
    rounds: float,
    round_length: float,
    rest_length: float,
    start_s: float,
    end_s: float,
    knockdowns_a: str,
    knockdowns_b: str,
    timing_mode: str = "scheduled",
    progress: gr.Progress = _GRADIO_PROGRESS,
):
    _CANCEL_EVENT.clear()
    try:
        source = _video_path(video)
        enrollment_fields = _enrollment_config_fields(anchor_state, source, float(start_s or 0))
        scheduled_rounds = int(rounds)
        config = AnalysisConfig(
            fighter_a_name=(fighter_a or "Красный угол").strip(),
            fighter_b_name=(fighter_b or "Синий угол").strip(),
            fighter_a_record=(fighter_a_record or "").strip() or None,
            fighter_b_record=(fighter_b_record or "").strip() or None,
            fighter_a_portrait_path=_optional_file_path(fighter_a_portrait),
            fighter_b_portrait_path=_optional_file_path(fighter_b_portrait),
            fighter_a_stance=stance_a,
            fighter_b_stance=stance_b,
            hud_mode=hud_mode,
            **enrollment_fields,
            timing_mode=timing_mode,
            scheduled_rounds=scheduled_rounds,
            round_length_s=int(round_length),
            rest_length_s=int(rest_length),
            fight_start_s=float(start_s or 0),
            fight_end_s=float(end_s) if end_s and float(end_s) > 0 else None,
            confirmed_knockdowns_a_rounds=_parse_round_list(knockdowns_a, scheduled_rounds),
            confirmed_knockdowns_b_rounds=_parse_round_list(knockdowns_b, scheduled_rounds),
        )
        result = analyze_video(
            source,
            config,
            lambda value, description: progress(value, desc=description),
            cancel_callback=_CANCEL_EVENT.is_set,
        )
        events = [event.to_dict() for event in result.events]
        metadata = result.summary.get("metadata")
        duration_s = float(metadata.get("duration_s", 0) or 0) if isinstance(metadata, dict) else 0.0
        clips = [str(path) for path in sorted(result.clips_dir.glob("*.mp4"))]
        pending_review = _pending_identity_review_count(result.summary)
        bundle = _bundle_result(result) if not pending_review else None
        status = (
            f'<div class="status-done">Анализ завершён. Найдено кандидатов событий: '
            f'{len(events)}. Для каждого результата показана уверенность модели.</div>'
        )
        if pending_review:
            status = _enrollment_note(f"Предпросмотр готов. Подтвердите эпизоды во вкладке «Проверка личностей и сцен»: {pending_review}. После проверки будет доступен финальный экспорт.")
        return (
            status,
            str(_workspace_video_path(result.run_dir, result.annotated_video)),
            _workspace(events, result.summary, duration_s, result.run_dir),
            _overview(result.summary),
            _event_rows(events, result.summary),
            str(bundle) if bundle is not None else None,
            str(result.events_path),
            str(result.summary_path),
            clips,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            {
                "events": events,
                "summary": result.summary,
                "duration_s": duration_s,
                "run_dir": str(result.run_dir),
                "annotated_video": str(result.annotated_video),
                "workspace_video": str(_workspace_video_path(result.run_dir, result.annotated_video)),
                "events_path": str(result.events_path),
                "summary_path": str(result.summary_path),
                "log_path": str(result.log_path),
                "clips_dir": str(result.clips_dir),
                "render_stale": False,
            },
        )
    except AnalysisCancelledError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Анализ остановлен: {exc}") from exc
    finally:
        _CANCEL_EVENT.clear()


def _build_legacy_app() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue="red",
        secondary_hue="blue",
        neutral_hue="slate",
        font=["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
    )
    with gr.Blocks(
        theme=theme,
        css=CSS,
        title="Boxing Vision · локальный AI-анализ",
        analytics_enabled=False,
    ) as demo:
        anchor_state = gr.State(_empty_anchor_state())
        result_state = gr.State({"events": [], "summary": {}})
        gr.HTML(
            """
            <div id="hero">
              <div class="hero-kicker">Local computer vision · investor prototype</div>
              <div class="hero-title"><span>BOXING</span> <span>VISION</span></div>
              <div class="hero-sub">Загрузите трансляцию боя. Система локально отследит двух боксёров, выделит кандидаты ударов и соберёт размеченный ролик, статистику, таймкоды и прогноз модели.</div>
              <div class="hero-badge">Данные не покидают этот Mac</div>
            </div>
            """
        )
        gr.HTML(
            "Результаты показывают оценку модели с уровнем уверенности. Impact proxy — относительный индекс, а не физическая сила.",
            elem_id="research-note",
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=7, elem_id="upload-card"):
                gr.HTML('<div class="section-label">01 · исходная трансляция</div>')
                input_video = gr.Video(
                    label="MP4 / MOV / MKV · до 60 минут",
                    sources=["upload"],
                    format=None,
                    height=440,
                    include_audio=True,
                    show_download_button=False,
                )
                preview_image = gr.Image(
                    label="Кадр подтверждения · кликните A, затем B",
                    type="numpy",
                    interactive=False,
                    height=330,
                )
                anchor_note = gr.HTML(
                    '<div class="confirmation-note">После загрузки укажите на кадре сначала красного, затем синего бойца.</div>'
                )
                reset_anchors = gr.Button("Сбросить выбор бойцов", size="sm")
                video_meta = gr.HTML(
                    '<div class="status-ready">Загрузите локальный видеофайл. Он не покинет этот Mac.</div>'
                )
            with gr.Column(scale=4, elem_id="settings-card"):
                gr.HTML('<div class="section-label">02 · параметры боя</div>')
                fighter_a = gr.Textbox(label="Красный угол", value="Боксёр A")
                fighter_b = gr.Textbox(label="Синий угол", value="Боксёр B")
                with gr.Row():
                    stance_a = gr.Dropdown(
                        label="Стойка A",
                        choices=[("Неизвестна", "unknown"), ("Правша", "orthodox"), ("Левша", "southpaw")],
                        value="unknown",
                    )
                    stance_b = gr.Dropdown(
                        label="Стойка B",
                        choices=[("Неизвестна", "unknown"), ("Правша", "orthodox"), ("Левша", "southpaw")],
                        value="unknown",
                    )
                with gr.Row():
                    rounds = gr.Number(label="Раундов", value=12, precision=0, minimum=1, maximum=24)
                    round_length = gr.Number(
                        label="Секунд в раунде", value=180, precision=0, minimum=60, maximum=300
                    )
                    rest_length = gr.Number(
                        label="Перерыв, сек", value=60, precision=0, minimum=0, maximum=180
                    )
                with gr.Row():
                    start_s = gr.Number(label="Начало боя, сек", value=0, minimum=0)
                    end_s = gr.Number(label="Конец, сек (0 = весь файл)", value=0, minimum=0)
                with gr.Accordion("Подтверждённые нокдауны · необязательно", open=False):
                    knockdowns_a = gr.Textbox(
                        label="Раунды, где A был в нокдауне",
                        placeholder="Например: 2, 5",
                    )
                    knockdowns_b = gr.Textbox(
                        label="Раунды, где B был в нокдауне",
                        placeholder="Повторите номер для двух нокдаунов: 3, 3",
                    )
                    gr.Markdown(
                        "Только ручное подтверждение влияет на 10–8. Автоматические возможные падения остаются на проверке."
                    )
                with gr.Row():
                    analyze_button = gr.Button(
                        "Запустить анализ",
                        variant="primary",
                        elem_id="analyze-button",
                        scale=4,
                    )
                    cancel_button = gr.Button("Отменить", variant="stop", scale=1)
                gr.Markdown(
                    "Первый запуск загружает открытые веса RTMPose. Полный бой обрабатывается дольше своей длительности; прогресс будет показан сверху."
                )

        job_status = gr.HTML(elem_id="job-status")

        with gr.Group(visible=False) as result_group:
            gr.HTML(
                '<div class="result-header"><h2>Разбор боя</h2><span>готовый аналитический пакет</span></div>'
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=7, elem_id="result-video-card"):
                    output_video = gr.Video(
                        label="Размеченная трансляция",
                        height=520,
                        autoplay=False,
                        show_download_button=True,
                        elem_id="annotated-video",
                    )
                with gr.Column(scale=4, elem_id="stats-card"):
                    overview_html = gr.HTML()

            gr.HTML('<div class="result-header"><h2>Таймлайн</h2><span>цвет = предполагаемый исход</span></div>')
            timeline_html = gr.HTML(elem_id="timeline-card")
            with gr.Row(elem_classes="timeline-controls"):
                event_filter = gr.Dropdown(
                    label="Фильтр событий",
                    choices=[
                        ("Все кандидаты", "all"),
                        ("Вероятные попадания", "likely_landed"),
                        ("Блоки", "blocked"),
                        ("Промахи", "missed"),
                        ("Неясные", "unclear"),
                        ("Possible replay", "possible_replay"),
                        ("Возможные падения", "possible_knockdown"),
                    ],
                    value="all",
                )
                seek_event = gr.Dropdown(
                    label="Перейти к событию в видео",
                    choices=[],
                    value=None,
                    interactive=True,
                )
            with gr.Row():
                review_action = gr.Radio(
                    label="Ручная проверка выбранного события",
                    choices=[
                        ("Подтвердить кандидат", "confirmed"),
                        ("Отклонить", "rejected"),
                        ("Это реальный эпизод, не повтор", "real_not_replay"),
                    ],
                    value="confirmed",
                    scale=4,
                )
                review_button = gr.Button("СОХРАНИТЬ REVIEW И ПЕРЕСЧИТАТЬ", scale=2)

            gr.HTML('<div class="result-header"><h2>События</h2><span>поиск и фильтр внутри таблицы</span></div>')
            event_table = gr.Dataframe(
                headers=[
                    "Event ID",
                    "Таймкод",
                    "Интервал",
                    "Раунд",
                    "Атакующий",
                    "Рука",
                    "Тип",
                    "Цель",
                    "Результат",
                    "Confidence, %",
                    "Impact",
                    "Контекст",
                    "Review status",
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "number",
                    "str",
                    "str",
                    "str",
                    "str",
                    "str",
                    "number",
                    "number",
                    "str",
                    "str",
                ],
                row_count=(0, "dynamic"),
                col_count=(13, "fixed"),
                interactive=False,
                show_search="filter",
                show_row_numbers=True,
                max_height=520,
                wrap=True,
                elem_id="events-table",
            )

            gr.HTML('<div class="result-header"><h2>Скачать</h2><span>видео, данные и клипы</span></div>')
            with gr.Column(elem_id="downloads-card"):
                bundle_file = gr.File(label="Полный пакет ZIP")
                with gr.Row():
                    events_file = gr.File(label="events.json")
                    summary_file = gr.File(label="summary.json")
                clips_file = gr.File(label="Ключевые эпизоды", file_count="multiple")
                gr.HTML(
                    '<div class="legal-foot">Для инвесторского показа используйте видео, на которое у вас есть права. Физическая сила удара и официальный вердикт из монокулярной трансляции не заявляются.</div>'
                )

        input_video.change(
            _prepare_confirmation,
            inputs=[input_video, start_s],
            outputs=[video_meta, preview_image, anchor_state, anchor_note],
            queue=False,
        )
        start_s.change(
            _prepare_confirmation,
            inputs=[input_video, start_s],
            outputs=[video_meta, preview_image, anchor_state, anchor_note],
            queue=False,
        )
        reset_anchors.click(
            _prepare_confirmation,
            inputs=[input_video, start_s],
            outputs=[video_meta, preview_image, anchor_state, anchor_note],
            queue=False,
        )
        preview_image.select(
            _select_fighter_anchor,
            inputs=[preview_image, anchor_state],
            outputs=[preview_image, anchor_state, anchor_note],
            queue=False,
        )
        analyze_button.click(
            _run_analysis,
            inputs=[
                input_video,
                anchor_state,
                fighter_a,
                fighter_b,
                stance_a,
                stance_b,
                rounds,
                round_length,
                rest_length,
                start_s,
                end_s,
                knockdowns_a,
                knockdowns_b,
            ],
            outputs=[
                job_status,
                output_video,
                overview_html,
                timeline_html,
                event_table,
                bundle_file,
                events_file,
                summary_file,
                clips_file,
                result_group,
                seek_event,
                result_state,
            ],
            concurrency_limit=1,
            concurrency_id="boxing-analysis",
        )
        cancel_button.click(
            _request_cancel,
            outputs=job_status,
            queue=False,
        )
        event_filter.change(
            _filter_event_results,
            inputs=[event_filter, result_state],
            outputs=[timeline_html, event_table, seek_event],
            queue=False,
        )
        review_button.click(
            _review_event,
            inputs=[seek_event, review_action, event_filter, result_state],
            outputs=[
                job_status,
                overview_html,
                timeline_html,
                event_table,
                seek_event,
                result_state,
                bundle_file,
                events_file,
                summary_file,
            ],
            concurrency_limit=1,
            concurrency_id="boxing-analysis",
        )
        seek_event.change(
            None,
            inputs=seek_event,
            queue=False,
            js="""
            (value) => {
              if (!value) return [];
              const seconds = Number(String(value).split('|').pop());
              const video = document.querySelector('#annotated-video video');
              if (video && Number.isFinite(seconds)) {
                video.currentTime = Math.max(0, seconds - 0.35);
                video.scrollIntoView({behavior: 'smooth', block: 'center'});
                video.play().catch(() => {});
              }
              return [];
            }
            """,
        )
    return demo


def _desktop_chrome_css() -> str:
    """Hide framework attribution/navigation only in the native window shell.

    This is a presentation preference, not an API or security restriction.
    The ordinary browser application keeps Gradio's footer unchanged.
    """
    if os.environ.get("BOXING_VISION_DESKTOP", "").strip() != "1":
        return ""
    return "\n/* Native desktop shell: no browser framework footer. */\n.gradio-container footer { display: none !important; }\n"


def build_app() -> gr.Blocks:
    """Build the Gradio control surface and the JS-driven result workspace."""

    font_paths, font_css = installed_display_font_assets()
    # Serve only the chosen installed files, never the whole Fonts directory.
    # HTTP font loading also works when a browser refuses CSS local() lookup.
    gr.set_static_paths(paths=[STATIC_DIR, *font_paths])
    theme = gr.themes.Base(
        primary_hue="red",
        secondary_hue="blue",
        neutral_hue="slate",
        font=["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
    )

    initial_state: dict[str, object] = {"events": [], "summary": {}}
    demo_error = ""
    demo_run = os.environ.get("BOXING_VISION_DEMO_RUN", "").strip()
    if demo_run:
        try:
            initial_state = _load_existing_run(demo_run)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            demo_error = f"Не удалось открыть готовый анализ: {escape(str(exc))}"
            initial_state = {"events": [], "summary": {}}
    has_initial_result = bool(initial_state.get("events")) or bool(
        initial_state.get("annotated_video")
    )
    initial_events = (
        [dict(event) for event in initial_state.get("events", []) if isinstance(event, dict)]
        if isinstance(initial_state.get("events"), list)
        else []
    )
    initial_summary = (
        dict(initial_state.get("summary", {}))
        if isinstance(initial_state.get("summary"), dict)
        else {}
    )
    initial_duration = float(initial_state.get("duration_s", 0) or 0)
    initial_run_dir = initial_state.get("run_dir")
    initial_read_only_demo = _is_read_only_demo(initial_state)
    initial_workspace = _workspace(
        initial_events,
        initial_summary,
        initial_duration,
        str(initial_run_dir) if initial_run_dir else None,
    )
    initial_clips = (
        [str(path) for path in sorted(Path(str(initial_state["clips_dir"])).glob("*.mp4"))]
        if initial_state.get("clips_dir")
        else []
    )
    initial_video_export, initial_bundle_export = _available_export_files(initial_state)
    initial_bundle = initial_bundle_export.get("value")

    # Gradio 5.50 executes ``js`` as a load callback. Keep the implementation
    # in its own file, but wrap its IIFE in the callback shape Gradio expects.
    js_source = "() => {\n" + _STATIC_JS.read_text(encoding="utf-8") + "\n}"
    with gr.Blocks(
        theme=theme,
        # Inline the production workspace styles as well as serving the file.
        # This prevents an already-open investor demo from retaining a stale
        # browser-cached stylesheet after a local app restart.
        css=CSS + "\n" + _STATIC_CSS_SOURCE + "\n" + font_css + _desktop_chrome_css(),
        css_paths=[_STATIC_CSS],
        js=js_source,
        title="Boxing Vision · локальный AI-анализ",
        analytics_enabled=False,
        fill_width=True,
    ) as demo:
        anchor_state = gr.State(_empty_enrollment_state())
        result_state = gr.State(initial_state)
        identity_correction_state = gr.State({})
        hero_block = gr.HTML(
            """
            <div id="hero">
              <div class="hero-kicker">Local computer vision · investor prototype</div>
              <div class="hero-title"><span>BOXING</span> <span>VISION</span></div>
              <div class="hero-sub">Профессиональное рабочее место для локального разбора боя: видео, события с оценкой уверенности, карты головы и корпуса и покадровый таймлайн.</div>
              <div class="hero-badge">Данные не покидают этот Mac</div>
            </div>
            """,
            visible=not has_initial_result,
        )
        research_block = gr.HTML(
            "Результаты показывают оценку модели с уровнем уверенности. Показатель интенсивности — относительный индекс, а не физическая сила.",
            elem_id="research-note",
            visible=not has_initial_result,
        )

        with gr.Group(visible=not has_initial_result) as setup_group:  # noqa: SIM117
            with gr.Row(equal_height=False):
                with gr.Column(scale=7, elem_id="upload-card"):
                    gr.HTML('<div class="section-label">01 · исходная трансляция</div>')
                    input_video = gr.Video(
                        label="MP4 / MOV / MKV · до 60 минут",
                        sources=["upload"],
                        format=None,
                        height=440,
                        include_audio=True,
                        show_download_button=False,
                    )
                    preview_image = gr.Image(
                        label="Подтверждение бойцов · настоящие рамки детектора",
                        type="numpy",
                        interactive=False,
                        height=330,
                    )
                    anchor_note = gr.HTML(
                        _enrollment_note("Проверьте A и B на трёх кадрах. Рабочая область необязательна.")
                    )
                    region_mode = gr.Radio(
                        [("Без ограничения", "none"), ("Рабочая область", "manual")],
                        value="none", label="Где искать бойцов",
                    )
                    enrollment_frame = gr.Radio(
                        [("Кадр 1", 0), ("Кадр 2", 1), ("Кадр 3", 2)],
                        value=0, label="Калибровочные кадры",
                    )
                    enrollment_role = gr.Radio(
                        _region_role_choices("none"),
                        value="fighter_a", label="Что исправить кликом",
                    )
                    clear_frame_selection = gr.Button("Очистить выбор на кадре")
                    with gr.Row():
                        confirm_frames = gr.Button("Подтвердить три кадра", variant="primary")
                        reset_anchors = gr.Button("Предложить кадры заново")
                    with gr.Accordion("Заменить выбранный кадр", open=False):
                        enrollment_time = gr.Number(label="Время в исходном видео, сек", value=0, minimum=0)
                        replace_frame = gr.Button("Показать другой момент")
                    video_meta = gr.HTML(
                        '<div class="status-ready">Загрузите локальный видеофайл. Он не покинет этот Mac.</div>'
                    )
                with gr.Column(scale=4, elem_id="settings-card"):
                    gr.HTML('<div class="section-label">02 · параметры боя</div>')
                    fighter_a = gr.Textbox(label="Красный угол · имя", value="Боксёр A")
                    fighter_b = gr.Textbox(label="Синий угол · имя", value="Боксёр B")
                    with gr.Accordion("Профили бойцов · необязательно", open=False):
                        with gr.Row():
                            fighter_a_record = gr.Textbox(label="Рекорд A", placeholder="14–2")
                            fighter_b_record = gr.Textbox(label="Рекорд B", placeholder="12–1")
                        with gr.Row():
                            fighter_a_portrait = gr.Image(
                                label="Портрет A",
                                type="filepath",
                                sources=["upload"],
                                image_mode="RGB",
                                height=150,
                            )
                            fighter_b_portrait = gr.Image(
                                label="Портрет B",
                                type="filepath",
                                sources=["upload"],
                                image_mode="RGB",
                                height=150,
                            )
                    with gr.Row():
                        stance_a = gr.Dropdown(
                            label="Стойка A",
                            choices=[("Неизвестна", "unknown"), ("Правша", "orthodox"), ("Левша", "southpaw")],
                            value="unknown",
                        )
                        stance_b = gr.Dropdown(
                            label="Стойка B",
                            choices=[("Неизвестна", "unknown"), ("Правша", "orthodox"), ("Левша", "southpaw")],
                            value="unknown",
                        )
                    hud_mode = gr.Dropdown(
                        label="HUD в итоговом MP4",
                        choices=[
                            ("Компактный investor HUD", "compact"),
                            ("Технический HUD", "technical"),
                            ("Без HUD · только bbox/pose", "none"),
                        ],
                        value="compact",
                    )
                    timing_mode = gr.Radio(
                        [("Всё видео непрерывно", "continuous"), ("По расписанию раундов", "scheduled")],
                        value="continuous", label="Режим времени",
                        info="В непрерывном режиме анализ не выключается через три минуты. Сцены и повторы по-прежнему проверяются.",
                    )
                    with gr.Row(visible=False) as schedule_settings:
                        rounds = gr.Number(label="Раундов", value=12, precision=0, minimum=1, maximum=24)
                        round_length = gr.Number(label="Секунд в раунде", value=180, precision=0, minimum=60, maximum=300)
                        rest_length = gr.Number(label="Перерыв, сек", value=60, precision=0, minimum=0, maximum=180)
                    with gr.Row():
                        start_s = gr.Number(label="Начало боя, сек", value=0, minimum=0)
                        end_s = gr.Number(label="Конец, сек (0 = весь файл)", value=0, minimum=0)
                    with gr.Accordion("Подтверждённые нокдауны · необязательно", open=False):
                        knockdowns_a = gr.Textbox(label="Раунды, где A был в нокдауне", placeholder="2, 5")
                        knockdowns_b = gr.Textbox(label="Раунды, где B был в нокдауне", placeholder="3, 3")
                        gr.Markdown("Только ручное подтверждение влияет на 10–8.")
                    with gr.Row():
                        analyze_button = gr.Button(
                            "Запустить анализ",
                            variant="primary",
                            elem_id="analyze-button",
                            scale=4,
                        )
                        cancel_button = gr.Button("Отменить", variant="stop", scale=1)
                    gr.Markdown(
                        "Первый запуск загружает открытые веса RTMPose. Обработка выполняется локально и потоково."
                    )

        initial_status = (
            f'<div class="status-ready">{demo_error}</div>'
            if demo_error
            else ""
        )
        job_status = gr.HTML(initial_status, elem_id="job-status")

        with gr.Group(
            visible=has_initial_result,
            elem_id="bv-result-workspace",
        ) as result_group:
            with gr.Row(elem_classes=["result-header"]):
                gr.HTML("<h2>Разбор боя</h2><span>Интерактивное рабочее место</span>")
                new_analysis_button = gr.Button("Новый анализ", size="sm")
            with gr.Row(equal_height=False, elem_id="bv-result-grid"):
                fighter_a_panel = gr.HTML(
                    render_fighter_panel_mount("fighter_a"),
                    elem_id="bv-panel-a-card",
                    padding=False,
                )
                with gr.Column(elem_id="bv-center-stack"):
                    with gr.Column(elem_id="result-video-card"):
                        output_video = gr.Video(
                            value=str(initial_state.get("workspace_video") or initial_state.get("annotated_video")) if has_initial_result else None,
                            label="Видео боя",
                            autoplay=False,
                            show_download_button=False,
                            elem_id="annotated-video",
                        )
                    workspace_html = gr.HTML(
                        initial_workspace,
                        elem_id="bv-workspace-chrome",
                        padding=False,
                    )
                fighter_b_panel = gr.HTML(
                    render_fighter_panel_mount("fighter_b"),
                    elem_id="bv-panel-b-card",
                    padding=False,
                )

            with gr.Tabs():
                with gr.Tab("Итоги"):
                    overview_html = gr.HTML(
                        _overview(initial_summary) if has_initial_result else ""
                    )
                with gr.Tab("Проверка личностей и сцен", visible=not initial_read_only_demo) as identity_review_tab:
                    with gr.Accordion("Исправить бойца в кадре", open=True):
                        with gr.Row():
                            identity_correction_time = gr.Number(label="Момент видео, сек", value=0, minimum=0)
                            identity_correction_current = gr.Button("Взять текущий кадр видео")
                            identity_correction_show = gr.Button("Показать выбранный момент")
                        identity_correction_role = gr.Radio(
                            [("Это красный A", "FIGHTER_A"), ("Это синий B", "FIGHTER_B")],
                            value="FIGHTER_A", label="Выберите бойца, затем нажмите на его рамку",
                        )
                        identity_correction_preview = gr.Image(label="Исправление только выбранного сегмента",
                                                               type="numpy", interactive=False, height=360)
                        identity_correction_note = gr.HTML(_enrollment_note(
                            "Остановите видео в нужном месте и возьмите текущий кадр. Изменение не затрагивает весь ID траектории."))
                    initial_review_choices, _, initial_review_note, _ = _refresh_identity_review(initial_state)
                    identity_review_item = gr.Dropdown(
                        choices=initial_review_choices.get("choices", []), value=None,
                        label="Эпизод для проверки", interactive=True,
                    )
                    identity_review_preview = gr.Image(label="Человек или сцена из сохранённого прохода", interactive=False, height=360)
                    identity_review_note = gr.HTML(initial_review_note)
                    identity_review_action = gr.Radio(choices=[], label="Результат проверки", interactive=True)
                    identity_review_save = gr.Button("Сохранить и пересчитать", variant="primary")
                    initial_preview_note, initial_preview_button = _tracking_preview_status(initial_state)
                    tracking_preview_note = gr.HTML(initial_preview_note)
                    tracking_preview_refresh = gr.Button("Обновить трекинг в плеере", variant="secondary",
                                                         interactive=initial_preview_button.get("interactive", False))
                    gr.Markdown("Исправьте несколько эпизодов, затем обновите плеер один раз. Повторное распознавание не запускается; финальный экспорт выполняется отдельно.")
                with gr.Tab("Полный журнал"):
                    event_table = gr.Dataframe(
                        value=_event_rows(initial_events, initial_summary) if has_initial_result else [],
                        headers=[
                            "ID события", "Таймкод", "Интервал", "Раунд", "Атакующий",
                            "Рука", "Тип", "Цель", "Результат", "Confidence, %",
                            "Интенсивность", "Контекст", "Статус проверки",
                        ],
                        datatype=[
                            "str", "str", "str", "number", "str", "str", "str",
                            "str", "str", "number", "number", "str", "str",
                        ],
                        row_count=(0, "dynamic"),
                        col_count=(13, "fixed"),
                        interactive=False,
                        show_search="filter",
                        show_row_numbers=True,
                        max_height=560,
                        wrap=True,
                        elem_id="events-table",
                    )
                with gr.Tab("Экспорт", visible=not initial_read_only_demo) as export_tab:
                    with gr.Row():
                        rebuild_button = gr.Button("Пересобрать MP4", variant="secondary", visible=not initial_read_only_demo, interactive=not initial_read_only_demo)
                    with gr.Column(elem_id="downloads-card"):
                        export_video_file = gr.File(value=initial_video_export.get("value"),
                                                    visible=initial_video_export.get("visible", False),
                                                    label="Итоговое видео MP4")
                        bundle_file = gr.File(value=initial_bundle, visible=initial_bundle_export.get("visible", False), label="Полный пакет ZIP")
                        with gr.Row():
                            events_file = gr.File(
                                value=str(initial_state.get("events_path")) if has_initial_result else None,
                                label="events.json",
                            )
                            summary_file = gr.File(
                                value=str(initial_state.get("summary_path")) if has_initial_result else None,
                                label="summary.json",
                            )
                        clips_file = gr.File(
                            value=initial_clips,
                            label="Ключевые эпизоды",
                            file_count="multiple",
                        )
                        gr.HTML(
                            '<div class="legal-foot">После review MP4 и ZIP требуют render-only пересборки. Для показа используйте видео, на которое у вас есть права.</div>'
                        )

            review_command = gr.Textbox(
                value="",
                elem_id="bv-review-command",
                visible=False,
                container=False,
            )
            review_submit = gr.Button(
                "review",
                elem_id="bv-review-submit",
                visible=False,
            )

        input_video.change(
            _prepare_enrollment_for_region,
            inputs=[input_video, start_s, region_mode],
            outputs=[video_meta, preview_image, anchor_state, anchor_note, enrollment_frame, enrollment_time],
            concurrency_limit=1,
            concurrency_id="boxing-analysis",
        )
        start_s.change(
            _prepare_enrollment_for_region,
            inputs=[input_video, start_s, region_mode],
            outputs=[video_meta, preview_image, anchor_state, anchor_note, enrollment_frame, enrollment_time],
            concurrency_limit=1,
            concurrency_id="boxing-analysis",
        )
        reset_anchors.click(
            _prepare_enrollment_for_region,
            inputs=[input_video, start_s, region_mode],
            outputs=[video_meta, preview_image, anchor_state, anchor_note, enrollment_frame, enrollment_time],
            concurrency_limit=1,
            concurrency_id="boxing-analysis",
        )
        preview_image.select(
            _select_enrollment_target,
            inputs=[anchor_state, enrollment_role],
            outputs=[preview_image, anchor_state, anchor_note, enrollment_role],
            queue=False,
        )
        region_mode.change(
            _change_working_region,
            inputs=[anchor_state, region_mode],
            outputs=[preview_image, anchor_state, anchor_note, enrollment_role],
            queue=False,
        )
        timing_mode.change(lambda value: gr.update(visible=value == "scheduled"),
                           inputs=[timing_mode], outputs=[schedule_settings], queue=False)
        enrollment_frame.change(
            _show_enrollment_frame,
            inputs=[anchor_state, enrollment_frame],
            outputs=[preview_image, anchor_state, anchor_note, enrollment_time],
            queue=False,
        )
        clear_frame_selection.click(
            _clear_enrollment_selection,
            inputs=[anchor_state, enrollment_role],
            outputs=[preview_image, anchor_state, anchor_note, enrollment_role],
            queue=False,
        )
        confirm_frames.click(
            _confirm_enrollment_ui,
            inputs=[anchor_state],
            outputs=[preview_image, anchor_state, anchor_note],
            queue=False,
        )
        replace_frame.click(
            _replace_enrollment_frame,
            inputs=[input_video, anchor_state, enrollment_time],
            outputs=[preview_image, anchor_state, anchor_note],
            concurrency_limit=1,
            concurrency_id="boxing-analysis",
        )
        analyze_button.click(
            _run_analysis,
            inputs=[
                input_video,
                anchor_state,
                fighter_a,
                fighter_b,
                fighter_a_record,
                fighter_b_record,
                fighter_a_portrait,
                fighter_b_portrait,
                stance_a,
                stance_b,
                hud_mode,
                rounds,
                round_length,
                rest_length,
                start_s,
                end_s,
                knockdowns_a,
                knockdowns_b,
                timing_mode,
            ],
            outputs=[
                job_status,
                output_video,
                workspace_html,
                overview_html,
                event_table,
                bundle_file,
                events_file,
                summary_file,
                clips_file,
                result_group,
                setup_group,
                hero_block,
                research_block,
                result_state,
            ],
            concurrency_limit=1,
            concurrency_id="boxing-analysis",
        )
        cancel_button.click(_request_cancel, outputs=job_status, queue=False)
        result_state.change(
            _refresh_identity_review,
            inputs=[result_state],
            outputs=[identity_review_item, identity_review_preview, identity_review_note, identity_review_action],
            queue=False,
        )
        result_state.change(
            _available_export_files,
            inputs=[result_state],
            outputs=[export_video_file, bundle_file],
            queue=False,
        )
        result_state.change(
            _tracking_preview_status,
            inputs=[result_state], outputs=[tracking_preview_note, tracking_preview_refresh], queue=False,
        )
        result_state.change(
            _result_mutation_controls, inputs=[result_state],
            outputs=[identity_review_tab, rebuild_button, export_tab], queue=False,
        )
        tracking_preview_refresh.click(
            _refresh_tracking_player, inputs=[result_state],
            outputs=[tracking_preview_note, output_video, workspace_html, overview_html, event_table,
                     result_state, bundle_file, summary_file],
            concurrency_limit=1, concurrency_id="boxing-analysis",
        )
        identity_correction_current.click(
            None, inputs=[identity_correction_time], outputs=[identity_correction_time], queue=False,
            js="""(value) => {
              const video = document.querySelector('#annotated-video video');
              if (video) { video.pause(); return Math.max(0, Number(video.currentTime) || 0); }
              return value;
            }""",
        ).then(
            _load_identity_correction_frame, inputs=[result_state, identity_correction_time],
            outputs=[identity_correction_preview, identity_correction_state, identity_correction_note],
            concurrency_limit=1, concurrency_id="boxing-analysis",
        )
        identity_correction_show.click(
            _load_identity_correction_frame, inputs=[result_state, identity_correction_time],
            outputs=[identity_correction_preview, identity_correction_state, identity_correction_note],
            concurrency_limit=1, concurrency_id="boxing-analysis",
        )
        identity_correction_preview.select(
            _select_identity_correction,
            inputs=[identity_correction_state, identity_correction_role, result_state],
            outputs=[identity_correction_note, workspace_html, overview_html, event_table, result_state,
                     bundle_file, events_file, summary_file],
            concurrency_limit=1, concurrency_id="boxing-analysis",
        )
        identity_review_item.change(
            _show_identity_review,
            inputs=[identity_review_item, result_state],
            outputs=[identity_review_preview, identity_review_note, identity_review_action],
            queue=False,
        )
        identity_review_save.click(
            _apply_identity_review,
            inputs=[identity_review_item, identity_review_action, result_state],
            outputs=[job_status, workspace_html, overview_html, event_table, result_state,
                     bundle_file, events_file, summary_file],
            concurrency_limit=1,
            concurrency_id="boxing-analysis",
        )
        review_submit.click(
            _review_workspace_event,
            inputs=[review_command, result_state],
            outputs=[
                job_status,
                workspace_html,
                overview_html,
                event_table,
                result_state,
                bundle_file,
                events_file,
                summary_file,
            ],
            concurrency_limit=1,
            concurrency_id="boxing-analysis",
        )
        rebuild_button.click(
            _rebuild_reviewed_video,
            inputs=[result_state],
            outputs=[
                job_status,
                output_video,
                workspace_html,
                overview_html,
                event_table,
                result_state,
                bundle_file,
                summary_file,
            ],
            concurrency_limit=1,
            concurrency_id="boxing-analysis",
        )
        new_analysis_button.click(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            ),
            outputs=[setup_group, result_group, hero_block, research_block],
            queue=False,
            js="""() => {
              const workspace = window.BoxingVisionWorkspace;
              if (workspace?.isTheaterActive()) workspace.toggleTheater();
              return [];
            }""",
        )

    del fighter_a_panel, fighter_b_panel
    return demo


__all__ = ["build_app"]
