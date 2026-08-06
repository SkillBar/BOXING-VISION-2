from __future__ import annotations

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
from .config import AnalysisConfig
from .contracts import PunchEvent
from .pipeline import AnalysisCancelledError, analyze_video
from .scoring import build_fight_summary
from .video import validate_video

_GRADIO_PROGRESS = gr.Progress()
_CANCEL_EVENT = Event()

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
    <div class="legal-foot">При первичной инициализации левый боксёр получает красный ID, правый — синий. Если углы перепутаны, поменяйте имена местами.</div>
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
        <div class="metric"><b>{impact:.0f}</b><span>impact proxy / 100</span></div>
      </div>
      <div class="tech-line">Блоки: {int(stats.get('blocked', 0) or 0)} · Промахи: {int(stats.get('missed', 0) or 0)} · Неясно: {int(stats.get('unclear', 0) or 0)}<br>{escape(tech_text)}</div>
    </div>
    """


def _overview(summary: dict[str, object]) -> str:
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
            f'{escape(str(card.get("reason", "Экспериментальная оценка")))}</span></div>'
        )
        for card in round_scores
        if isinstance(card, dict)
    )
    quality_warning = (
        " · низкая уверенность, результат требует ручной проверки"
        if tracking < 0.55 or event_confidence < 0.55
        else ""
    )
    return f"""
    <div class="winner-card">
      <div class="winner-eyebrow">Экспериментальный прогноз · неофициально</div>
      <div class="winner-name">{winner_name}</div>
      <div class="winner-score">Сумма карточек: {total_a} — {total_b} · уверенность {winner_confidence:.0%}</div>
      <div class="quality-line">Качество трекинга {tracking:.0%} · средняя уверенность событий {event_confidence:.0%}{quality_warning}</div>
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
    scheduled_rounds = max(1, len(round_cards))
    metadata = dict(previous.get("metadata")) if isinstance(previous.get("metadata"), dict) else {}
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
                fighter["id"] = fighter_id
                fighter["corner"] = corner
                fighter["stats"] = {
                    key: value
                    for key, value in fighter.items()
                    if key not in {"id", "name", "corner", "stats"}
                }
    winner_id = rebuilt.get("winner_id")
    rebuilt["winner"] = {
        "fighter_id": winner_id,
        "name": str(rebuilt.get("winner_name") or "Недостаточно данных"),
        "confidence": float(rebuilt.get("confidence", 0) or 0),
        "label": "Экспериментальный прогноз" if winner_id else "Экспериментальная ничья",
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
    return rebuilt


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
        "Статистика и экспериментальный счёт пересчитаны. Разметка внутри MP4 остаётся "
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


def _run_analysis(
    video: object,
    anchor_state: dict[str, object] | None,
    fighter_a: str,
    fighter_b: str,
    stance_a: str,
    stance_b: str,
    rounds: float,
    round_length: float,
    rest_length: float,
    start_s: float,
    end_s: float,
    knockdowns_a: str,
    knockdowns_b: str,
    progress: gr.Progress = _GRADIO_PROGRESS,
):
    _CANCEL_EVENT.clear()
    try:
        source = _video_path(video)
        fighter_a_anchor, fighter_b_anchor = _confirmed_anchors(anchor_state)
        scheduled_rounds = int(rounds)
        config = AnalysisConfig(
            fighter_a_name=(fighter_a or "Красный угол").strip(),
            fighter_b_name=(fighter_b or "Синий угол").strip(),
            fighter_a_stance=stance_a,
            fighter_b_stance=stance_b,
            fighter_a_anchor=fighter_a_anchor,
            fighter_b_anchor=fighter_b_anchor,
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
        bundle = _bundle_result(result)
        status = (
            f'<div class="status-done">Анализ завершён. Найдено кандидатов событий: '
            f'{len(events)}. Все оценки остаются экспериментальными.</div>'
        )
        return (
            status,
            str(result.annotated_video),
            _overview(result.summary),
            _timeline(events, duration_s),
            _event_rows(events, result.summary),
            str(bundle),
            str(result.events_path),
            str(result.summary_path),
            clips,
            gr.update(visible=True),
            gr.update(choices=_seek_choices(events, result.summary), value=None),
            {
                "events": events,
                "summary": result.summary,
                "duration_s": duration_s,
                "run_dir": str(result.run_dir),
                "annotated_video": str(result.annotated_video),
                "events_path": str(result.events_path),
                "summary_path": str(result.summary_path),
                "log_path": str(result.log_path),
                "clips_dir": str(result.clips_dir),
            },
        )
    except AnalysisCancelledError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Анализ остановлен: {exc}") from exc
    finally:
        _CANCEL_EVENT.clear()


def build_app() -> gr.Blocks:
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
              <div class="hero-sub">Загрузите трансляцию боя. Система локально отследит двух боксёров, выделит кандидаты ударов и соберёт размеченный ролик, статистику, таймкоды и экспериментальный прогноз.</div>
              <div class="hero-badge">ДАННЫЕ НЕ ПОКИДАЮТ ЭТОТ MAC</div>
            </div>
            """
        )
        gr.HTML(
            "Все результаты — confidence-aware исследовательская аналитика. Impact proxy не является физической силой, а прогноз победителя не является официальным судейским решением.",
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
                reset_anchors = gr.Button("СБРОСИТЬ ВЫБОР БОЙЦОВ", size="sm")
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
                        "ЗАПУСТИТЬ АНАЛИЗ",
                        variant="primary",
                        elem_id="analyze-button",
                        scale=4,
                    )
                    cancel_button = gr.Button("ОТМЕНИТЬ", variant="stop", scale=1)
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


__all__ = ["build_app"]
