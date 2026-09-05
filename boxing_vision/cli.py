from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AnalysisConfig
from .pipeline import analyze_video


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Локальный анализ боксёрской трансляции")
    parser.add_argument("input", type=Path, help="MP4, MOV или MKV")
    parser.add_argument("--fighter-a", default="Красный угол")
    parser.add_argument("--fighter-b", default="Синий угол")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--round-length", type=int, default=180)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--analysis-fps", type=float, default=15.0)
    parser.add_argument("--enrollment-json", type=Path, help="Подтверждённые enrollment_samples[3] и ring_rois")
    parser.add_argument("--legacy-anchor", action="store_true", help="Совместимость со старым CLI, без investor quality gate")
    parser.add_argument(
        "--hud-mode",
        choices=("compact", "technical", "none"),
        default="technical",
        help="technical сохраняет прежний подробный HUD CLI",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    enrollment = json.loads(args.enrollment_json.read_text(encoding="utf-8")) if args.enrollment_json else {}
    if not enrollment and not args.legacy_anchor:
        raise SystemExit("Нужна --enrollment-json с тремя подтверждёнными кадрами. Калибровка доступна на сайте; --legacy-anchor сохраняет старый режим.")
    config = AnalysisConfig(
        fighter_a_name=args.fighter_a,
        fighter_b_name=args.fighter_b,
        scheduled_rounds=args.rounds,
        round_length_s=args.round_length,
        fight_start_s=args.start,
        fight_end_s=args.end,
        analysis_fps=args.analysis_fps,
        hud_mode=args.hud_mode,
        enrollment_mode="legacy_anchor" if args.legacy_anchor else "auto_confirm",
        enrollment_confirmed=bool(enrollment.get("confirmed")),
        enrollment_samples=tuple(enrollment.get("enrollment_samples", [])),
        enrollment_frames=tuple(float(item["time_s"]) for item in enrollment.get("enrollment_samples", [])),
        ring_rois=tuple(tuple(point) for point in enrollment.get("ring_rois", [])),
    )

    def progress(value: float, description: str) -> None:
        print(f"{value:6.1%}  {description}", flush=True)

    result = analyze_video(args.input, config, progress)
    print(f"Видео: {result.annotated_video}")
    print(f"События: {result.events_path}")
    print(f"Сводка: {result.summary_path}")


if __name__ == "__main__":
    main()
