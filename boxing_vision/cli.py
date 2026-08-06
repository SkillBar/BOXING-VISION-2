from __future__ import annotations

import argparse
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
    parser.add_argument("--analysis-fps", type=float, default=10.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = AnalysisConfig(
        fighter_a_name=args.fighter_a,
        fighter_b_name=args.fighter_b,
        scheduled_rounds=args.rounds,
        round_length_s=args.round_length,
        fight_start_s=args.start,
        fight_end_s=args.end,
        analysis_fps=args.analysis_fps,
    )

    def progress(value: float, description: str) -> None:
        print(f"{value:6.1%}  {description}", flush=True)

    result = analyze_video(args.input, config, progress)
    print(f"Видео: {result.annotated_video}")
    print(f"События: {result.events_path}")
    print(f"Сводка: {result.summary_path}")


if __name__ == "__main__":
    main()
