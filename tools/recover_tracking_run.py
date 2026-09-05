"""Recompute an existing detector cache into a NEW run, keeping the source intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxing_vision.artifacts import atomic_write_json, create_job_artifacts
from boxing_vision.cache_review import redecode_from_cache
from boxing_vision.pipeline import (
    rebuild_from_cache,
    rebuild_tracking_preview_from_cache,
)


def clone_run(source: Path, runs_dir: Path) -> Path:
    for name in ("events.json", "summary.json", ".render_cache/detections.jsonl.gz",
                 ".render_cache/first_pass.json", ".render_cache/config.json", ".render_cache/normalized.mp4"):
        if not (source / name).is_file():
            raise ValueError(f"Нет исходного артефакта: {name}")
    target = create_job_artifacts(runs_dir).run_dir
    for entry in source.iterdir():
        if entry.name == ".work" or entry.suffix == ".zip":
            continue
        if entry.is_dir():
            shutil.copytree(entry, target / entry.name, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target / entry.name)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--timing-mode", choices=("continuous", "scheduled"), default="continuous")
    parser.add_argument("--region-mode", choices=("none", "manual", "auto"), default="none")
    parser.add_argument("--refresh-appearance", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    target = clone_run(source, (args.runs_dir or source.parent).resolve())
    print(json.dumps({"new_run": str(target)}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    if args.refresh_appearance:
        from boxing_vision.appearance_refresh import refresh_cached_appearance
        refreshed = target / ".appearance-refresh"
        refresh_cached_appearance(source / ".render_cache", refreshed,
            progress=lambda *message: print("appearance", *message, flush=True))
        for name in ("detections.jsonl.gz", "identity_profile.json", "appearance_refresh.json"):
            shutil.copy2(refreshed / name, target / ".render_cache" / name)
    summary = redecode_from_cache(target, timing_mode=args.timing_mode, region_mode=args.region_mode)
    report = {"source_run": source.name, "run": target.name,
              "identity_decoder": "track-graph-v3", "quality": summary["quality"],
              "evaluation_kind": "internal_confirmation_not_expert_precision",
              "recompute_seconds": round(time.perf_counter() - started, 3)}
    for name in ("events.json", "summary.json", ".render_cache/detections.jsonl.gz"):
        with (source / name).open("rb") as handle:
            report.setdefault("source_sha256", {})[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    atomic_write_json(target / "recovery_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.skip_render:
        last = -1
        def progress(value, message):
            nonlocal last
            percent = int(value * 100)
            if percent != last:
                last = percent
                print(f"{percent}% {message}", flush=True)
        if summary["quality"].get("required_review_count", 0):
            result = rebuild_tracking_preview_from_cache(target, progress)
        else:
            result = rebuild_from_cache(target, progress)
        print(json.dumps({"video": str(result)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
