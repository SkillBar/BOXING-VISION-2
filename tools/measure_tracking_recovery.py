"""Compare whole-run cached evidence; never label internal counts as precision."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxing_vision.artifacts import atomic_write_json


def read_lines(path):
    if not path.is_file():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def measure(root: Path) -> dict:
    cache = root / ".render_cache"
    observations = read_lines(cache / "observations.jsonl.gz")
    display = read_lines(cache / "display_tracks.jsonl.gz")
    frames = read_lines(cache / "detections.jsonl.gz")
    diagnostics = read_lines(root / "tracking_diagnostics.jsonl")
    summary = json.loads((root / "summary.json").read_text())
    events = json.loads((root / "events.json").read_text())
    if isinstance(events, dict):
        events = events.get("events", [])
    by_time = defaultdict(list)
    for observation in observations:
        by_time[int(observation["timestamp_ms"])].append(observation)
    show_time = defaultdict(list)
    for item in display:
        show_time[int(item["timestamp_ms"])].append(item)
    intervals = []
    duration = round(float(summary["metadata"]["duration_s"]) * 1000)
    for start, end in ((0, duration), (12000, 16000), (15000, 32000),
                       (41000, 61000), (91000, 175000), (180000, duration)):
        stamps = [frame["timestamp_ms"] for frame in frames if start <= frame["timestamp_ms"] < end]
        observations_here = [item for stamp in stamps for item in by_time[stamp]]
        display_here = [item for stamp in stamps for item in show_time[stamp] if item["display_state"] != "LOST"]
        intervals.append({
            "start_ms": start, "end_ms": end, "analysis_samples": len(stamps),
            "observation_roles": dict(Counter(item["fighter_id"] for item in observations_here)),
            "display_states": dict(Counter(item["display_state"] for item in display_here)),
            "neutral_observed_boxes": sum(item["identity_state"] == "UNKNOWN" and item["display_state"] == "OBSERVED" for item in display_here),
            "samples_with_two_measured_display_boxes": sum(sum(item["display_state"] == "OBSERVED" for item in show_time[stamp]) == 2 for stamp in stamps),
        })
    hashes = {}
    for name in ("events.json", "summary.json", ".render_cache/detections.jsonl.gz"):
        with (root / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"run": root.name, "measurement": "cached_evidence_not_expert_ground_truth",
            "expert_precision_measured": False, "visibility_coverage_measured": False,
            "quality": summary["quality"], "intervals": intervals,
            "last_event_peak_ms": max((item["peak_ms"] for item in events), default=None),
            "last_observation_ms_by_role": {role: max((item["timestamp_ms"] for item in observations if item["fighter_id"] == role), default=None)
                                             for role in ("fighter_a", "fighter_b")},
            "rejection_reasons": dict(Counter(item.get("reason") for item in diagnostics if not item.get("selected_fighter_id"))),
            "sha256": hashes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("recovered", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.resolve().parent in {args.original.resolve(), args.recovered.resolve()}:
        parser.error("Save the comparison outside either historical/result run")
    result = {"original": measure(args.original), "recovered": measure(args.recovered),
              "acceptance_passed": False, "reason": "Requires continuous expert visibility/identity annotation; counts do not establish accuracy"}
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
