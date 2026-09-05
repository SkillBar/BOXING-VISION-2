"""Reproduce the identity preflight from cached detections, without loading ML.

Outputs internal confirmation coverage, NOT ground-truth precision or IDSW.
The original run is read-only; --output must point outside the run directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxing_vision.artifacts import atomic_write_json
from boxing_vision.pose import TwoFighterTracker
from boxing_vision.preflight import build_preflight_report
from boxing_vision.tracking import OfflineIdentityDecoder, TrackingFrame


def evaluate(run_dir: Path, window_ms: int = 10000) -> dict:
    cache = run_dir / ".render_cache"
    profile_path = cache / "identity_profile.json"
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text())
    else:
        profile = json.loads((cache / "first_pass.json").read_text())["identity_profile"]
    tracker = TwoFighterTracker()
    tracker.import_identity_profile(profile)
    with gzip.open(cache / "detections.jsonl.gz", "rt", encoding="utf-8") as handle:
        frames = []
        for line in handle:
            frame = TrackingFrame.from_dict(json.loads(line))
            if frame.timestamp_ms >= window_ms:
                break
            frames.append(frame)
    observations, diagnostics, tracklets = [], [], []
    for _, shot in groupby(frames, key=lambda frame: frame.shot_id):
        decoder = OfflineIdentityDecoder(tracker)
        observations.extend(decoder.decode(list(shot)))
        diagnostics.extend(decoder.diagnostics)
        tracklets.extend(decoder.tracklets)
    frame_states = [{"timestamp_ms": frame.timestamp_ms, "shot_id": frame.shot_id,
                     "is_scene_cut": frame.is_scene_cut, "scene_state": frame.scene_state} for frame in frames]
    report = build_preflight_report(observations, frame_states, diagnostics,
                                    window_ms=window_ms, minimum_margin=tracker.gallery.min_margin)
    sources = defaultdict(list)
    for row in diagnostics:
        if row["source_track_id"] is not None:
            sources[f"shot-{row['shot_id']}-track-{row['source_track_id']}"].append(row)
    with (cache / "detections.jsonl.gz").open("rb") as handle:
        detector_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "run": run_dir.name,
        "evaluation_kind": "internal_identity_confirmation_not_ground_truth_accuracy",
        "identity_decoder_version": OfflineIdentityDecoder.version,
        "detector_cache_sha256": detector_sha,
        "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        "preflight": report,
        "tracks": {key: {"frames": len(rows),
                         "states": dict(Counter(row["identity_state"] for row in rows)),
                         "reasons": dict(Counter(row["reason"] for row in rows))}
                   for key, rows in sources.items()},
        "tracklets": tracklets,
        "observations": [{"timestamp_ms": o.timestamp_ms, "shot_id": o.shot_id,
                          "source_track_id": o.source_track_id, "fighter_id": o.fighter_id,
                          "identity_confidence": o.identity_confidence, "identity_margin": o.identity_margin}
                         for o in observations],
        "diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-ms", type=int, default=10000)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.window_ms <= 0:
        parser.error("window-ms must be positive")
    if args.output and args.output.resolve().is_relative_to(root):
        parser.error("output must not overwrite the historical run")
    report = evaluate(root, args.window_ms)
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("run", "preflight", "tracks")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
