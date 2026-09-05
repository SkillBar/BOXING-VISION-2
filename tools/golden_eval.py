"""Evaluate predictions against explicit human labels, never against predictions.

Event JSON is a list of PunchEvent-compatible dictionaries. Identity truth is
{"frames": [{"timestamp_ms": 0, "scene_state": "ACTIVE_FIGHT", "persons":
 [{"person_id": "person-1", "identity": "fighter_a", "bbox": [x1,y1,x2,y2],
   "visible": true}]}]}. Prediction identities use PoseObservation dictionaries.
The CLI will not generate ground truth or infer dataset rights.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _events(value: Any) -> list[dict]:
    result = value.get("events") if isinstance(value, dict) else value
    if not isinstance(result, list):
        raise TypeError("Expected an explicit list of event labels")
    seen = set()
    for event in result:
        for required in ("event_id", "attacker_id", "hand", "peak_ms"):
            if required not in event:
                raise ValueError(f"Event missing {required}")
        key = (event.get("fight_id", ""), event["event_id"])
        if key in seen:
            raise ValueError("Duplicate event_id within one fight")
        seen.add(key)
        if not np.isfinite(event["peak_ms"]):
            raise ValueError("Non-finite event timestamp")
    return result


def match_events(
    predictions: list[dict], truth: list[dict], tolerance_ms=300, *, require_hand=True
):
    """Maximum-cardinality, minimum-time-error one-to-one assignment."""
    if tolerance_ms <= 0:
        raise ValueError("Tolerance must be positive")
    groups: dict[tuple, tuple[list[int], list[int]]] = {}
    for side, events in enumerate((predictions, truth)):
        for index, event in enumerate(events):
            key = (
                event.get("fight_id", ""),
                event["attacker_id"],
                event["hand"] if require_hand else None,
            )
            groups.setdefault(key, ([], []))[side].append(index)
    matches = []
    for left, right in groups.values():
        if not left or not right:
            continue
        errors = np.abs(
            np.subtract.outer(
                [predictions[i]["peak_ms"] for i in left],
                [truth[j]["peak_ms"] for j in right],
            )
        )
        # Invalid costs exceed the sum of all valid assignment costs.
        cost = np.where(
            errors <= tolerance_ms, errors / tolerance_ms, len(left) + len(right) + 1
        )
        rows, columns = linear_sum_assignment(cost)
        matches.extend(
            (left[int(i)], right[int(j)])
            for i, j in zip(rows, columns)
            if errors[i, j] <= tolerance_ms
        )
    return sorted(matches)


def _divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _f1(tp, fp, fn):
    return _divide(2 * tp, 2 * tp + fp + fn)


def expected_calibration_error(scores, correctness, bins=10):
    if bins <= 0 or len(scores) != len(correctness):
        raise ValueError(
            "Calibration requires positive bins and equal score/label lengths"
        )
    if not scores:
        return None
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Calibration scores must be finite numbers in [0,1]")
    correct = np.asarray(correctness, dtype=float)
    assignments = np.minimum((values * bins).astype(int), bins - 1)
    return float(
        sum(
            np.mean(assignments == index)
            * abs(
                np.mean(values[assignments == index])
                - np.mean(correct[assignments == index])
            )
            for index in range(bins)
            if np.any(assignments == index)
        )
    )


def _label(event, field):
    value = event.get(field, "unknown")
    if field == "outcome" and value == "likely_landed":
        return "landed"
    return value


def _attribute_metrics(predictions, truth, matches, field, confidence_field):
    rows = [
        (predictions[i], truth[j])
        for i, j in matches
        if _label(truth[j], field) not in {None, "unknown", "unclear"}
    ]
    classes = sorted({_label(target, field) for _, target in rows})
    by_class = {}
    for label in classes:
        tp = sum(
            _label(pred, field) == label and _label(target, field) == label
            for pred, target in rows
        )
        fp = sum(
            _label(pred, field) == label and _label(target, field) != label
            for pred, target in rows
        )
        fn = sum(
            _label(pred, field) != label and _label(target, field) == label
            for pred, target in rows
        )
        by_class[label] = {"tp": tp, "fp": fp, "fn": fn, "f1": _f1(tp, fp, fn)}
    correct = [_label(pred, field) == _label(target, field) for pred, target in rows]
    scored = [
        (pred[confidence_field], same)
        for (pred, _), same in zip(rows, correct)
        if pred.get(confidence_field) is not None
    ]
    return {
        "matched_labeled_count": len(rows),
        "accuracy": _divide(sum(correct), len(correct)),
        "macro_f1": float(np.mean([value["f1"] for value in by_class.values()]))
        if by_class
        else None,
        "classes": by_class,
        "prediction_abstentions": sum(
            _label(pred, field) in {None, "unknown", "unclear"} for pred, _ in rows
        ),
        "ece": expected_calibration_error(
            [x[0] for x in scored], [x[1] for x in scored]
        ),
        "ece_n": len(scored),
        "ece_note": "Matched labeled samples only; descriptive ECE does not establish calibrated model provenance",
    }


def evaluate_events(
    prediction_payload, truth_payload, *, tolerance_ms=300, active_duration_ms=None
):
    predictions = [
        event
        for event in _events(prediction_payload)
        if str(event.get("review_status", "")).upper() != "REJECTED"
        and not event.get("is_replay", False)
        and event.get("proposal_status") != "abstained"
    ]
    truth = _events(truth_payload)
    matches = match_events(predictions, truth, tolerance_ms)
    tp, fp, fn = (
        len(matches),
        len(predictions) - len(matches),
        len(truth) - len(matches),
    )
    matched_predictions, matched_truth = (
        {i for i, _ in matches},
        {j for _, j in matches},
    )
    hand_matches = match_events(predictions, truth, tolerance_ms, require_hand=False)
    # Hand accuracy uses independent attacker/time matching, otherwise strict
    # attacker+hand detection matching would make this statistic tautologically 1.
    attributes = {
        "technique": _attribute_metrics(
            predictions, truth, matches, "technique", "classification_confidence"
        ),
        "target": _attribute_metrics(
            predictions, truth, matches, "target", "target_confidence"
        ),
        "outcome": _attribute_metrics(
            predictions, truth, matches, "outcome", "outcome_confidence"
        ),
        "hand": _attribute_metrics(
            predictions, truth, hand_matches, "hand", "hand_confidence"
        ),
    }
    round_counts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for side, events in enumerate((predictions, truth)):
        for event in events:
            round_counts[
                (event.get("fight_id", ""), event.get("round", 0), event["attacker_id"])
            ][side] += 1
    return {
        "ground_truth_source": "provided_human_labels",
        "tolerance_ms": tolerance_ms,
        "matching": "one_to_one_fight_attacker_hand",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _divide(tp, tp + fp),
        "recall": _divide(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "false_positives_per_minute": _divide(fp * 60000, active_duration_ms)
        if active_duration_ms
        else None,
        "count_mae_per_round_fighter": float(
            np.mean([abs(a - b) for a, b in round_counts.values()])
        )
        if round_counts
        else None,
        "peak_error_median_ms": float(
            np.median(
                [
                    abs(predictions[i]["peak_ms"] - truth[j]["peak_ms"])
                    for i, j in matches
                ]
            )
        )
        if matches
        else None,
        "unmatched_predictions": [
            event["event_id"]
            for i, event in enumerate(predictions)
            if i not in matched_predictions
        ],
        "missed_ground_truth": [
            event["event_id"] for i, event in enumerate(truth) if i not in matched_truth
        ],
        "attributes": attributes,
        "hand_matching": "independent one_to_one_fight_attacker_time_without_hand_constraint",
    }


def validate_dataset_manifest(manifest):
    fights = manifest.get("fights")
    if not isinstance(fights, list) or not fights:
        raise ValueError("Dataset manifest requires nonempty fights list")
    identifiers, hashes, athlete_splits = {}, {}, defaultdict(set)
    unknown_rights = []
    for fight in fights:
        fight_id, split = fight.get("fight_id"), fight.get("split")
        if not fight_id or split not in {"train", "development", "calibration", "test"}:
            raise ValueError("Each fight requires fight_id and explicit split")
        if fight_id in identifiers:
            raise ValueError(f"Fight {fight_id} occurs more than once; split leakage")
        identifiers[fight_id] = split
        digest = fight.get("source_sha256")
        if digest:
            if digest in hashes:
                raise ValueError(
                    "The same source video appears under multiple fight IDs"
                )
            hashes[digest] = fight_id
        for athlete in fight.get("athlete_ids", []):
            athlete_splits[athlete].add(split)
        if fight.get("rights_status", "unknown") != "approved":
            unknown_rights.append(fight_id)
    if any(len(splits) > 1 for splits in athlete_splits.values()):
        raise ValueError("Athlete overlap across dataset splits")
    return {
        "fight_count": len(fights),
        "splits": {
            split: list(identifiers.values()).count(split)
            for split in sorted(set(identifiers.values()))
        },
        "rights_unverified_fights": unknown_rights,
        "training_data_approved": not unknown_rights,
    }


def _bbox(value):
    return np.asarray(
        [value[key] for key in ("x1", "y1", "x2", "y2")]
        if isinstance(value, dict)
        else value,
        dtype=float,
    )


def _iou(first, second):
    a, b = _bbox(first), _bbox(second)
    if (
        a.shape != (4,)
        or b.shape != (4,)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Expected finite xyxy box")
    size = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(size))
    union = float(
        np.prod(np.maximum(0, a[2:] - a[:2]))
        + np.prod(np.maximum(0, b[2:] - b[:2]))
        - intersection
    )
    return intersection / union if union > 0 else 0


def evaluate_identity(
    predictions, truth, *, iou_threshold=0.5, timestamp_tolerance_ms=34
):
    frames = truth.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Identity evaluation requires explicit human-labeled frames")
    by_time = defaultdict(list)
    fight_times = defaultdict(set)
    for observation in predictions:
        fight_id, stamp = observation.get("fight_id", ""), observation["timestamp_ms"]
        by_time[(fight_id, stamp)].append(observation)
        fight_times[fight_id].add(stamp)
    assignments = correct = fighter_gt = pair_frames = verified_pairs = (
        other_assignments
    ) = switches = 0
    previous_roles = {}
    for frame in sorted(
        frames, key=lambda item: (item.get("fight_id", ""), item["timestamp_ms"])
    ):
        if frame.get("scene_state") != "ACTIVE_FIGHT":
            continue
        fight_id = frame.get("fight_id", "")
        times = np.asarray(sorted(fight_times[fight_id]))
        persons = [person for person in frame["persons"] if person.get("visible", True)]
        index = (
            int(np.argmin(np.abs(times - frame["timestamp_ms"])))
            if times.size
            else None
        )
        detected = (
            by_time[(fight_id, int(times[index]))]
            if index is not None
            and abs(times[index] - frame["timestamp_ms"]) <= timestamp_tolerance_ms
            else []
        )
        assigned = [
            person
            for person in detected
            if person.get("fighter_id") in {"fighter_a", "fighter_b"}
            and person.get("identity_state", "") not in {"UNKNOWN", "OTHER"}
        ]
        assignments += len(assigned)
        fighter_gt += sum(
            person["identity"] in {"fighter_a", "fighter_b"} for person in persons
        )
        gt_pair = {person["identity"] for person in persons} >= {
            "fighter_a",
            "fighter_b",
        }
        pair_frames += int(gt_pair)
        frame_correct = set()
        if assigned and persons:
            overlaps = np.asarray(
                [
                    [
                        _iou(a.get("detector_bbox") or a["bbox"], b["bbox"])
                        for b in persons
                    ]
                    for a in assigned
                ]
            )
            rows, columns = linear_sum_assignment(
                np.where(
                    overlaps >= iou_threshold,
                    1 - overlaps,
                    len(assigned) + len(persons) + 1,
                )
            )
            for row, column in zip(rows, columns):
                if overlaps[row, column] < iou_threshold:
                    continue
                actual, prediction = persons[int(column)], assigned[int(row)]
                if actual["identity"] == prediction["fighter_id"]:
                    correct += 1
                    frame_correct.add(actual["identity"])
                if actual["identity"] == "other":
                    other_assignments += 1
                person_id = (fight_id, actual["person_id"])
                if actual["identity"] in {"fighter_a", "fighter_b"}:
                    if (
                        person_id in previous_roles
                        and previous_roles[person_id] != prediction["fighter_id"]
                    ):
                        switches += 1
                    previous_roles[person_id] = prediction["fighter_id"]
        verified_pairs += int(gt_pair and frame_correct >= {"fighter_a", "fighter_b"})
    return {
        "identity_precision": _divide(correct, assignments),
        "identity_recall": _divide(correct, fighter_gt),
        "verified_pair_coverage": _divide(verified_pairs, pair_frames),
        "correct_assignments": correct,
        "predicted_assignments": assignments,
        "visible_gt_fighter_instances": fighter_gt,
        "visible_gt_pair_frames": pair_frames,
        "verified_pair_frames": verified_pairs,
        "other_person_assignments": other_assignments,
        "id_switches": switches,
        "coverage_denominator": "human-labeled ACTIVE_FIGHT frames with both fighters visible",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--tolerance-ms", type=int, default=300)
    parser.add_argument("--active-duration-ms", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    predictions, truth = (
        json.loads(path.read_text()) for path in (args.predictions, args.truth)
    )
    report = (
        evaluate_identity(predictions, truth)
        if args.identity
        else evaluate_events(
            predictions,
            truth,
            tolerance_ms=args.tolerance_ms,
            active_duration_ms=args.active_duration_ms,
        )
    )
    if args.manifest:
        report["dataset"] = validate_dataset_manifest(
            json.loads(args.manifest.read_text())
        )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
