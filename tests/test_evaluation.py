from __future__ import annotations

from copy import deepcopy

import pytest

from tools.golden_eval import (
    evaluate_events,
    evaluate_identity,
    expected_calibration_error,
    validate_dataset_manifest,
)


def event(identifier, peak=1000, **kwargs):
    return dict(
        event_id=identifier,
        peak_ms=peak,
        attacker_id="fighter_a",
        hand="left",
        round=1,
        technique="hook",
        target="body",
        outcome="landed",
        **kwargs,
    )


def test_duplicate_predictions_are_false_positives_and_missing_predictions_are_false_negatives():
    truth = [event("gt1"), event("gt2", 3000)]
    result = evaluate_events([event("p1"), event("duplicate", 1010)], truth)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["f1"] == 0.5
    assert result["missed_ground_truth"] == ["gt2"]


def test_abstained_or_replay_predictions_cannot_remove_ground_truth_from_denominator():
    predictions = [
        event("p", proposal_status="abstained"),
        event("replay", 3000, is_replay=True),
    ]
    result = evaluate_events(predictions, [event("gt"), event("gt2", 3000)])
    assert result["tp"] == 0 and result["fn"] == 2 and result["recall"] == 0


def test_rejected_contract_enum_is_not_a_prediction():
    assert (
        evaluate_events([event("rejected", review_status="REJECTED")], [event("gt")])[
            "fn"
        ]
        == 1
    )


def test_matching_maximizes_cardinality_instead_of_greedy_nearest():
    result = evaluate_events(
        [event("p1", 1100), event("p2", 1300)],
        [event("g1", 900), event("g2", 1200)],
        tolerance_ms=200,
    )
    assert result["tp"] == 2


def test_unknown_type_counts_as_classification_error_and_hand_accuracy_is_not_tautological():
    predictions = [event("p1"), event("p2", 3000)]
    predictions[0]["technique"] = "unknown"
    predictions[1]["hand"] = "right"
    result = evaluate_events(predictions, [event("g1"), event("g2", 3000)])
    assert result["tp"] == 1 and result["fn"] == 1 and result["fp"] == 1
    assert result["attributes"]["technique"]["accuracy"] == 0
    assert result["attributes"]["hand"]["accuracy"] == 0.5


def test_matches_never_cross_fights():
    result = evaluate_events([event("p", fight_id="one")], [event("g", fight_id="two")])
    assert result["tp"] == 0 and result["fn"] == 1


def test_empty_ground_truth_does_not_create_accuracy_and_calibration_checks_ranges():
    assert evaluate_events([], [])["recall"] is None
    assert evaluate_events([], [])["attributes"]["outcome"]["ece"] is None
    assert expected_calibration_error([0.8, 0.8], [True, False]) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        expected_calibration_error([1.1], [True])


def test_dataset_disallows_fight_source_and_athlete_leakage_and_does_not_assume_rights():
    manifest = {
        "fights": [
            {"fight_id": "one", "split": "train", "source_sha256": "a"},
            {"fight_id": "two", "split": "test", "source_sha256": "b"},
        ]
    }
    assert validate_dataset_manifest(manifest)["training_data_approved"] is False
    for field, value in (("fight_id", "one"), ("source_sha256", "a")):
        duplicate = deepcopy(manifest)
        duplicate["fights"][1][field] = value
        with pytest.raises(ValueError):
            validate_dataset_manifest(duplicate)
    manifest["fights"][0]["athlete_ids"] = ["athlete1"]
    manifest["fights"][1]["athlete_ids"] = ["athlete1"]
    with pytest.raises(ValueError, match="Athlete"):
        validate_dataset_manifest(manifest)


def test_identity_coverage_uses_human_visible_frames_and_referee_is_false_assignment():
    people = [
        {"person_id": "a", "identity": "fighter_a", "bbox": [0, 0, 10, 20]},
        {"person_id": "b", "identity": "fighter_b", "bbox": [20, 0, 30, 20]},
        {"person_id": "ref", "identity": "other", "bbox": [40, 0, 50, 20]},
    ]
    truth = {
        "frames": [
            {
                "timestamp_ms": timestamp,
                "scene_state": "ACTIVE_FIGHT",
                "persons": people,
            }
            for timestamp in (0, 100)
        ]
    }
    predictions = [
        {"timestamp_ms": 0, "fighter_id": "fighter_a", "bbox": [0, 0, 10, 20]},
        {"timestamp_ms": 0, "fighter_id": "fighter_b", "bbox": [20, 0, 30, 20]},
        {"timestamp_ms": 100, "fighter_id": "fighter_a", "bbox": [40, 0, 50, 20]},
    ]
    report = evaluate_identity(predictions, truth)
    assert report["identity_precision"] == pytest.approx(2 / 3)
    assert report["identity_recall"] == 0.5
    assert report["verified_pair_coverage"] == 0.5
    assert report["other_person_assignments"] == 1


def test_identity_switch_is_measured_on_same_ground_truth_person():
    truth = {
        "frames": [
            {
                "timestamp_ms": timestamp,
                "scene_state": "ACTIVE_FIGHT",
                "persons": [
                    {
                        "person_id": "red",
                        "identity": "fighter_a",
                        "bbox": [0, 0, 10, 20],
                    }
                ],
            }
            for timestamp in (0, 100)
        ]
    }
    predictions = [
        {"timestamp_ms": timestamp, "fighter_id": role, "bbox": [0, 0, 10, 20]}
        for timestamp, role in ((0, "fighter_a"), (100, "fighter_b"))
    ]
    assert evaluate_identity(predictions, truth)["id_switches"] == 1


def test_identity_matching_never_crosses_fights_with_same_timestamp():
    truth = {
        "frames": [
            {
                "fight_id": "one",
                "timestamp_ms": 0,
                "scene_state": "ACTIVE_FIGHT",
                "persons": [
                    {"person_id": "a", "identity": "fighter_a", "bbox": [0, 0, 10, 20]}
                ],
            }
        ]
    }
    prediction = [
        {
            "fight_id": "two",
            "timestamp_ms": 0,
            "fighter_id": "fighter_a",
            "bbox": [0, 0, 10, 20],
        }
    ]
    assert evaluate_identity(prediction, truth)["identity_recall"] == 0
