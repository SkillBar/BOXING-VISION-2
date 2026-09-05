from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from boxing_vision.contracts import (
    BBox,
    IdentityState,
    Keypoint,
    PoseObservation,
    PunchEvent,
    SceneState,
)
from boxing_vision.model_registry import ACM_JOINTS, validate_acm_bundle
from boxing_vision.punch_models import (
    AcmPunchClassifier,
    PunchClassification,
    build_classification_window,
    classify_candidate_events,
    normalize_acm_pose,
)


def pose(fighter="fighter_a", timestamp=1000, **kwargs):
    coordinates = ((100, 100), (140, 100), (90, 140), (160, 140), (110, 110), (150, 110), (110, 180), (140, 180))
    keypoints = {name: Keypoint(float(x), float(y), .95) for name, (x, y) in zip(ACM_JOINTS, coordinates)}
    return PoseObservation(
        frame_index=round(timestamp * .03), timestamp_ms=timestamp, fighter_id=fighter,
        bbox=BBox(60, 60, 180, 300), keypoints=keypoints,
        identity_confidence=.95, identity_margin=.4, **kwargs,
    )


def sequence(*, dense=True):
    times = np.rint(np.arange(25) * 1000 / 30 + 600).astype(int) if dense else range(500, 1501, 100)
    return [pose(fighter, int(timestamp)) for timestamp in times for fighter in ("fighter_a", "fighter_b")]


def event(technique="hook"):
    return PunchEvent(
        event_id="test", round=1, start_ms=700, peak_ms=1000, end_ms=1200,
        attacker_id="fighter_a", defender_id="fighter_b", hand="left",
        technique=technique, target="body", outcome="unclear", confidence=.61,
        impact_proxy_0_100=50,
    )


class FakeClassifier:
    def __init__(self, label="hook", score=.92, margin=.86):
        self.label, self.score, self.margin = label, score, margin

    def predict(self, features):
        assert features.shape == (1, 25, 16)
        return PunchClassification(self.label, self.score, self.margin, (.02, .02, .92, .04))


def test_source_exact_normalization_and_visibility_boundary():
    points = pose().keypoints
    result = normalize_acm_pose(points).reshape(8, 2)
    np.testing.assert_allclose(result[:2], [[-.5, 0], [.5, 0]])
    np.testing.assert_allclose(result[2], [-.75, 1])
    points["left_elbow"] = replace(points["left_elbow"], score=.5)
    assert not normalize_acm_pose(points).reshape(8, 2)[2].any()
    points["left_shoulder"] = replace(points["left_shoulder"], score=.5)
    assert normalize_acm_pose(points) is not None
    points["left_shoulder"] = replace(points["left_shoulder"], score=.499)
    assert normalize_acm_pose(points) is None


def test_windows_distinguish_actual_30fps_and_interpolated_input():
    dense = build_classification_window(event(), sequence())
    sparse = build_classification_window(event(), sequence(dense=False))
    assert dense.features.shape == sparse.features.shape == (1, 25, 16)
    assert dense.input_kind == "native_30fps"
    assert sparse.input_kind == "resampled"


@pytest.mark.parametrize("change", [
    {"identity_state": IdentityState.UNKNOWN}, {"scene_state": SceneState.BREAK},
    {"is_scene_cut": True}, {"shot_id": 2}, {"identity_margin": .05},
])
def test_missing_or_uncertain_defender_rejects_whole_window(change):
    observations = sequence()
    observations[25] = replace(observations[25], **change)
    assert build_classification_window(event(), observations).features is None


def test_sparse_classifier_is_diagnostic_only():
    original = event("unknown")
    result = classify_candidate_events([original], sequence(dense=False), FakeClassifier())[0]
    assert result.technique == "unknown"
    assert result.evidence["classification_abstention"] == "sparse_input_diagnostic_only"
    assert original.evidence == {}


def test_model_window_cannot_span_motion_tracklet_change():
    observations = [replace(obs, source_track_id=1) for obs in sequence()]
    observations[24] = replace(observations[24], source_track_id=9)
    result = build_classification_window(event(), observations)
    assert result.features is None and result.reason == "tracklet_boundary"


def test_confident_disagreement_abstains_without_changing_hand_or_outcome():
    result = classify_candidate_events([event("uppercut")], sequence(), FakeClassifier())[0]
    assert result.technique == "unknown"
    assert result.hand == "left" and result.outcome == "unclear"
    assert result.review_status == "needs_review"
    assert result.confidence == .61
    assert result.classification_confidence is None


def test_ready_model_can_propose_unknown_technique_for_review():
    result = classify_candidate_events([event("unknown")], sequence(), FakeClassifier())[0]
    assert result.technique == "hook"
    assert result.review_status == "needs_review"
    assert result.evidence["classification_consensus"] == "model_proposal_requires_review"


def test_model_jab_cross_never_overrides_anatomical_hand_or_known_stance():
    result = classify_candidate_events(
        [event("unknown")], sequence(), FakeClassifier("cross"), stances={"fighter_a": "orthodox"},
    )[0]
    assert result.hand == "left" and result.technique == "jab"


def test_weak_scores_do_not_overwrite_baseline_and_cancellation_is_propagated():
    result = classify_candidate_events([event()], sequence(), FakeClassifier("uppercut", .55, .1))[0]
    assert result.technique == "hook"
    with pytest.raises(InterruptedError):
        classify_candidate_events([event()], sequence(), FakeClassifier(), cancelled=lambda: True)


def test_local_registered_onnx_and_hash_integrity(tmp_path):
    bundle = Path(__file__).resolve().parents[1] / "models" / "acm40960-lstm-v1"
    if not (bundle / "manifest.json").exists():
        pytest.skip("Optional local ready-model bundle not installed")
    classifier = AcmPunchClassifier(bundle)
    prediction = classifier.predict(build_classification_window(event(), sequence()).features)
    assert prediction.label in {"jab", "cross", "hook", "uppercut"}
    assert np.isclose(sum(prediction.probabilities), 1)
    assert 0 <= prediction.score <= 1
    with pytest.raises(ValueError, match="finite"):
        classifier.predict(np.full((1, 25, 16), np.nan))
    for filename in ("manifest.json", "LICENSE", "model.onnx"):
        shutil.copyfile(bundle / filename, tmp_path / filename)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["onnx_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA256"):
        validate_acm_bundle(tmp_path)
