from __future__ import annotations

import builtins
from dataclasses import replace

import numpy as np
import pytest

from boxing_vision.contracts import BBox, IdentityState, Keypoint, ReviewStatus
from boxing_vision.identity import AppearancePart
from boxing_vision.pose import RawPose, TwoFighterTracker
from boxing_vision.tracking import (
    OfflineIdentityDecoder,
    ShotLocalBoTSORT,
    TrackingBackendUnavailable,
    TrackingFrame,
)

RED, BLUE, OTHER = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def person(x, appearance=RED, *, source=1, confidence=0.95):
    keys = {
        name: Keypoint(x + index % 2 * 20, 40 + index * 10, 0.96)
        for index, name in enumerate(
            (
                "nose",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_ankle",
                "right_ankle",
            )
        )
    }
    return RawPose(
        BBox(x - 30, 20, x + 40, 220, confidence),
        keys,
        confidence,
        appearance,
        source_track_id=source,
        detector_confidence=confidence,
        pose_confidence=0.96,
    )


def enrolled():
    tracker = TwoFighterTracker()
    tracker.enroll(
        {"fighter_a": [person(100)] * 3, "fighter_b": [person(300, BLUE)] * 3},
        [person(210, OTHER)],
    )
    return tracker


def frame(index, poses, scene="ACTIVE_FIGHT", shot=0):
    return TrackingFrame(index, index * 100, shot, poses, scene, index == 0)


def test_identity_confidence_is_independent_of_mean_pose_and_diagnostics_match():
    decoder = OfflineIdentityDecoder(enrolled())
    frames = [
        frame(
            index,
            [
                replace(person(100, RED, source=1), pose_confidence=0.2),
                replace(person(300, BLUE, source=2), pose_confidence=0.2),
            ],
        )
        for index in range(8)
    ]
    observations = decoder.decode(frames)
    assert observations
    by_source = {(obs.timestamp_ms, obs.source_track_id): obs for obs in observations}
    for diagnostic in decoder.diagnostics:
        if diagnostic.get("selected_fighter_id") is None:
            assert diagnostic["confidence"] == 0
            continue
        obs = by_source[(diagnostic["timestamp_ms"], diagnostic["source_track_id"])]
        assert diagnostic["confidence"] == pytest.approx(obs.identity_confidence)
        assert diagnostic["confidence"] == pytest.approx(1.0)
        assert diagnostic["pose_confidence"] == pytest.approx(0.2)
        assert diagnostic["identity_state"] == str(obs.identity_state)
        assert diagnostic["gallery_similarity"] == pytest.approx(1.0)


def test_real_botsort_low_score_continuity_and_shot_reset():
    tracker = ShotLocalBoTSORT(enable_cmc=True)
    # Textured stationary background also exercises sparse optical-flow CMC.
    image = np.random.default_rng(7).integers(0, 256, (240, 400, 3), dtype=np.uint8)
    first = tracker.update([person(100)], image, 0, 0)
    assert first[0].source_track_id is None
    for index in range(1, 5):
        tracked = tracker.update([person(100 + index)], image, index * 67, 0)
    source = tracked[0].source_track_id
    assert source is not None
    weak = tracker.update(
        [person(105, confidence=0.2), person(290, BLUE, confidence=0.2)], image, 335, 0
    )
    assert weak[0].source_track_id == source
    assert weak[1].source_track_id is None
    assert weak[0].detector_confidence == 0.2
    assert len(tracker.last_diagnostics) == 2
    after_cut = tracker.update([person(105, confidence=0.2)], image, 400, 1)
    assert after_cut[0].source_track_id is None
    assert tracker.tracklets[0]["tracklet_id"] == f"shot-0-track-{source}"


def test_real_botsort_never_substitutes_pose_confidence_for_missing_detector_score():
    tracker = ShotLocalBoTSORT(enable_cmc=False)
    invalid = replace(person(100), detector_confidence=None)
    with pytest.raises(ValueError, match="detector_confidence"):
        tracker.update([invalid], np.zeros((240, 400, 3), np.uint8), 0, 0)


def test_missing_botsort_dependency_is_explicit_not_a_fake_fallback(monkeypatch):
    original_import = builtins.__import__

    def import_without_trackers(name, *args, **kwargs):
        if name == "trackers":
            raise ImportError("deliberate missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_trackers)
    with pytest.raises(TrackingBackendUnavailable, match="trackers==2.6.0"):
        ShotLocalBoTSORT()


def test_tracking_frame_cache_roundtrip():
    original = frame(4, [person(100), person(200, OTHER)], scene="REPLAY", shot=2)
    original.tracker_predictions = [
        {
            "timestamp_ms": 400,
            "evidence_timestamp_ms": 300,
            "age_ms": 100,
            "source_track_id": 1,
            "shot_id": 2,
            "bbox": person(100).bbox.to_dict(),
        }
    ]
    assert TrackingFrame.from_dict(original.to_dict()) == original


def test_real_prediction_snapshot_is_bounded_read_only_and_resets_after_cut():
    tracker = ShotLocalBoTSORT(enable_cmc=False)
    image = np.zeros((240, 400, 3), np.uint8)
    for index in range(4):
        poses = tracker.update([person(100 + index)], image, index * 100, 0)
    source = poses[0].source_track_id
    tracker.update([], image, 400, 0)
    first = tracker.predictions
    assert len(first) == 1 and first[0]["source_track_id"] == source
    assert first[0]["evidence_timestamp_ms"] == 300 and first[0]["age_ms"] == 100
    assert tracker.predictions == first  # no additional Kalman predict on read
    tracker.update([], image, 1400, 0)
    assert not tracker.predictions
    tracker.update([], image, 1500, 1)
    assert not tracker.predictions


def test_offline_decoder_keeps_b_when_a_missing_and_referee_large():
    decoder = OfflineIdentityDecoder(enrolled())
    frames = [
        frame(
            i,
            [
                person(100, source=1),
                person(300, BLUE, source=2),
                person(200, OTHER, source=3),
            ],
        )
        for i in range(4)
    ]
    frames += [
        frame(i, [person(300, BLUE, source=2), person(140, OTHER, source=3)])
        for i in range(4, 7)
    ]
    decoded = decoder.decode(frames)
    assert not [obs for obs in decoded if obs.source_track_id == 3]
    assert {obs.fighter_id for obs in decoded if obs.timestamp_ms >= 400} == {
        "fighter_b"
    }
    assert {row["source_track_id"] for row in decoder.diagnostics} == {1, 2, 3}
    assert any(item["identity_state"] == "OTHER" for item in decoder.tracklets)


def test_votes_are_source_specific_not_combined_by_role():
    decoder = OfflineIdentityDecoder(enrolled())
    frames = [
        frame(i, [person(100, source=source)])
        for i, source in enumerate([1, 2, 3, 3, 3])
    ]
    decoded = decoder.decode(frames)
    assert [obs.timestamp_ms for obs in decoded] == [400]


def test_cut_reacquires_on_three_votes_independent_of_screen_side():
    decoder = OfflineIdentityDecoder(enrolled())
    decoded = decoder.decode(
        [
            frame(i, [person(300, source=7), person(100, BLUE, source=8)], shot=4)
            for i in range(5)
        ]
    )
    assert min(obs.timestamp_ms for obs in decoded) == 200
    assert all(
        obs.bbox.center[0] > 250 for obs in decoded if obs.fighter_id == "fighter_a"
    )
    assert all(
        obs.bbox.center[0] < 150 for obs in decoded if obs.fighter_id == "fighter_b"
    )


@pytest.mark.parametrize("scene", ["BREAK", "REPLAY", "NON_FIGHT", "UNCERTAIN"])
def test_non_fight_never_produces_fighter_observations(scene):
    decoder = OfflineIdentityDecoder(enrolled())
    assert (
        decoder.decode(
            [
                frame(i, [person(100), person(300, BLUE, source=2)], scene)
                for i in range(5)
            ]
        )
        == []
    )


def test_source_identity_conflict_splits_locally_preserving_safe_prefix_and_suffix():
    decoder = OfflineIdentityDecoder(enrolled())
    frames = [
        frame(i, [person(100, RED if i < 4 else BLUE, source=5)]) for i in range(8)
    ]
    observed = decoder.decode(frames)
    assert [(o.timestamp_ms, o.fighter_id) for o in observed] == [
        (200, "fighter_a"),
        (300, "fighter_a"),
        (600, "fighter_b"),
        (700, "fighter_b"),
    ]
    assert len({o.segment_id for o in observed}) == 2
    assert decoder.tracklets[1]["segment_reason"] == "identity_conflict_boundary"
    assert not any(
        row["reason"] == "source_identity_conflict" for row in decoder.diagnostics
    )


def test_review_override_replays_cache_without_detector_and_is_user_confirmed():
    tracker = enrolled()
    tracker.set_identity_overrides({"shot-0-track-5": "FIGHTER_A"})
    decoder = OfflineIdentityDecoder(tracker)
    decoded = decoder.decode(
        [frame(i, [person(100, OTHER, source=5)]) for i in range(5)]
    )
    assert len(decoded) == 3
    assert all(
        obs.identity_state == IdentityState.FIGHTER_A
        and obs.review_status == ReviewStatus.USER_CONFIRMED
        for obs in decoded
    )


def test_decoder_rejects_cross_shot_batch():
    with pytest.raises(ValueError, match="один shot"):
        OfflineIdentityDecoder(enrolled()).decode([frame(0, []), frame(1, [], shot=1)])


def clothed(x, descriptor, source=1):
    pose = person(x, descriptor, source=source)
    pose.appearance_parts = {
        key: AppearancePart(descriptor, 1.0) for key in ("torso", "shorts")
    }
    return pose


def clothing_tracker():
    tracker = TwoFighterTracker()
    tracker.enroll(
        {"fighter_a": [clothed(100, RED)] * 3, "fighter_b": [clothed(300, OTHER)] * 3}
    )
    return tracker


def test_tracklet_pools_enrolled_evidence_across_smooth_viewpoint_change():
    tracker = clothing_tracker()
    original_core = tracker.export_identity_profile()["core_parts"]
    descriptors = [RED] * 3 + [
        (0.9, 0.1, 0.0),
        (0.8, 0.2, 0.0),
        (0.65, 0.35, 0.0),
        (0.6, 0.4, 0.0),
    ]
    frames = [
        frame(i, [clothed(100 + i, descriptor)])
        for i, descriptor in enumerate(descriptors)
    ]
    assert not tracker.score_candidate(frames[-1].poses[0]).accepted
    decoder = OfflineIdentityDecoder(tracker)
    observed = decoder.decode(frames)
    assert [item.timestamp_ms for item in observed] == [200, 300, 400, 500, 600]
    assert decoder.diagnostics[-1]["reason"] == "observed_continuity"
    assert tracker.gallery.max_distance == 0.35 and tracker.gallery.min_margin == 0.12
    assert tracker.export_identity_profile()["core_parts"] == original_core


def test_weak_tracklet_cannot_establish_identity_without_three_strong_votes():
    tracker = clothing_tracker()
    frames = [frame(i, [clothed(100, (0.6, 0.4, 0.0))]) for i in range(8)]
    assert OfflineIdentityDecoder(tracker).decode(frames) == []


def test_break_is_a_barrier_but_missing_parts_on_observed_trajectory_are_not():
    tracker = clothing_tracker()
    frames = [frame(i, [clothed(100, RED)]) for i in range(3)]
    frames.append(frame(3, [clothed(100, RED)], scene="BREAK"))
    frames.extend(frame(i, [clothed(100, (0.6, 0.4, 0.0))]) for i in range(4, 8))
    observed = OfflineIdentityDecoder(tracker).decode(frames)
    assert [item.timestamp_ms for item in observed] == [200]
    hidden = clothed(100, RED)
    hidden.appearance_parts = {}
    frames = [frame(i, [clothed(100, RED)]) for i in range(3)] + [frame(3, [hidden])]
    observed = OfflineIdentityDecoder(clothing_tracker()).decode(frames)
    assert [item.timestamp_ms for item in observed] == [200, 300]
    assert observed[-1].identity_origin == "continuous"


def test_present_weak_frames_use_three_of_five_not_three_consecutive_votes():
    tracker = clothing_tracker()
    descriptors = [RED, (0.6, 0.4, 0.0), RED, (0.6, 0.4, 0.0), RED]
    frames = [frame(i, [clothed(100, value)]) for i, value in enumerate(descriptors)]
    observed = OfflineIdentityDecoder(tracker, evidence_radius_ms=0).decode(frames)
    assert [item.timestamp_ms for item in observed] == [400]


def test_interleaved_contender_cannot_preserve_votes_of_absent_reused_source():
    weak = (0.6, 0.4, 0)
    frames = [
        frame(0, [clothed(100, RED, source=1)]),
        frame(1, [clothed(100, RED, source=1)]),
        frame(2, [clothed(100, weak, source=1), clothed(300, RED, source=2)]),
        frame(3, [clothed(300, RED, source=2)]),
        frame(4, [clothed(800, RED, source=1), clothed(300, weak, source=2)]),
    ]
    decoder = OfflineIdentityDecoder(clothing_tracker(), evidence_radius_ms=0)
    assert decoder.decode(frames) == []


def test_default_evidence_radius_is_uniform_and_hard_capped():
    assert OfflineIdentityDecoder(clothing_tracker()).evidence_radius_ms == 1500
    assert OfflineIdentityDecoder.version == "track-graph-v3"
    for radius in [-1, 1501]:
        with pytest.raises(ValueError, match="1500"):
            OfflineIdentityDecoder(clothing_tracker(), evidence_radius_ms=radius)


@pytest.mark.parametrize("barrier", ["absent", "other", "break", "cut"])
def test_local_pool_cannot_skip_an_intermediate_hard_barrier(barrier):
    tracker = clothing_tracker()
    tracker.gallery.add_negative(BLUE)
    tracker.gallery.add_negative_parts(
        {"torso": AppearancePart(BLUE, 1.0), "shorts": AppearancePart(BLUE, 1.0)}
    )
    frames = [frame(i, [clothed(100, RED)]) for i in range(3)]
    hidden = clothed(100, RED)
    if barrier == "hidden":
        hidden.appearance_parts = {}
    elif barrier == "gloves_only":
        hidden.appearance_parts = {"left_glove": AppearancePart(RED, 1.0)}
    elif barrier == "other":
        hidden = clothed(100, BLUE)
    bridge = frame(
        3,
        [] if barrier == "absent" else [hidden],
        scene="BREAK" if barrier == "break" else "ACTIVE_FIGHT",
    )
    if barrier == "cut":
        bridge.is_scene_cut = True
    frames.append(bridge)
    frames += [frame(i, [clothed(100, (0.6, 0.4, 0.0))]) for i in range(4, 8)]
    observed = OfflineIdentityDecoder(tracker).decode(frames)
    assert [item.timestamp_ms for item in observed] == [200]


def test_same_shot_cut_resets_motion_and_requires_three_post_cut_votes():
    frames = [frame(i, [clothed(100, RED)]) for i in range(7)]
    frames[3].is_scene_cut = True
    observed = OfflineIdentityDecoder(clothing_tracker()).decode(frames)
    assert [item.timestamp_ms for item in observed] == [200, 500, 600]


def test_pooled_evidence_has_a_real_time_limit_despite_smooth_continuity():
    frames = [frame(i, [clothed(100, RED)]) for i in range(3)]
    frames += [
        frame(3, [clothed(100, (0.9, 0.1, 0.0))]),
        frame(4, [clothed(100, (0.8, 0.2, 0.0))]),
    ]
    frames += [frame(i, [clothed(100, (0.65, 0.35, 0.0))]) for i in range(5, 19)]
    decoder = OfflineIdentityDecoder(clothing_tracker(), evidence_radius_ms=1000)
    pooled = decoder._pooled_matches(frames)
    # All three strong samples must lie inside the evidence window, not merely
    # the most recent strong sample followed by an unlimited graph chain.
    assert "1" in pooled[12]
    assert not any(pooled[index] for index in range(13, len(frames)))


def test_isolated_opposite_match_does_not_disable_later_local_evidence(monkeypatch):
    from boxing_vision.identity import IdentityMatch

    tracker = clothing_tracker()
    descriptors = [RED] * 7 + [
        (0.9, 0.1, 0),
        (0.8, 0.2, 0),
        (0.65, 0.35, 0),
        (0.6, 0.4, 0),
    ]
    frames = [frame(i, [clothed(100, value)]) for i, value in enumerate(descriptors)]
    original = tracker.score_candidate

    def isolated_appearance_error(pose, **kwargs):
        if pose is frames[3].poses[0]:
            return IdentityMatch(
                IdentityState.FIGHTER_B, 0.1, 0.5, 1.0, True, "gallery_match"
            )
        return original(pose, **kwargs)

    monkeypatch.setattr(tracker, "score_candidate", isolated_appearance_error)
    decoder = OfflineIdentityDecoder(tracker)
    observed = decoder.decode(frames)
    assert observed and {item.fighter_id for item in observed} == {"fighter_a"}
    assert decoder.diagnostics[-1]["reason"] == "observed_continuity"
    assert not decoder.tracklets[0]["identity_conflict"]


def test_pooled_identity_never_overrides_candidate_closer_to_opponent():
    tracker = clothing_tracker()
    frames = [frame(i, [clothed(100, RED)]) for i in range(3)]
    # A gradual path keeps adjacent distances small, but the end is closer to B.
    descriptors = [(0.6, 0.4, 0), (0.3, 0.4, 0.3), (0.15, 0.4, 0.45)]
    frames += [
        frame(i + 3, [clothed(100, value)]) for i, value in enumerate(descriptors)
    ]
    decoder = OfflineIdentityDecoder(tracker)
    assert not decoder._pooled_matches(frames)[-1]


def test_local_pool_does_not_follow_reused_source_through_large_bbox_jump():
    frames = [frame(i, [clothed(100, RED)]) for i in range(3)]
    frames += [frame(i, [clothed(800, (0.6, 0.4, 0.0))]) for i in range(3, 7)]
    decoder = OfflineIdentityDecoder(clothing_tracker())
    assert [item.timestamp_ms for item in decoder.decode(frames)] == [200]


def test_explicit_unknown_override_stays_unknown():
    tracker = clothing_tracker()
    tracker.set_identity_overrides({"shot-0-track-1": "UNKNOWN"})
    decoder = OfflineIdentityDecoder(tracker)
    frames = [frame(i, [clothed(100, RED)]) for i in range(8)]
    assert not any(decoder._pooled_matches(frames))
    assert decoder.decode(frames) == []


def test_continuous_detected_box_retains_identity_with_missing_parts_not_missing_person():
    frames = [frame(i, [clothed(100 + i, RED)]) for i in range(4)]
    for i in range(4, 28):
        pose = clothed(100 + i, RED)
        pose.appearance_parts = {}
        frames.append(frame(i, [pose]))
    frames.append(frame(28, []))
    decoder = OfflineIdentityDecoder(clothing_tracker())
    observed = decoder.decode(frames)
    assert observed[-1].timestamp_ms == 2700
    assert observed[-1].identity_origin == "continuous"
    assert not any(o.timestamp_ms == 2800 for o in observed)


def test_partial_parts_cannot_invent_identity_without_confirmed_ancestry():
    frames = []
    for i in range(10):
        pose = clothed(100, RED)
        pose.appearance_parts = {"shorts": AppearancePart(RED, 0.422)}
        frames.append(frame(i, [pose]))
    assert OfflineIdentityDecoder(clothing_tracker()).decode(frames) == []


@pytest.mark.parametrize("gap_ms,linked", [(800, True), (1600, False)])
def test_new_source_stitch_has_bounded_physical_link_not_inherited_vote_counter(
    gap_ms, linked
):
    frames = [frame(i, [clothed(100, RED, source=1)]) for i in range(4)]
    end = frames[-1].timestamp_ms
    for i in range(4):
        frames.append(
            TrackingFrame(
                4 + i, end + gap_ms + i * 100, 0, [clothed(110, RED, source=8)]
            )
        )
    decoder = OfflineIdentityDecoder(clothing_tracker())
    observed = decoder.decode(frames)
    before = [o for o in observed if o.source_track_id == 1]
    after = [o for o in observed if o.source_track_id == 8]
    assert after[0].timestamp_ms == end + gap_ms + 200
    assert (before[0].physical_track_id == after[0].physical_track_id) is linked


def test_segment_override_is_deterministic_local_and_user_confirmed():
    frames = [
        frame(i, [person(100, RED if i < 4 else BLUE, source=5)]) for i in range(8)
    ]
    initial = OfflineIdentityDecoder(enrolled())
    initial.decode(frames)
    ids = [item["segment_id"] for item in initial.tracklets]
    decoder = OfflineIdentityDecoder(
        enrolled(), segment_identity_overrides={ids[0]: "UNKNOWN"}
    )
    observed = decoder.decode(frames)
    assert [item["segment_id"] for item in decoder.tracklets] == ids
    assert {(o.timestamp_ms, o.fighter_id) for o in observed} == {
        (600, "fighter_b"),
        (700, "fighter_b"),
    }
    overridden = OfflineIdentityDecoder(
        enrolled(), segment_identity_overrides={ids[0]: "FIGHTER_B"}
    )
    observed = overridden.decode(frames)
    assert all(
        o.review_status == ReviewStatus.USER_CONFIRMED
        for o in observed
        if o.segment_id == ids[0]
    )
    assert all(
        o.identity_origin != "user_confirmed"
        for o in observed
        if o.segment_id == ids[1]
    )
    with pytest.raises(ValueError, match="segment_id"):
        OfflineIdentityDecoder(
            enrolled(),
            segment_identity_overrides={"shot-0-track-5-segment-9999": "FIGHTER_A"},
        ).decode(frames)


def test_visible_other_is_never_required_fighter_review():
    decoder = OfflineIdentityDecoder(enrolled())
    decoder.decode([frame(i, [person(100, OTHER)]) for i in range(8)])
    assert decoder.tracklets and not any(
        item["required_review"] for item in decoder.tracklets
    )


def test_single_missing_detection_does_not_split_physical_segment_or_create_pose():
    frames = [frame(i, [] if i == 4 else [clothed(100, RED)]) for i in range(10)]
    decoder = OfflineIdentityDecoder(clothing_tracker())
    observed = decoder.decode(frames)
    assert len(decoder.tracklets) == 1
    assert not any(o.timestamp_ms == 400 for o in observed)
    assert min(o.timestamp_ms for o in observed if o.timestamp_ms > 400) == 700


def test_occlusion_boundary_keeps_physical_ancestry_without_inventing_identity():
    frames = [frame(i, [clothed(100, RED)]) for i in range(4)]
    for i in range(4, 8):
        hidden = clothed(100, RED)
        hidden.appearance_parts = {}
        # A second person overlaps, so appearance-free A/B continuity is unsafe.
        frames.append(frame(i, [hidden, clothed(110, OTHER, source=2)]))
    decoder = OfflineIdentityDecoder(clothing_tracker())
    observed = decoder.decode(frames)
    assert not any(o.source_track_id == 1 and o.timestamp_ms >= 400 for o in observed)
    rows = [row for row in decoder.diagnostics if row["source_track_id"] == 1]
    assert len({row["segment_id"] for row in rows}) == 2
    assert len({row["physical_track_id"] for row in rows}) == 1
    assert rows[-1]["link_evidence"]["identity_transferred"] is False


def test_one_appearance_negative_does_not_permanently_cut_a_good_trajectory(
    monkeypatch,
):
    from boxing_vision.identity import IdentityMatch

    tracker = clothing_tracker()
    frames = [frame(i, [clothed(100, RED)]) for i in range(10)]
    original = tracker.score_candidate

    def outlier(pose, **kwargs):
        if pose is frames[4].poses[0]:
            return IdentityMatch(
                IdentityState.OTHER, 0.3, 0.3, 0.0, False, "negative_gallery_closer"
            )
        return original(pose, **kwargs)

    monkeypatch.setattr(tracker, "score_candidate", outlier)
    decoder = OfflineIdentityDecoder(tracker)
    observed = decoder.decode(frames)
    assert observed[-1].timestamp_ms == 900
    assert len(decoder.tracklets) == 1
    assert not any(row["identity_state"] == "OTHER" for row in decoder.diagnostics)


def test_legacy_adaptation_cannot_reuse_pooled_identity_when_opponent_is_closer():
    tracker = TwoFighterTracker(
        core_appearances={
            IdentityState.FIGHTER_A: [RED] * 3,
            IdentityState.FIGHTER_B: [OTHER] * 3,
        }
    )
    descriptors = [RED] * 3 + [
        (0.85, 0.10, 0.05),
        (0.7, 0.17, 0.13),
        (0.60, 0.15, 0.25),
        (0.52, 0.12, 0.36),
        (0.46, 0.10, 0.44),
    ]
    frames = [
        frame(
            index,
            [
                clothed(100, descriptor, source=1),
                clothed(300, (0, 0.015, 0.985), source=2),
            ],
        )
        for index, descriptor in enumerate(descriptors)
    ]
    decoder = OfflineIdentityDecoder(tracker)
    observed = decoder.decode(frames)
    final = frames[-1].poses[0]
    distances, _ = tracker.gallery.distances(final.appearance, final.appearance_parts)
    assert distances[IdentityState.FIGHTER_B] < distances[IdentityState.FIGHTER_A]
    assert not any(
        item.source_track_id == 1 and item.timestamp_ms == 700 for item in observed
    )


def test_appearance_link_cannot_skip_intervening_other_to_older_ancestor(monkeypatch):
    """An occlusion ending later must not reopen a pre-OTHER identity anchor."""
    from boxing_vision.identity import IdentityMatch

    tracker = clothing_tracker()
    frames = [frame(i, [clothed(100, RED)]) for i in range(5)]
    for index in range(5, 13):
        weak = clothed(100, RED)
        weak.appearance_parts = {"shorts": AppearancePart(RED, 0.422)}
        overlapping = [clothed(110, OTHER, source=2)] if index < 9 else []
        frames.append(frame(index, [weak, *overlapping]))
    original = tracker.score_candidate

    def hard_other_barrier(pose, **kwargs):
        if pose is frames[4].poses[0]:
            return IdentityMatch(
                IdentityState.OTHER, 0.1, 0.5, 0.0, False, "outside_ring"
            )
        return original(pose, **kwargs)

    monkeypatch.setattr(tracker, "score_candidate", hard_other_barrier)
    decoder = OfflineIdentityDecoder(tracker)
    observed = decoder.decode(frames)
    assert {o.timestamp_ms for o in observed if o.source_track_id == 1} == {200, 300}
    later = [
        row
        for row in decoder.diagnostics
        if row["source_track_id"] == 1 and row["timestamp_ms"] >= 900
    ]
    assert later
    assert all(
        (row.get("link_evidence") or {}).get("predecessor_segment_id")
        != "shot-0-track-1-segment-0"
        for row in later
    )


def test_offline_anchor_confirms_continuous_partial_track_not_only_1500ms_window():
    tracker = clothing_tracker()
    core_before = tracker.export_identity_profile()["core_parts"]
    frames = []
    for index in range(30):
        pose = clothed(100, RED)
        if index < 26:
            pose.appearance_parts = {"shorts": AppearancePart(RED, 0.422)}
        frames.append(frame(index, [pose]))
    decoder = OfflineIdentityDecoder(tracker)
    observed = decoder.decode(frames)
    earlier = [o for o in observed if o.timestamp_ms < 2600]
    assert earlier and min(o.timestamp_ms for o in earlier) == 200
    assert all(o.identity_origin == "offline_confirmed" for o in earlier)
    assert all(o.fighter_id == "fighter_a" for o in observed)
    assert tracker.export_identity_profile()["core_parts"] == core_before
    diagnostics = [row for row in decoder.diagnostics if row["timestamp_ms"] == 200]
    assert diagnostics[0]["evidence_timestamp_ms"] == 2800


@pytest.mark.parametrize("barrier", ["other", "overlap", "cut", "missing_parts"])
def test_offline_future_anchor_does_not_cross_identity_or_visibility_barrier(
    monkeypatch, barrier
):
    from boxing_vision.identity import IdentityMatch

    tracker = clothing_tracker()
    frames = []
    for index in range(30):
        pose = clothed(100, RED)
        if index < 26:
            pose.appearance_parts = {"shorts": AppearancePart(RED, 0.422)}
        if index == 8 and barrier == "missing_parts":
            pose.appearance_parts = {}
        extra = (
            [clothed(110, OTHER, source=2)]
            if index == 8 and barrier == "overlap"
            else []
        )
        item = frame(index, [pose, *extra])
        item.is_scene_cut = item.is_scene_cut or (index == 8 and barrier == "cut")
        frames.append(item)
    if barrier == "other":
        original = tracker.score_candidate

        def score(pose, **kwargs):
            if pose is frames[8].poses[0]:
                return IdentityMatch(
                    IdentityState.OTHER, 0.1, 0.5, 0, False, "outside_ring"
                )
            return original(pose, **kwargs)

        monkeypatch.setattr(tracker, "score_candidate", score)
    decoder = OfflineIdentityDecoder(tracker)
    observed = decoder.decode(frames)
    assert not any(o.timestamp_ms < 800 for o in observed)
    assert any(o.timestamp_ms >= 2600 for o in observed)


def test_sparse_source_votes_cannot_create_active_or_future_identity_anchor():
    frames = []
    for index in range(12):
        poses = []
        if index in {0, 3, 6} or index >= 7:
            pose = clothed(100, RED)
            if index >= 7:
                pose.appearance_parts = {"shorts": AppearancePart(RED, 0.422)}
            poses.append(pose)
        frames.append(TrackingFrame(index, index * 50, 0, poses))
    decoder = OfflineIdentityDecoder(clothing_tracker())
    assert not decoder.decode(frames)
