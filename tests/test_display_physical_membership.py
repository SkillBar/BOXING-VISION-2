from dataclasses import replace

from boxing_vision.contracts import BBox, Keypoint, PoseObservation
from boxing_vision.display_tracking import build_display_tracks
from boxing_vision.pose import RawPose
from boxing_vision.tracking import TrackingFrame


def evidence():
    raw = RawPose(BBox(30, 20, 130, 250), {"left_wrist": Keypoint(60, 90, .9)},
                  confidence=.9, source_track_id=19, detector_confidence=.9)
    frames = [TrackingFrame(i, i * 100, 0, [raw]) for i in range(8)]
    observation = PoseObservation(7, 700, "fighter_b", raw.bbox, raw.keypoints,
        source_track_id=19, identity_confidence=.9, identity_margin=.3,
        physical_track_id="verified-chain", segment_id="after-overlap")
    diagnostics = [{"timestamp_ms": frame.timestamp_ms, "shot_id": 0, "source_track_id": 19,
                    "physical_track_id": "verified-chain", "segment_id": "before" if i < 4 else "after-overlap",
                    "identity_state": "UNKNOWN"} for i, frame in enumerate(frames)]
    return frames, observation, diagnostics


def test_future_physical_confirmation_only_shows_actual_neutral_measurements():
    frames, observation, diagnostics = evidence()
    tracks = build_display_tracks(frames, [observation], diagnostics)
    actual = [track for track in tracks if track.display_state == "OBSERVED"]
    assert len(actual) == 8
    assert all(track.identity_state == "UNKNOWN" and track.identity_confidence == 0 for track in actual[:-1])
    assert actual[0].identity_origin == "physical_membership_only"
    assert actual[0].keypoints == frames[0].poses[0].keypoints
    assert actual[-1].identity_state == "FIGHTER_B"


def test_future_identity_cannot_produce_prediction_at_an_empty_frame():
    frames, observation, diagnostics = evidence()
    frames[1] = replace(frames[1], poses=[])
    tracks = build_display_tracks(frames, [observation], diagnostics)
    assert not any(track.display_state == "PREDICTED" for track in tracks)
    assert not any(track.timestamp_ms == 100 and track.display_state == "OBSERVED" for track in tracks)


def test_other_barrier_blocks_future_eligibility_but_preserves_later_confirmed_box():
    frames, observation, diagnostics = evidence()
    diagnostics[3]["identity_state"] = "OTHER"
    tracks = build_display_tracks(frames, [observation], diagnostics)
    assert [track.timestamp_ms for track in tracks if track.display_state == "OBSERVED"] == [700]


def test_mixed_identity_chain_does_not_backfill_earlier_boxes():
    frames, observation, diagnostics = evidence()
    another = replace(observation, timestamp_ms=600, fighter_id="fighter_a", identity_state="FIGHTER_A")
    tracks = build_display_tracks(frames, [observation, another], diagnostics)
    assert not any(track.timestamp_ms < 600 for track in tracks)
