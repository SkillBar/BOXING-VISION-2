from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from boxing_vision.contracts import (
    BBox,
    DisplayState,
    DisplayTrack,
    IdentityState,
    Keypoint,
    PoseObservation,
    RenderFrameContext,
)
from boxing_vision.display_tracking import (
    DisplayTrackSampler,
    build_display_tracks,
    display_track_from_dict,
)
from boxing_vision.pose import RawPose
from boxing_vision.render import FIGHTER_COLOR_BY_ID, FrameRenderer


def person(source=1, x=100, points=True):
    return RawPose(
        bbox=BBox(x, 60, x + 100, 340, 0.88),
        keypoints={
            "nose": Keypoint(x + 50, 82, 0.95),
            "left_eye": Keypoint(x + 45, 80, 0.95),
            "right_eye": Keypoint(x + 55, 80, 0.95),
            "left_shoulder": Keypoint(x + 30, 135, 0.95),
            "left_elbow": Keypoint(x + 15, 175, 0.95),
            "left_wrist": Keypoint(x + 20, 212, 0.95),
            "right_shoulder": Keypoint(x + 65, 135, 0.95),
            "right_elbow": Keypoint(x + 80, 165, 0.95),
            "right_wrist": Keypoint(x + 75, 210, 0.95),
            "left_hip": Keypoint(x + 40, 230, 0.95),
            "right_hip": Keypoint(x + 60, 230, 0.95),
        }
        if points
        else {},
        source_track_id=source,
        detector_confidence=0.88,
        pose_confidence=0.9 if points else 0.1,
    )


def observed(raw=None, time=0, fighter="fighter_a", segment="s1", shot=0):
    raw = raw or person()
    return PoseObservation(
        frame_index=time // 67,
        timestamp_ms=time,
        fighter_id=fighter,
        bbox=raw.bbox,
        keypoints=raw.keypoints,
        source_track_id=raw.source_track_id,
        detector_confidence=0.88,
        identity_confidence=0.92,
        identity_margin=0.30,
        shot_id=shot,
        segment_id=segment,
        physical_track_id=f"{shot}:{fighter}",
        identity_origin="immutable_gallery",
    )


def frame(time, poses=(), *, shot=0, scene="ACTIVE_FIGHT", cut=False, predictions=()):
    return SimpleNamespace(
        timestamp_ms=time,
        frame_index=time // 67,
        shot_id=shot,
        poses=list(poses),
        scene_state=scene,
        is_scene_cut=cut,
        tracker_predictions=list(predictions),
    )


def diagnostic(time, source=1, **values):
    return {"timestamp_ms": time, "shot_id": 0, "source_track_id": source, **values}


def display(time=0, **kwargs):
    raw = person()
    return DisplayTrack(
        timestamp_ms=time,
        evidence_timestamp_ms=time,
        bbox=raw.bbox,
        keypoints=raw.keypoints,
        fighter_id="fighter_a",
        identity_state=IdentityState.FIGHTER_A,
        source_track_id=1,
        segment_id="s1",
        physical_track_id="p1",
        **kwargs,
    )


def test_missing_pose_or_appearance_keeps_measured_box_neutral_not_all_people():
    raw = person()
    frames = [
        frame(0, [raw, person(2, 320)]),
        frame(67, [person(points=False), person(2, 320)]),
    ]
    analytical = [observed(raw)]
    before = [item.to_dict() for item in analytical]
    tracks = build_display_tracks(frames, analytical)
    measured = [item for item in tracks if item.display_state == "OBSERVED"]
    assert len(measured) == 2
    assert all(item.source_track_id == 1 for item in measured)
    assert measured[1].identity_state == "UNKNOWN"
    assert measured[1].keypoints == {}
    assert measured[1].bbox == frames[1].poses[0].bbox
    assert before == [item.to_dict() for item in analytical]
    assert not any(isinstance(item, PoseObservation) for item in tracks)


@pytest.mark.parametrize("gap", [200, 1000, 1500])
def test_prediction_is_bounded_has_no_joints_and_does_not_extend_identity(gap):
    raw = person()
    tracks = build_display_tracks([frame(0, [raw]), frame(gap)], [observed(raw)])
    last = tracks[-1]
    assert last.display_state == ("PREDICTED" if gap <= 1000 else "LOST")
    assert last.evidence_timestamp_ms == 0
    assert last.keypoints == {}
    if gap <= 1000:
        assert last.identity_origin == "legacy_motion_prediction"


def test_real_botsort_cmc_snapshot_is_used_without_repredicting():
    raw = person()
    snapshot = {
        "source_track_id": 1,
        "shot_id": 0,
        "timestamp_ms": 200,
        "evidence_timestamp_ms": 0,
        "bbox": BBox(117, 65, 217, 345).to_dict(),
    }
    tracks = build_display_tracks(
        [frame(0, [raw]), frame(200, predictions=[snapshot])], [observed(raw)]
    )
    assert tracks[-1].bbox.x1 == 117
    assert tracks[-1].identity_origin == "botsort_cmc_prediction"
    assert tracks[-1].keypoints == {}


@pytest.mark.parametrize(
    "after",
    [
        frame(200, shot=1, cut=True),
        frame(200, scene="BREAK"),
        frame(200, scene="REPLAY"),
        frame(200, cut=True),
    ],
)
def test_cut_and_nonfight_stop_forecast(after):
    tracks = build_display_tracks([frame(0, [person()]), after], [observed()])
    assert not [
        item
        for item in tracks
        if item.timestamp_ms == 200 and item.display_state != "LOST"
    ]


@pytest.mark.parametrize(
    "reason,state",
    [
        ("negative_gallery_closer", "OTHER"),
        ("user_rejected", "UNKNOWN"),
        ("geometry_jump", "UNKNOWN"),
    ],
)
def test_hard_rejection_cannot_later_resume_old_role_without_confirmation(
    reason, state
):
    frames = [
        frame(0, [person()]),
        frame(67, [person()]),
        frame(134, [person()]),
        frame(201),
    ]
    tracks = build_display_tracks(
        frames, [observed()], [diagnostic(67, reason=reason, identity_state=state)]
    )
    assert all(
        item.timestamp_ms == 0 or item.display_state == "LOST" for item in tracks
    )


def test_identity_conflict_keeps_measured_neutral_box_but_stops_prediction():
    tracks = build_display_tracks(
        [frame(0, [person()]), frame(67, [person()]), frame(134)],
        [observed()],
        [diagnostic(67, reason="identity_conflict", identity_state="UNKNOWN")],
    )
    disputed = next(item for item in tracks if item.timestamp_ms == 67)
    assert disputed.display_state == "OBSERVED"
    assert disputed.identity_state == "UNKNOWN"
    assert tracks[-1].display_state == "LOST"


def test_same_source_new_segment_keeps_continuous_neutral_person_and_reconfirms():
    frames = [frame(0, [person()]), frame(67, [person()]), frame(134, [person()])]
    diag = [diagnostic(67, segment_id="s2"), diagnostic(134, segment_id="s2")]
    tracks = build_display_tracks(
        frames,
        [observed(), observed(time=134, fighter="fighter_b", segment="s2")],
        diag,
    )
    disputed = [
        item
        for item in tracks
        if item.timestamp_ms == 67 and item.display_state == "OBSERVED"
    ]
    assert len(disputed) == 1
    assert disputed[0].identity_state == "UNKNOWN"
    assert tracks[-1].fighter_id == "fighter_b"
    assert tracks[-1].segment_id == "s2"


def test_unrelated_segment_geometry_jump_does_not_keep_old_eligibility():
    tracks = build_display_tracks(
        [frame(0, [person()]), frame(67, [person(x=450)])],
        [observed()],
        [diagnostic(67, segment_id="other-person")],
    )
    assert not [
        item
        for item in tracks
        if item.timestamp_ms == 67 and item.display_state == "OBSERVED"
    ]


def test_physical_graph_link_can_show_new_source_neutral_without_false_identity():
    tracks = build_display_tracks(
        [frame(0, [person()]), frame(67, [person(8)])],
        [observed()],
        [diagnostic(67, source=8, segment_id="s2", physical_track_id="0:fighter_a")],
    )
    visible = DisplayTrackSampler(tracks).sample(67)
    assert len(visible) == 1
    assert visible[0].source_track_id == 8
    assert visible[0].identity_state == "UNKNOWN"


def test_physical_only_overlap_ancestry_survives_short_missing_measurement_neutrally():
    # Real graph metadata: the same physical person spans an occlusion boundary,
    # without transferring A/B. A single missing detection must not erase that
    # ancestry after the UNKNOWN measurement stopped any coloured forecast.
    diagnostics = [
        diagnostic(
            time,
            segment_id="s2",
            physical_track_id="0:fighter_a",
            identity_state="UNKNOWN",
            segment_reason="occlusion_boundary",
            link_evidence={
                "kind": "physical_continuity",
                "identity_transferred": False,
                "predecessor_segment_id": "s1",
                "gap_ms": 67,
            },
        )
        for time in (67, 201)
    ]
    tracks = build_display_tracks(
        [
            frame(0, [person()]),
            frame(67, [person()]),
            frame(134),
            frame(201, [person()]),
            frame(268),
        ],
        [observed()],
        diagnostics,
    )
    sampled = DisplayTrackSampler(tracks).sample(201)
    assert len(sampled) == 1
    assert sampled[0].display_state == "OBSERVED"
    assert sampled[0].identity_state == "UNKNOWN"
    assert sampled[0].identity_confidence == 0
    assert not [
        item
        for item in tracks
        if item.timestamp_ms > 0 and item.display_state == "PREDICTED"
    ]
    renderer = FrameRenderer(hud_mode="compact")
    ops = []
    renderer.draw_tracking_layer(np.zeros((360, 640, 3), np.uint8), sampled, ops)
    assert any(item[1] == "Личность уточняется" for item in ops)
    assert not any(item[1] in {"A", "B"} for item in ops)


@pytest.mark.parametrize("barrier", ["OTHER", "CUT", "GAP"])
def test_physical_ancestry_cannot_bypass_other_cut_or_expiration(barrier):
    rows = [
        diagnostic(
            67,
            segment_id="s2",
            physical_track_id="0:fighter_a",
            identity_state="UNKNOWN",
        )
    ]
    frames = [frame(0, [person()]), frame(67, [person()]), frame(134)]
    return_time = 201
    if barrier == "OTHER":
        frames[-1] = frame(134, [person()])
        rows.append(
            diagnostic(
                134,
                segment_id="s2",
                physical_track_id="0:fighter_a",
                identity_state="OTHER",
            )
        )
    elif barrier == "CUT":
        frames[-1] = frame(134, cut=True)
    else:
        return_time = 1800
    frames.append(frame(return_time, [person()]))
    rows.append(
        diagnostic(
            return_time,
            segment_id="s2",
            physical_track_id="0:fighter_a",
            identity_state="UNKNOWN",
        )
    )
    tracks = build_display_tracks(frames, [observed()], rows)
    assert DisplayTrackSampler(tracks).sample(return_time) == []


def test_source_recovery_never_duplicates_old_prediction_for_same_fighter():
    frames = [frame(0, [person()]), frame(67), frame(134, [person(8)])]
    tracks = build_display_tracks(
        frames, [observed(), observed(person(8), time=134, segment="s2")]
    )
    sampled = DisplayTrackSampler(tracks).sample(134, RenderFrameContext(134))
    assert len(sampled) == 1
    assert sampled[0].source_track_id == 8


def test_uncertain_measured_identity_does_not_turn_into_colored_prediction():
    tracks = build_display_tracks(
        [frame(0, [person()]), frame(67, [person()]), frame(134)], [observed()]
    )
    assert tracks[-1].display_state == "LOST"
    assert tracks[-2].identity_state == "UNKNOWN"


def test_frame_exit_stops_forecast_instead_of_pin_to_border():
    prediction = {
        "source_track_id": 1,
        "shot_id": 0,
        "bbox": BBox(680, 60, 780, 340).to_dict(),
        "evidence_timestamp_ms": 0,
    }
    tracks = build_display_tracks(
        [frame(0, [person()]), frame(200, predictions=[prediction])],
        [observed()],
        frame_size=(640, 360),
    )
    assert tracks[-1].display_state == "LOST"


def test_sampler_interpolates_only_shared_measured_joints_and_never_across_conflict():
    first = display()
    second = replace(
        display(100),
        bbox=BBox(120, 60, 220, 340),
        keypoints={"left_wrist": Keypoint(160, 230, 0.9)},
    )
    sampled = DisplayTrackSampler([first, second]).sample(50)
    assert len(sampled) == 1
    assert sampled[0].bbox.x1 == 110
    assert set(sampled[0].keypoints) == {"left_wrist"}
    assert sampled[0].display_state == "INTERPOLATED"
    conflict = replace(second, identity_state="UNKNOWN")
    assert DisplayTrackSampler([first, conflict]).sample(50)[0].bbox.x1 == 100


def test_sampler_respects_lost_boundary_shot_and_one_second_prediction_limit():
    first = display()
    lost = replace(first, timestamp_ms=100, display_state="LOST")
    assert DisplayTrackSampler([first, lost]).sample(120) == []
    assert (
        DisplayTrackSampler([first]).sample(0, RenderFrameContext(0, shot_id=1)) == []
    )
    predicted = replace(
        first, timestamp_ms=1000, display_state="PREDICTED", keypoints={}
    )
    assert DisplayTrackSampler([predicted]).sample(1001) == []


def test_display_json_roundtrip_does_not_infer_identity_from_label():
    item = replace(display(), identity_state=IdentityState.UNKNOWN)
    restored = display_track_from_dict(item.to_dict())
    assert restored == item
    assert restored.identity_state == "UNKNOWN"


def test_full_box_head_arms_and_wrists_visible_without_analytical_observations():
    image = np.zeros((360, 640, 3), np.uint8)
    rendered = FrameRenderer(hud_mode="compact").draw(
        image, [], include_hud=False, display_tracks=[display()]
    )
    # Midpoint of vertical edge distinguishes a full box from corner brackets.
    assert rendered[195, 100, 2] > 150
    assert np.count_nonzero(rendered[70:110, 125:175]) > 50
    assert np.count_nonzero(rendered[202:220, 110:130]) > 10
    assert np.count_nonzero(rendered[201:220, 165:185]) > 10
    assert np.array_equal(image, np.zeros_like(image))


def test_unknown_render_is_neutral_and_prediction_has_no_invented_skeleton():
    image = np.zeros((360, 640, 3), np.uint8)
    unknown = replace(display(), identity_state="UNKNOWN", keypoints={})
    neutral = FrameRenderer(hud_mode="compact").draw(
        image, [], include_hud=False, display_tracks=[unknown]
    )
    color = np.asarray(FIGHTER_COLOR_BY_ID["fighter_a"])
    assert not np.any(np.all(neutral == color, axis=2))
    predicted = replace(
        display(), timestamp_ms=200, display_state=DisplayState.PREDICTED
    )
    renderer = FrameRenderer(hud_mode="compact")
    output = renderer.draw(image, [], include_hud=False, display_tracks=[predicted])
    assert np.count_nonzero(output[125:245, 110:190]) == 0


def test_missing_one_wrist_does_not_hide_other_hand_or_box():
    image = np.zeros((360, 640, 3), np.uint8)
    renderer = FrameRenderer(hud_mode="compact")
    first = display()
    renderer.draw(image, [], include_hud=False, display_tracks=[first], timestamp_ms=0)
    keys = dict(first.keypoints)
    keys.pop("left_wrist")
    second = replace(first, timestamp_ms=250, evidence_timestamp_ms=250, keypoints=keys)
    output = renderer.draw(
        image, [], include_hud=False, display_tracks=[second], timestamp_ms=250
    )
    assert np.count_nonzero(output[205:218, 169:181]) > 10
    assert output[195, 100, 2] > 150


def test_prediction_label_is_in_preview_and_event_description_export_only():
    image = np.zeros((360, 640, 3), np.uint8)
    track = replace(
        display(), timestamp_ms=200, display_state="PREDICTED", keypoints={}
    )
    preview = []
    renderer = FrameRenderer(hud_mode="compact")
    exported = renderer.draw(
        image,
        [],
        display_tracks=[track],
        timestamp_ms=200,
        tracking_frame_callback=preview.append,
        summary={"fighters": {"fighter_a": {"name": "Красный угол", "attempts": 4}}},
    )
    assert len(preview) == 1
    assert np.count_nonzero(preview[0][35:59, 100:230]) > 0
    assert not np.array_equal(exported, preview[0])
