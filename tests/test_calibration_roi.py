from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from boxing_vision.calibration import (
    confirm_enrollment,
    enrollment_support_keypoints,
    validate_ring_floor_roi,
)
from boxing_vision.contracts import BBox, Keypoint
from boxing_vision.pose import RawPose


def enrollment(scale: float = 1.0) -> dict:
    h, w = round(100 * scale), round(200 * scale)
    sample = {"image": np.zeros((h, w, 3), np.uint8), "time_s": 0,
              "boxes": [[value * scale for value in box] for box in ((10, 5, 80, 95), (110, 5, 180, 95))],
              "selection": {"fighter_a": 0, "fighter_b": 1},
              "support_keypoints": [
                  {"left_ankle": {"x": .15, "y": .90, "score": .95},
                   "right_ankle": {"x": .30, "y": .91, "score": .95}},
                  {"left_ankle": {"x": .65, "y": .86, "score": .95},
                   "right_ankle": {"x": .80, "y": .88, "score": .95}},
              ]}
    views = [dict(deepcopy(sample), time_s=time) for time in (0, 4, 8)]
    return {"views": views, "ring_points": [(0, .3), (1, .3), (1, .45), (0, .45)], "confirmed": False}


@pytest.mark.parametrize("scale", [.5, 1, 6])
def test_rope_strip_excluding_both_visible_fighters_is_rejected_at_any_scale(scale) -> None:
    state = enrollment(scale)
    with pytest.raises(ValueError, match="полоса канатов"):
        validate_ring_floor_roi(state)
    with pytest.raises(ValueError, match="места, где стоят оба бойца"):
        confirm_enrollment(state)
    assert state["confirmed"] is False


def test_perspective_floor_polygon_accepts_supports_inside_it() -> None:
    state = enrollment()
    state["ring_points"] = [(.09, .70), (.89, .75), (.97, .98), (.02, .98)]
    result = validate_ring_floor_roi(state)
    assert result == {"status": "checked", "checked_roles": ["fighter_a", "fighter_b"], "outside_roles": []}
    assert confirm_enrollment(state)[0]["confirmed"] is True


def test_soles_near_polygon_edge_are_not_rejected_by_ankle_to_floor_offset() -> None:
    state = enrollment()
    state["ring_points"] = [(0, .92), (1, .92), (1, .98), (0, .98)]
    assert validate_ring_floor_roi(state)["outside_roles"] == []


@pytest.mark.parametrize("evidence", [{}, {"left_ankle": {"x": .65, "y": .86, "score": .3}},
                                     {"left_ankle": {"x": .65, "y": .999, "score": .95}}])
def test_missing_low_confidence_or_clipped_ankles_do_not_become_bbox_bottoms(evidence) -> None:
    state = enrollment()
    state["views"][0]["support_keypoints"][1] = evidence
    result = validate_ring_floor_roi(state)
    assert result["status"] == "insufficient_evidence"
    assert result["checked_roles"] == ["fighter_a"]
    assert confirm_enrollment(state)[0]["confirmed"] is True


def test_one_fighter_outside_is_not_sufficient_to_reject_a_camera_angle() -> None:
    state = enrollment()
    state["ring_points"] = [(.55, .75), (.95, .75), (.95, .98), (.55, .98)]
    assert validate_ring_floor_roi(state)["outside_roles"] == ["fighter_a"]


def test_one_foot_inside_is_enough_when_the_other_is_raised_or_occluded() -> None:
    state = enrollment()
    state["ring_points"] = [(0, .84), (1, .84), (1, .98), (0, .98)]
    for support in state["views"][0]["support_keypoints"]:
        support["left_ankle"]["y"] = .7
    assert validate_ring_floor_roi(state)["outside_roles"] == []


def test_roi_validation_uses_first_view_and_selected_people_only() -> None:
    state = enrollment()
    state["ring_points"] = [(0, .8), (1, .8), (1, .98), (0, .98)]
    # Other camera views and a nonselected person cannot invalidate this ROI.
    for view in state["views"][1:]:
        for support in view["support_keypoints"]:
            support["left_ankle"]["y"] = support["right_ankle"]["y"] = .4
    state["views"][0]["boxes"].append([80, 5, 105, 95])
    state["views"][0]["support_keypoints"].append({"left_ankle": {"x": .45, "y": .5, "score": 1}})
    assert validate_ring_floor_roi(state)["outside_roles"] == []


def test_legacy_view_without_keypoints_is_not_falsely_certified_or_rejected() -> None:
    state = enrollment()
    state["views"][0].pop("support_keypoints")
    assert validate_ring_floor_roi(state)["status"] == "insufficient_evidence"
    assert confirm_enrollment(state)[0]["confirmed"] is True


def test_support_evidence_is_derived_from_existing_poses_without_clamping() -> None:
    poses = [RawPose(BBox(0, 0, 200, 100), {
        "left_ankle": Keypoint(30, 90, .9),
        "right_ankle": Keypoint(220, 101, .95),
        "nose": Keypoint(30, 10, .99),
    })]
    support = enrollment_support_keypoints(poses, 200, 100)
    assert support == [{"left_ankle": {"x": .15, "y": .9, "score": .9},
                        "right_ankle": {"x": 1.1, "y": 1.01, "score": .95}}]
    assert poses[0].keypoints["right_ankle"].x == 220
