from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from boxing_vision.contracts import BBox, PoseObservation, ReviewStatus, SceneState
from boxing_vision.identity import (
    AppearancePart,
    IdentityGallery,
    IdentityState,
    appearance_distance,
    is_confirmed_identity,
    part_appearance_distance,
)


def test_confirmed_identity_is_shared_and_keeps_legacy_margin_compatibility():
    observation = PoseObservation(
        0,
        0,
        "fighter_a",
        BBox(0, 0, 20, 40),
        {},
        identity_confidence=0.55,
        identity_margin=0.12,
    )
    assert is_confirmed_identity(observation)
    assert is_confirmed_identity(replace(observation, identity_margin=None))
    assert not is_confirmed_identity(observation, minimum_margin=0.2)
    for changes in (
        {"identity_confidence": 0.549},
        {"identity_confidence": float("nan")},
        {"identity_confidence": float("inf")},
        {"identity_margin": float("nan")},
        {"identity_state": IdentityState.FIGHTER_B},
        {"identity_state": IdentityState.UNKNOWN},
        {"scene_state": SceneState.BREAK},
        {"review_status": ReviewStatus.REJECTED},
        {"review_status": ReviewStatus.NEEDS_REVIEW},
    ):
        assert not is_confirmed_identity(replace(observation, **changes)), changes


def test_core_gallery_is_write_once_and_returned_as_immutable_samples() -> None:
    gallery = IdentityGallery()
    gallery.set_core_once(IdentityState.FIGHTER_A, [(4.0, 0.0, 0.0)])

    assert gallery.core[IdentityState.FIGHTER_A] == ((1.0, 0.0, 0.0),)
    with pytest.raises(RuntimeError):
        gallery.set_core_once(IdentityState.FIGHTER_A, [(0.0, 1.0, 0.0)])


def test_negative_gallery_rejects_referee_even_when_fighter_distance_is_valid() -> None:
    gallery = IdentityGallery(
        core={
            IdentityState.FIGHTER_A: [(1.0, 0.0, 0.0)],
            IdentityState.FIGHTER_B: [(0.0, 1.0, 0.0)],
        },
        negative=[(0.0, 0.0, 1.0)],
    )

    match = gallery.match((0.0, 0.0, 1.0))

    assert match.state == IdentityState.OTHER
    assert match.accepted is False
    assert match.reason == "negative_gallery_closer"


def test_adaptive_gallery_requires_every_contamination_guard() -> None:
    gallery = IdentityGallery(
        core={
            IdentityState.FIGHTER_A: [(1.0, 0.0, 0.0)],
            IdentityState.FIGHTER_B: [(0.0, 1.0, 0.0)],
        }
    )
    descriptor = (0.95, 0.05, 0.0)

    assert (
        gallery.update_adaptive(
            IdentityState.FIGHTER_A,
            descriptor,
            confidence=0.89,
            margin=0.30,
            stable_frames=8,
            overlap_or_clinch=False,
            active_fight=True,
        )
        is False
    )
    assert (
        gallery.update_adaptive(
            IdentityState.FIGHTER_A,
            descriptor,
            confidence=0.95,
            margin=0.19,
            stable_frames=8,
            overlap_or_clinch=False,
            active_fight=True,
        )
        is False
    )
    assert (
        gallery.update_adaptive(
            IdentityState.FIGHTER_A,
            descriptor,
            confidence=0.95,
            margin=0.30,
            stable_frames=4,
            overlap_or_clinch=False,
            active_fight=True,
        )
        is False
    )
    assert (
        gallery.update_adaptive(
            IdentityState.FIGHTER_A,
            descriptor,
            confidence=0.95,
            margin=0.30,
            stable_frames=8,
            overlap_or_clinch=True,
            active_fight=True,
        )
        is False
    )
    assert (
        gallery.update_adaptive(
            IdentityState.FIGHTER_A,
            descriptor,
            confidence=0.95,
            margin=0.30,
            stable_frames=8,
            overlap_or_clinch=False,
            active_fight=False,
        )
        is False
    )
    assert gallery.adaptive_samples(IdentityState.FIGHTER_A) == ()

    assert (
        gallery.update_adaptive(
            IdentityState.FIGHTER_A,
            descriptor,
            confidence=0.95,
            margin=0.30,
            stable_frames=8,
            overlap_or_clinch=False,
            active_fight=True,
        )
        is True
    )
    assert len(gallery.adaptive_samples(IdentityState.FIGHTER_A)) == 1


def test_adaptive_gallery_honors_configured_confidence_and_margin_thresholds() -> None:
    gallery = IdentityGallery(
        core={
            IdentityState.FIGHTER_A: [(1.0, 0.0, 0.0)],
            IdentityState.FIGHTER_B: [(0.0, 1.0, 0.0)],
        },
        adaptive_confidence_min=0.82,
        adaptive_margin_min=0.27,
    )
    descriptor = (0.95, 0.05, 0.0)

    assert (
        gallery.update_adaptive(
            IdentityState.FIGHTER_A,
            descriptor,
            confidence=0.81,
            margin=0.30,
            stable_frames=5,
            overlap_or_clinch=False,
            active_fight=True,
        )
        is False
    )
    assert (
        gallery.update_adaptive(
            IdentityState.FIGHTER_A,
            descriptor,
            confidence=0.90,
            margin=0.26,
            stable_frames=5,
            overlap_or_clinch=False,
            active_fight=True,
        )
        is False
    )
    assert (
        gallery.update_adaptive(
            IdentityState.FIGHTER_A,
            descriptor,
            confidence=0.82,
            margin=0.27,
            stable_frames=5,
            overlap_or_clinch=False,
            active_fight=True,
        )
        is True
    )


def test_appearance_distance_is_safe_for_missing_or_zero_features() -> None:
    assert appearance_distance(None, (1.0, 0.0)) == 1.0
    assert appearance_distance((0.0, 0.0), (0.0, 0.0)) == 1.0
    assert appearance_distance((1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)


def test_gloves_support_identity_but_cannot_establish_it_alone() -> None:
    gloves = {
        name: AppearancePart((1.0, 0.0), 1.0) for name in ("left_glove", "right_glove")
    }
    assert part_appearance_distance(gloves, gloves) is None
    with_torso = {**gloves, "torso": AppearancePart((0.0, 1.0), 0.95)}
    assert part_appearance_distance(with_torso, with_torso) == pytest.approx(0.0)


def test_foreground_parts_overrule_polluted_bbox_but_not_when_occluded() -> None:
    gallery = IdentityGallery(
        core={"FIGHTER_A": [(1.0, 0.0)], "FIGHTER_B": [(0.0, 1.0)]}
    )
    red = {"torso": AppearancePart((1.0, 0.0), 1.0)}
    blue = {"torso": AppearancePart((0.0, 1.0), 1.0)}
    gallery.set_core_parts_once("FIGHTER_A", [red])
    gallery.set_core_parts_once("FIGHTER_B", [blue])
    assert gallery.match((0.0, 1.0), red).state == IdentityState.FIGHTER_A
    assert not gallery.match((1.0, 0.0), {}).accepted
    assert not gallery.match(
        (1.0, 0.0), {"torso": AppearancePart((1.0, 0.0), 0.2)}
    ).accepted
    with pytest.raises(RuntimeError):
        gallery.set_core_parts_once("FIGHTER_A", [blue])


def test_identity_profile_roundtrip_preserves_evidence_not_future_adaptation() -> None:
    gallery = IdentityGallery(
        core={"FIGHTER_A": [(1.0, 0.0)], "FIGHTER_B": [(0.0, 1.0)]},
        max_distance=0.3,
        min_margin=0.2,
    )
    gallery.set_core_parts_once(
        "FIGHTER_A", [{"torso": AppearancePart((1.0, 0.0), 0.9)}]
    )
    gallery.set_core_parts_once(
        "FIGHTER_B", [{"torso": AppearancePart((0.0, 1.0), 0.9)}]
    )
    gallery.add_negative_parts({"torso": AppearancePart((0.5, 0.5), 0.9)})
    gallery.update_adaptive(
        "FIGHTER_A",
        (0.99, 0.01),
        confidence=0.99,
        margin=0.99,
        stable_frames=10,
        overlap_or_clinch=False,
        active_fight=True,
    )
    restored = IdentityGallery.from_export(gallery.export())
    assert restored.core == gallery.core
    assert restored.max_distance == 0.3
    assert restored.min_margin == 0.2
    assert restored.export()["core_parts"] == gallery.export()["core_parts"]
    assert not restored.adaptive_samples("FIGHTER_A")


def test_grey_fabric_undefined_hue_is_not_identity_evidence():
    first, second = np.zeros(112), np.zeros(112)
    first[0], second[7 * 4] = 0.65, 0.65
    first[48 + 4 * 8 + 4] = second[48 + 4 * 8 + 4] = 0.35
    assert part_appearance_distance(
        {"torso": AppearancePart(tuple(first), 1.0)},
        {"torso": AppearancePart(tuple(second), 1.0)},
    ) == pytest.approx(0.0)


def test_saturated_opponent_colors_remain_distinguishable():
    red, blue = np.zeros(112), np.zeros(112)
    red[3], blue[7 * 4 + 3] = 0.65, 0.65
    red[48 + 4 * 8 + 4] = blue[48 + 4 * 8 + 4] = 0.35
    assert (
        part_appearance_distance(
            {"torso": AppearancePart(tuple(red), 1.0)},
            {"torso": AppearancePart(tuple(blue), 1.0)},
        )
        > 0.35
    )


def test_partial_shorts_compare_continuity_but_do_not_establish_identity():
    parts = {"shorts": AppearancePart((1.0, 0.0), 0.422)}
    assert part_appearance_distance(parts, parts) is None
    assert part_appearance_distance(parts, parts, allow_partial=True) == 0
    gallery = IdentityGallery(
        core={"FIGHTER_A": [(1.0, 0.0)], "FIGHTER_B": [(0.0, 1.0)]}
    )
    gallery.set_core_parts_once(
        "FIGHTER_A", [{"shorts": AppearancePart((1.0, 0.0), 1.0)}]
    )
    gallery.set_core_parts_once(
        "FIGHTER_B", [{"shorts": AppearancePart((0.0, 1.0), 1.0)}]
    )
    assert gallery.match((1.0, 0.0), parts).reason == "appearance_partial"


def test_missing_alternative_parts_do_not_erase_real_score_or_invent_margin():
    gallery = IdentityGallery(
        core={"FIGHTER_A": [(1.0, 0.0)], "FIGHTER_B": [(0.0, 1.0)]}
    )
    torso = {"torso": AppearancePart((1.0, 0.0), 1.0)}
    gallery.set_core_parts_once("FIGHTER_A", [torso])
    gallery.set_core_parts_once(
        "FIGHTER_B", [{"shorts": AppearancePart((0.0, 1.0), 1.0)}]
    )
    distances, _ = gallery.distances((1.0, 0.0), torso)
    assert distances[IdentityState.FIGHTER_A] == 0
    assert distances[IdentityState.FIGHTER_B] == 1
    match = gallery.match((1.0, 0.0), torso)
    assert (
        not match.accepted
        and match.margin == 0
        and match.reason == "appearance_partial"
    )


def test_mirror_like_negative_does_not_poison_immutable_fighter_gallery():
    gallery = IdentityGallery(
        core={"FIGHTER_A": [(1.0, 0.0, 0.0)], "FIGHTER_B": [(0.0, 1.0, 0.0)]}
    )
    red = {"torso": AppearancePart((1.0, 0.0, 0.0), 1.0)}
    blue = {"torso": AppearancePart((0.0, 1.0, 0.0), 1.0)}
    gallery.set_core_parts_once("FIGHTER_A", [red])
    gallery.set_core_parts_once("FIGHTER_B", [blue])
    gallery.add_negative((1.0, 0.0, 0.0))
    gallery.add_negative_parts(red)
    assert not gallery.negative and not gallery.export()["negative_parts"]
    assert gallery.match((1.0, 0.0, 0.0), red).state == IdentityState.FIGHTER_A
    gallery.add_negative((0.0, 0.0, 1.0))
    assert len(gallery.negative) == 1


def test_adaptive_parts_are_used_but_cannot_bootstrap_from_inherited_labels():
    gallery = IdentityGallery(
        core={"FIGHTER_A": [(1.0, 0.0, 0.0)], "FIGHTER_B": [(0.0, 0.0, 1.0)]}
    )

    def parts(values):
        return {
            "torso": AppearancePart(values, 1.0),
            "shorts": AppearancePart(values, 1.0),
        }

    gallery.set_core_parts_once("FIGHTER_A", [parts((1.0, 0.0, 0.0))])
    gallery.set_core_parts_once("FIGHTER_B", [parts((0.0, 0.0, 1.0))])
    descriptor = (0.99, 0.01, 0.0)
    conditions = {
        "confidence": 0.99,
        "margin": 0.99,
        "stable_frames": 8,
        "overlap_or_clinch": False,
        "active_fight": True,
        "parts": parts(descriptor),
    }
    core = gallery.export()["core_parts"]
    assert not gallery.update_adaptive(
        "FIGHTER_A", descriptor, evidence_origin="continuous", **conditions
    )
    assert gallery.update_adaptive("FIGHTER_A", descriptor, **conditions)
    assert (
        gallery.distances(descriptor, parts(descriptor))[0][IdentityState.FIGHTER_A]
        == 0
    )
    assert (
        gallery.distances(descriptor, parts(descriptor), include_adaptive=False)[0][
            IdentityState.FIGHTER_A
        ]
        > 0
    )
    assert gallery.export()["core_parts"] == core
    assert not IdentityGallery.from_export(gallery.export()).export()["adaptive_parts"]
