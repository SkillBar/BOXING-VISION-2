from __future__ import annotations

import numpy as np

from boxing_vision.video import ReplayDetector, SceneSegment, SceneState


def _frames(count: int) -> list[np.ndarray]:
    generator = np.random.default_rng(20260904)
    return [
        generator.integers(0, 256, size=(90, 160, 3), dtype=np.uint8)
        for _ in range(count)
    ]


def test_one_dhash_match_cannot_mark_a_long_shot_as_replay() -> None:
    source = _frames(31)
    detector = ReplayDetector(min_sequence_ms=2_000)
    for index, frame in enumerate(source):
        assert detector.update(frame, index * 100, is_scene_cut=False) is False

    # Only the first frame repeats.  The following shot is unrelated and even
    # a very distant timestamp cannot turn that single hash into a replay.
    assert detector.update(source[0], 5_000, is_scene_cut=True) is False
    unrelated = _frames(25)
    for index, frame in enumerate(unrelated, start=1):
        assert detector.update(frame, 5_000 + index * 100, is_scene_cut=False) is False
    assert detector.update(unrelated[-1], 800_000, is_scene_cut=False) is False

    assert detector.is_replay is False
    assert detector.confirmed_start_ms is None


def test_replay_requires_two_second_time_consistent_sequence() -> None:
    source = _frames(31)
    detector = ReplayDetector(min_sequence_ms=2_000)
    for index, frame in enumerate(source):
        detector.update(frame, index * 100, is_scene_cut=False)

    signals = [
        detector.update(
            frame,
            5_000 + index * 100,
            is_scene_cut=index == 0,
        )
        for index, frame in enumerate(source[:24])
    ]

    assert signals[:20] == [False] * 20
    assert any(signals[20:])
    assert detector.is_replay is True
    assert detector.confirmed_start_ms == 5_000
    assert detector.confirmed_source_start_ms is not None


def test_replay_candidate_never_crosses_next_shot_boundary() -> None:
    source = _frames(31)
    detector = ReplayDetector(min_sequence_ms=2_000)
    for index, frame in enumerate(source):
        detector.update(frame, index * 100, is_scene_cut=False)

    detector.update(source[0], 5_000, is_scene_cut=True)
    for index, frame in enumerate(source[1:10], start=1):
        detector.update(frame, 5_000 + index * 100, is_scene_cut=False)
    detector.update(_frames(1)[0], 6_000, is_scene_cut=True)

    assert detector.shot_id == 2
    assert detector.is_replay is False
    assert detector.confirmed_start_ms is None


def test_scene_state_contract_is_string_serializable() -> None:
    segment = SceneSegment(
        shot_id=3,
        start_ms=10_000,
        end_ms=12_000,
        state=SceneState.ACTIVE_FIGHT,
        confidence=0.94,
    )

    assert segment.state == "ACTIVE_FIGHT"


def test_static_detailed_shots_do_not_false_confirm_as_replay() -> None:
    # A checkerboard has plenty of spatial detail, so this specifically tests
    # the temporal-diversity gate rather than the blank-frame shortcut.
    grid = (np.indices((90, 160)).sum(axis=0) % 2 * 255).astype(np.uint8)
    frame = np.repeat(grid[:, :, None], 3, axis=2)
    detector = ReplayDetector(min_sequence_ms=2_000)
    for index in range(31):
        detector.update(frame, index * 100, is_scene_cut=False)

    signals = [
        detector.update(frame, 5_000 + index * 100, is_scene_cut=index == 0)
        for index in range(31)
    ]

    assert signals == [False] * len(signals)
    assert detector.is_replay is False


def test_low_information_frames_are_not_indexed_or_confirmed() -> None:
    detector = ReplayDetector(history_size=32)
    black = np.zeros((90, 160, 3), dtype=np.uint8)
    for index in range(80):
        detector.update(black, index * 100, is_scene_cut=index in {0, 40})

    assert detector.is_replay is False
    assert len(detector._history_ids) == 0


def test_replay_hypotheses_and_history_are_strictly_bounded() -> None:
    source = _frames(80)
    detector = ReplayDetector(
        history_size=32,
        max_candidates=5,
        max_proposals_per_frame=3,
    )
    for index, frame in enumerate(source):
        detector.update(frame, index * 100, is_scene_cut=index in {0, 40})

    saw_candidate = False
    for index, frame in enumerate(source[48:60]):
        detector.update(frame, 10_000 + index * 100, is_scene_cut=index == 0)
        saw_candidate = saw_candidate or bool(detector._candidates)
        assert len(detector._candidates) <= 5
        assert len(detector._history_ids) <= 32
        assert sum(len(bucket) for bucket in detector._hash_bands.values()) <= 32 * 8
    assert saw_candidate
