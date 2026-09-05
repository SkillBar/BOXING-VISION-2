"""Temporal punch proposal primitives.

The current detector remains pose-heuristic based, but this module keeps the
temporal decision isolated from geometry and event labelling.  A learned model
can later implement :class:`TemporalProposalProvider` and consume the same
feature vectors without changing the public ``detect_punch_events`` contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

TEMPORAL_FEATURE_NAMES = (
    "extension",
    "wrist_x",
    "wrist_y",
    "speed",
    "outward_speed",
    "target_speed",
    "elbow_angle_normalized",
    "visibility",
)


@dataclass(frozen=True, slots=True)
class TemporalFeature:
    """One normalized pose-motion sample for one fighter hand."""

    timestamp_ms: int
    extension: float
    speed: float
    outward_speed: float
    target_speed: float
    elbow_angle: float
    visibility: float
    wrist_x: float = 0.0
    wrist_y: float = 0.0
    is_boundary: bool = False

    def vector(self) -> np.ndarray:
        """Return the stable feature order expected by a future temporal model."""

        return np.asarray(
            (
                self.extension,
                self.wrist_x,
                self.wrist_y,
                self.speed,
                self.outward_speed,
                self.target_speed,
                self.elbow_angle / 180.0,
                self.visibility,
            ),
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class TemporalProposal:
    """A complete extension -> peak -> return punch-shaped interval."""

    start_index: int
    peak_index: int
    end_index: int
    score: float
    return_ratio: float
    reach_gain: float
    excursion_gain: float
    target_progress: float
    outbound_samples: int


@dataclass(frozen=True, slots=True)
class HysteresisConfig:
    min_speed: float
    min_directional_speed: float
    min_reach_gain: float
    min_duration_ms: int
    max_duration_ms: int
    refractory_ms: int
    max_gap_ms: int = 350
    onset_speed_ratio: float = 0.72
    sustain_speed_ratio: float = 0.28
    return_speed_ratio: float = 0.55
    min_return_ratio: float = 0.30


@runtime_checkable
class TemporalProposalProvider(Protocol):
    """Replaceable boundary for a future learned temporal proposal model."""

    def propose(
        self,
        features: Sequence[TemporalFeature],
        config: HysteresisConfig,
    ) -> list[TemporalProposal]: ...


class HysteresisPunchProposer:
    """Conservative per-hand finite-state proposal generator.

    A burst is accepted only after a directional onset, a meaningful reach or
    target-directed excursion, and a visible return phase.  That final return
    requirement rejects camera/pose jumps and held-out guards that previously
    produced several local velocity peaks for one movement.
    """

    def propose(
        self,
        features: Sequence[TemporalFeature],
        config: HysteresisConfig,
    ) -> list[TemporalProposal]:
        if len(features) < 4:
            return []

        proposals: list[TemporalProposal] = []
        state = "idle"
        start_index = 0
        peak_index = 0
        baseline_extension = 0.0
        baseline_wrist = np.zeros(2, dtype=np.float64)
        peak_extension = 0.0
        peak_excursion = 0.0
        peak_salience = 0.0
        target_progress = 0.0
        outbound_samples = 0
        cooldown_until_ms = -1

        def reset() -> None:
            nonlocal state, target_progress, peak_salience, outbound_samples
            state = "idle"
            target_progress = 0.0
            peak_salience = 0.0
            outbound_samples = 0

        for index in range(1, len(features)):
            previous = features[index - 1]
            current = features[index]
            elapsed_ms = current.timestamp_ms - previous.timestamp_ms
            boundary = (
                current.is_boundary
                or elapsed_ms <= 0
                or elapsed_ms > config.max_gap_ms
            )
            if boundary:
                reset()
                cooldown_until_ms = current.timestamp_ms
                continue

            directional_speed = max(current.outward_speed, current.target_speed)
            onset = (
                current.timestamp_ms >= cooldown_until_ms
                and current.speed >= config.min_speed * config.onset_speed_ratio
                and directional_speed
                >= config.min_directional_speed * config.onset_speed_ratio
            )

            if state == "idle":
                if not onset:
                    continue
                start_index = max(0, index - 1)
                peak_index = index
                baseline_extension = features[start_index].extension
                baseline_wrist = np.asarray(
                    (features[start_index].wrist_x, features[start_index].wrist_y),
                    dtype=np.float64,
                )
                peak_extension = current.extension
                peak_excursion = float(
                    np.linalg.norm(
                        np.asarray((current.wrist_x, current.wrist_y)) - baseline_wrist
                    )
                )
                peak_salience = self._salience(current, baseline_extension)
                target_progress = max(0.0, current.target_speed) * elapsed_ms / 1000.0
                outbound_samples = 1
                state = "extending"
                continue

            duration_ms = current.timestamp_ms - features[start_index].timestamp_ms
            if duration_ms > config.max_duration_ms:
                reset()
                cooldown_until_ms = current.timestamp_ms + config.refractory_ms
                continue

            target_progress += max(0.0, current.target_speed) * elapsed_ms / 1000.0
            if (
                current.outward_speed
                >= config.min_directional_speed * config.sustain_speed_ratio
                or current.target_speed
                >= config.min_directional_speed * config.sustain_speed_ratio
                and current.extension >= peak_extension - 0.08
            ):
                outbound_samples += 1
            salience = self._salience(current, baseline_extension)
            current_excursion = float(
                np.linalg.norm(
                    np.asarray((current.wrist_x, current.wrist_y)) - baseline_wrist
                )
            )
            if (
                current_excursion > peak_excursion + 1e-6
                or salience > peak_salience * 1.08
                and current.outward_speed >= 0
            ):
                peak_index = index
                peak_excursion = max(peak_excursion, current_excursion)
                peak_salience = max(peak_salience, salience)
            peak_extension = max(peak_extension, current.extension)

            reach_gain = max(0.0, peak_extension - baseline_extension)
            return_amount = max(0.0, peak_excursion - current_excursion)
            return_ratio = return_amount / max(0.08, peak_excursion)
            retracting = (
                current.outward_speed
                <= -config.min_directional_speed * config.return_speed_ratio
                or return_ratio >= config.min_return_ratio * 0.55
            )
            if state == "extending" and retracting and index > peak_index:
                state = "returning"

            if state != "returning":
                # A burst that loses all velocity before reaching a meaningful
                # excursion was guard noise, not an incomplete punch.
                stalled = (
                    current.speed < config.min_speed * config.sustain_speed_ratio
                    and directional_speed
                    < config.min_directional_speed * config.sustain_speed_ratio
                )
                if stalled and duration_ms >= config.min_duration_ms:
                    reset()
                continue

            meaningful_excursion = (
                reach_gain >= config.min_reach_gain
                or peak_excursion >= config.min_reach_gain
                or target_progress >= config.min_reach_gain * 1.35
            )
            complete_return = return_ratio >= config.min_return_ratio
            if (
                duration_ms >= config.min_duration_ms
                and meaningful_excursion
                and outbound_samples >= 2
                and complete_return
                and index > peak_index
            ):
                phase_score = min(
                    1.0,
                    0.34
                    + 0.28 * min(1.0, reach_gain / max(config.min_reach_gain, 1e-6))
                    + 0.20 * min(1.0, return_ratio)
                    + 0.18 * min(1.0, peak_salience / max(config.min_speed, 1e-6)),
                )
                proposals.append(
                    TemporalProposal(
                        start_index=start_index,
                        peak_index=peak_index,
                        end_index=index,
                        score=phase_score,
                        return_ratio=min(1.0, return_ratio),
                        reach_gain=reach_gain,
                        excursion_gain=peak_excursion,
                        target_progress=target_progress,
                        outbound_samples=outbound_samples,
                    )
                )
                reset()
                cooldown_until_ms = current.timestamp_ms + config.refractory_ms
            elif complete_return and index > peak_index:
                # A one-frame spike completed its return but did not sustain an
                # extension phase. Discard it immediately rather than letting
                # it absorb a later real punch into the same proposal.
                reset()
                cooldown_until_ms = current.timestamp_ms

        return proposals

    @staticmethod
    def _salience(feature: TemporalFeature, baseline_extension: float) -> float:
        excursion = max(0.0, feature.extension - baseline_extension)
        directional = max(0.0, feature.outward_speed, feature.target_speed)
        return 0.48 * feature.speed + 0.34 * excursion + 0.18 * directional


def feature_matrix(features: Sequence[TemporalFeature]) -> np.ndarray:
    """Build a ``T x 6`` float32 matrix for model training or inference."""

    if not features:
        return np.empty((0, len(TEMPORAL_FEATURE_NAMES)), dtype=np.float32)
    return np.stack([feature.vector() for feature in features])


__all__ = [
    "TEMPORAL_FEATURE_NAMES",
    "HysteresisConfig",
    "HysteresisPunchProposer",
    "TemporalFeature",
    "TemporalProposal",
    "TemporalProposalProvider",
    "feature_matrix",
]
