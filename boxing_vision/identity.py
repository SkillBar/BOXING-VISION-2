"""Conservative, shot-local identity primitives for the two-fighter pipeline.

The module deliberately separates immutable enrollment evidence from the
adaptive gallery.  A bad motion-track assignment must never be able to rewrite
the evidence that says who fighter A and fighter B are.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .contracts import IdentityState, PoseObservation, ReviewStatus, SceneState


def is_confirmed_identity(
    observation: PoseObservation,
    *,
    minimum_confidence: float = 0.55,
    minimum_margin: float = 0.12,
) -> bool:
    """One eligibility rule for preflight, coverage, and visible overlays.

    Identity evidence is independent of detector and pose confidence. A poor
    wrist/ankle estimate cannot erase an otherwise verified person; the punch
    pipeline separately checks visible joints. An absent margin remains
    supported for legacy observation caches.
    """
    expected = {
        "fighter_a": IdentityState.FIGHTER_A,
        "fighter_b": IdentityState.FIGHTER_B,
    }.get(observation.fighter_id)
    confidence = float(observation.identity_confidence or 0.0)
    margin = observation.identity_margin
    return bool(
        expected is not None
        and observation.identity_state == expected
        and math.isfinite(confidence)
        and minimum_confidence <= confidence <= 1.0
        and (
            margin is None
            or (math.isfinite(float(margin)) and float(margin) >= minimum_margin)
        )
        and observation.scene_state == SceneState.ACTIVE_FIGHT
        and observation.review_status
        not in {ReviewStatus.NEEDS_REVIEW, ReviewStatus.REJECTED}
    )


@dataclass(frozen=True, slots=True)
class AppearancePart:
    """A foreground-only colour feature and its visible, uncontaminated fraction."""

    histogram: tuple[float, ...]
    reliability: float


PART_WEIGHTS = {
    "torso": 0.40,
    "shorts": 0.35,
    "left_glove": 0.125,
    "right_glove": 0.125,
    "headgear": 0.20,
    "waistband": 0.15,
}


@lru_cache(maxsize=4096)
def _stable_part_descriptor(histogram: tuple[float, ...]) -> tuple[float, ...]:
    """Ignore undefined hue of grey fabric and soften Lab quantization edges.

    Hue of low-saturation pixels changes wildly under compression/lighting.
    It must not turn the same black shirt into a different person. This fixed
    representation normalization applies identically to enrollment and cache.
    """
    if len(histogram) != 112:
        return histogram
    array = np.asarray(histogram, dtype=np.float32)
    hs = array[:48].reshape(12, 4).copy()
    hs[0, 0] = hs[:, 0].sum()
    hs[1:, 0] = 0.0
    ab = array[48:].reshape(8, 8).copy()
    original_mass = float(ab.sum())
    side = math.exp(-1 / (2 * 0.6**2))
    kernel = np.asarray([side, 1.0, side], dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(ab, ((1, 1), (0, 0)), mode="edge")
    ab = sum(kernel[i] * padded[i : i + 8] for i in range(3))
    padded = np.pad(ab, ((0, 0), (1, 1)), mode="edge")
    ab = sum(kernel[i] * padded[:, i : i + 8] for i in range(3))
    if ab.sum() > 0:
        ab *= original_mass / ab.sum()
    return tuple(float(v) for v in np.concatenate((hs.ravel(), ab.ravel())))


def part_appearance_distance(
    first: Mapping[str, AppearancePart],
    second: Mapping[str, AppearancePart],
    *,
    allow_partial: bool = False,
) -> float | None:
    """Compare visible matching parts; gloves alone cannot establish identity."""

    weighted = 0.0
    support = 0.0
    body_support = False
    for name, weight in PART_WEIGHTS.items():
        a, b = first.get(name), second.get(name)
        if a is None or b is None:
            continue
        reliability = min(a.reliability, b.reliability)
        if reliability < 0.25:
            continue
        contribution = weight * reliability
        body_support |= name in {"torso", "shorts", "headgear", "waistband"}
        weighted += contribution * appearance_distance(
            _stable_part_descriptor(a.histogram), _stable_part_descriptor(b.histogram)
        )
        support += contribution
    # A visible waistband/shorts fragment can compare an already anchored
    # trajectory, but never acquires a new identity by itself.
    minimum_support = 0.08 if allow_partial else 0.25
    return weighted / support if support >= minimum_support and body_support else None


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    """A gallery comparison result; low distance and high margin are safer."""

    state: IdentityState
    distance: float
    margin: float
    negative_distance: float
    accepted: bool
    reason: str


def appearance_distance(
    first: Sequence[float] | None,
    second: Sequence[float] | None,
) -> float:
    """Bhattacharyya/Hellinger distance for normalized histogram features."""

    if first is None or second is None or len(first) != len(second):
        return 1.0
    first_array = np.asarray(first, dtype=np.float32)
    second_array = np.asarray(second, dtype=np.float32)
    if first_array.size == 0 or not np.all(np.isfinite(first_array)):
        return 1.0
    if not np.all(np.isfinite(second_array)):
        return 1.0
    first_total = float(np.maximum(first_array, 0.0).sum())
    second_total = float(np.maximum(second_array, 0.0).sum())
    if first_total <= 1e-9 or second_total <= 1e-9:
        return 1.0
    first_array = np.maximum(first_array, 0.0) / first_total
    second_array = np.maximum(second_array, 0.0) / second_total
    coefficient = float(np.sum(np.sqrt(first_array * second_array)))
    return min(1.0, max(0.0, math.sqrt(max(0.0, 1.0 - coefficient))))


def _normalized_descriptor(descriptor: Sequence[float]) -> tuple[float, ...]:
    values = np.asarray(descriptor, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(
            "Appearance descriptor должен быть конечным одномерным вектором"
        )
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 1e-9:
        raise ValueError("Appearance descriptor не может быть нулевым")
    return tuple(float(value) for value in values / total)


@dataclass(slots=True)
class IdentityGallery:
    """Immutable core enrollment plus tightly guarded adaptive samples."""

    max_distance: float = 0.35
    min_margin: float = 0.12
    adaptive_confidence_min: float = 0.90
    adaptive_margin_min: float = 0.20
    max_adaptive_samples: int = 12
    _core: dict[IdentityState, tuple[tuple[float, ...], ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _adaptive: dict[IdentityState, list[tuple[float, ...]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _negative: list[tuple[float, ...]] = field(
        default_factory=list, init=False, repr=False
    )
    _core_parts: dict[IdentityState, tuple[dict[str, AppearancePart], ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _negative_parts: list[dict[str, AppearancePart]] = field(
        default_factory=list, init=False, repr=False
    )
    _adaptive_parts: dict[IdentityState, list[dict[str, AppearancePart]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __init__(
        self,
        core: Mapping[IdentityState | str, Sequence[Sequence[float]]] | None = None,
        negative: Sequence[Sequence[float]] | None = None,
        *,
        max_distance: float = 0.35,
        min_margin: float = 0.12,
        adaptive_confidence_min: float = 0.90,
        adaptive_margin_min: float = 0.20,
        max_adaptive_samples: int = 12,
    ) -> None:
        self.max_distance = float(max_distance)
        self.min_margin = float(min_margin)
        self.adaptive_confidence_min = float(adaptive_confidence_min)
        self.adaptive_margin_min = float(adaptive_margin_min)
        self.max_adaptive_samples = max(1, int(max_adaptive_samples))
        self._core = {}
        self._adaptive = {}
        self._negative = []
        self._core_parts: dict[
            IdentityState, tuple[dict[str, AppearancePart], ...]
        ] = {}
        self._negative_parts: list[dict[str, AppearancePart]] = []
        self._adaptive_parts = {}
        for raw_state, descriptors in (core or {}).items():
            state = IdentityState(raw_state)
            self.set_core_once(state, descriptors)
        for descriptor in negative or ():
            self.add_negative(descriptor)

    @staticmethod
    def _fighter_state(state: IdentityState | str) -> IdentityState:
        normalized = IdentityState(state)
        if normalized not in {IdentityState.FIGHTER_A, IdentityState.FIGHTER_B}:
            raise ValueError(
                "Core/adaptive gallery поддерживает только FIGHTER_A/FIGHTER_B"
            )
        return normalized

    @property
    def core(self) -> Mapping[IdentityState, tuple[tuple[float, ...], ...]]:
        # Tuples prevent callers from mutating enrollment evidence in-place.
        return dict(self._core)

    @property
    def negative(self) -> tuple[tuple[float, ...], ...]:
        return tuple(self._negative)

    def adaptive_samples(
        self, state: IdentityState | str
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(self._adaptive.get(self._fighter_state(state), ()))

    def set_core_once(
        self,
        state: IdentityState | str,
        descriptors: Sequence[Sequence[float]],
    ) -> None:
        fighter_state = self._fighter_state(state)
        if fighter_state in self._core:
            raise RuntimeError(f"Core gallery {fighter_state.value} уже зафиксирована")
        normalized = tuple(_normalized_descriptor(item) for item in descriptors)
        if not normalized:
            raise ValueError("Core gallery должна содержать хотя бы один descriptor")
        self._core[fighter_state] = normalized

    def add_negative(self, descriptor: Sequence[float]) -> None:
        normalized = _normalized_descriptor(descriptor)
        if self._core:
            dimensions = {
                len(item) for samples in self._core.values() for item in samples
            }
            if dimensions and len(normalized) not in dimensions:
                return
            # Unselected people can be mirrors of the enrolled boxer. Such a
            # sample is not a globally valid negative identity template.
            if (
                min(
                    (
                        appearance_distance(normalized, item)
                        for samples in self._core.values()
                        for item in samples
                    ),
                    default=1.0,
                )
                <= self.max_distance
            ):
                return
        if normalized not in self._negative:
            self._negative.append(normalized)
            del self._negative[:-64]

    def set_core_parts_once(
        self,
        state: IdentityState | str,
        samples: Sequence[Mapping[str, AppearancePart]],
    ) -> None:
        state = self._fighter_state(state)
        if state in self._core_parts:
            raise RuntimeError(f"Core parts {state.value} уже зафиксированы")
        self._core_parts[state] = tuple(dict(sample) for sample in samples if sample)

    def add_negative_parts(self, parts: Mapping[str, AppearancePart]) -> None:
        if any(
            distance is not None and distance <= self.max_distance
            for samples in self._core_parts.values()
            for sample in samples
            for distance in [part_appearance_distance(sample, parts)]
        ):
            return
        if parts and dict(parts) not in self._negative_parts:
            self._negative_parts.append(dict(parts))
            del self._negative_parts[:-64]

    def export(self) -> dict[str, object]:
        def encode(parts: Mapping[str, AppearancePart]) -> dict[str, object]:
            return {
                name: {
                    "histogram": list(part.histogram),
                    "reliability": part.reliability,
                }
                for name, part in parts.items()
            }

        return {
            "version": 3,
            "policy": {
                "max_distance": self.max_distance,
                "min_margin": self.min_margin,
                "adaptive_confidence_min": self.adaptive_confidence_min,
                "adaptive_margin_min": self.adaptive_margin_min,
                "max_adaptive_samples": self.max_adaptive_samples,
            },
            "core": {
                str(key): [list(sample) for sample in value]
                for key, value in self._core.items()
            },
            "core_parts": {
                str(key): [encode(sample) for sample in value]
                for key, value in self._core_parts.items()
            },
            "negative": [list(sample) for sample in self._negative],
            "negative_parts": [encode(sample) for sample in self._negative_parts],
            "adaptive": {
                str(key): [list(sample) for sample in value]
                for key, value in self._adaptive.items()
            },
            "adaptive_parts": {
                str(key): [encode(sample) for sample in value]
                for key, value in self._adaptive_parts.items()
            },
        }

    @classmethod
    def from_export(cls, profile: Mapping[str, object]) -> IdentityGallery:
        """Restore enrollment, not future adaptive samples, for deterministic review."""
        if profile.get("version") != 3 or not isinstance(profile.get("core"), Mapping):
            raise ValueError("Нужен identity profile v3 с реальными core descriptors")
        policy = profile.get("policy", {})
        if not isinstance(policy, Mapping):
            raise TypeError("Некорректная identity policy")
        allowed = {
            "max_distance",
            "min_margin",
            "adaptive_confidence_min",
            "adaptive_margin_min",
            "max_adaptive_samples",
        }
        gallery = cls(
            core=profile["core"],
            negative=profile.get("negative", []),
            **{key: value for key, value in policy.items() if key in allowed},
        )

        def decode(sample: Mapping[str, object]) -> dict[str, AppearancePart]:
            parts = {}
            for name, value in sample.items():
                if name not in PART_WEIGHTS or not isinstance(value, Mapping):
                    raise ValueError("Некорректный appearance part")
                reliability = float(value["reliability"])
                if not math.isfinite(reliability) or not 0 <= reliability <= 1:
                    raise ValueError("Некорректная надёжность appearance part")
                parts[name] = AppearancePart(
                    _normalized_descriptor(value["histogram"]), reliability
                )
            return parts

        for state, samples in profile.get("core_parts", {}).items():
            gallery.set_core_parts_once(state, [decode(sample) for sample in samples])
        for sample in profile.get("negative_parts", []):
            gallery.add_negative_parts(decode(sample))
        return gallery

    def distances(
        self,
        descriptor: Sequence[float] | None,
        parts: Mapping[str, AppearancePart] | None = None,
        *,
        include_adaptive: bool = True,
    ) -> tuple[dict[IdentityState, float], float]:
        """Expose the actual classifier evidence for offline decoding and review."""
        distances = {
            state: self._distance_to_state(
                state, descriptor, include_adaptive=include_adaptive
            )
            if descriptor is not None
            else 1.0
            for state in (IdentityState.FIGHTER_A, IdentityState.FIGHTER_B)
        }
        negative_distance = min(
            (appearance_distance(item, descriptor) for item in self._negative),
            default=1.0,
        )
        if parts is not None and all(
            self._core_parts.get(state) for state in distances
        ):
            part_distances = {
                state: min(
                    (
                        distance
                        for sample in self._core_parts[state]
                        + tuple(
                            self._adaptive_parts.get(state, ())
                            if include_adaptive
                            else ()
                        )
                        if (distance := part_appearance_distance(sample, parts))
                        is not None
                    ),
                    default=None,
                )
                for state in distances
            }
            # Keep the available evidence in diagnostics; match() separately
            # prevents an absent alternative from manufacturing a large margin.
            distances = {
                state: float(value) if value is not None else 1.0
                for state, value in part_distances.items()
            }
            negative_distance = min(
                (
                    distance
                    for sample in self._negative_parts
                    if (distance := part_appearance_distance(sample, parts)) is not None
                ),
                default=1.0,
            )
        return distances, negative_distance

    def _distance_to_state(
        self,
        state: IdentityState,
        descriptor: Sequence[float],
        *,
        include_adaptive: bool = True,
    ) -> float:
        samples = self._core.get(state, ()) + tuple(
            self._adaptive.get(state, ()) if include_adaptive else ()
        )
        return min(
            (appearance_distance(sample, descriptor) for sample in samples), default=1.0
        )

    def match(
        self,
        descriptor: Sequence[float] | None,
        parts: Mapping[str, AppearancePart] | None = None,
        *,
        include_adaptive: bool = True,
    ) -> IdentityMatch:
        if descriptor is None:
            return IdentityMatch(
                IdentityState.UNKNOWN,
                1.0,
                0.0,
                1.0,
                False,
                "appearance_missing",
            )
        distances, negative_distance = self.distances(
            descriptor, parts, include_adaptive=include_adaptive
        )
        ordered = sorted(distances.items(), key=lambda item: item[1])
        best_state, best_distance = ordered[0]
        alternative_distance = ordered[1][1]
        margin = alternative_distance - best_distance
        if parts is not None and all(
            self._core_parts.get(state) for state in distances
        ):
            comparable = [
                any(
                    part_appearance_distance(sample, parts) is not None
                    for sample in self._core_parts[state]
                    + tuple(
                        self._adaptive_parts.get(state, ()) if include_adaptive else ()
                    )
                )
                for state in distances
            ]
            if not all(comparable):
                return IdentityMatch(
                    IdentityState.UNKNOWN,
                    best_distance,
                    0.0,
                    negative_distance,
                    False,
                    "appearance_missing" if not parts else "appearance_partial",
                )
        if (
            negative_distance <= self.max_distance
            and best_distance - negative_distance >= self.min_margin
        ):
            return IdentityMatch(
                IdentityState.OTHER,
                best_distance,
                margin,
                negative_distance,
                False,
                "negative_gallery_closer",
            )
        if negative_distance < 1.0 and negative_distance <= best_distance:
            return IdentityMatch(
                IdentityState.UNKNOWN,
                best_distance,
                margin,
                negative_distance,
                False,
                "negative_gallery_ambiguous",
            )
        if best_distance > self.max_distance:
            return IdentityMatch(
                IdentityState.UNKNOWN,
                best_distance,
                margin,
                negative_distance,
                False,
                "gallery_distance",
            )
        if margin < self.min_margin:
            return IdentityMatch(
                IdentityState.UNKNOWN,
                best_distance,
                margin,
                negative_distance,
                False,
                "identity_margin",
            )
        return IdentityMatch(
            best_state,
            best_distance,
            margin,
            negative_distance,
            True,
            "gallery_match",
        )

    def update_adaptive(
        self,
        state: IdentityState | str,
        descriptor: Sequence[float] | None,
        *,
        confidence: float,
        margin: float,
        stable_frames: int,
        overlap_or_clinch: bool,
        active_fight: bool,
        parts: Mapping[str, AppearancePart] | None = None,
        evidence_origin: str = "enrollment",
    ) -> bool:
        """Add one adaptive sample only after every contamination guard passes."""

        fighter_state = self._fighter_state(state)
        if (
            descriptor is None
            or confidence < self.adaptive_confidence_min
            or margin < self.adaptive_margin_min
            or stable_frames < 5
            or overlap_or_clinch
            or not active_fight
            or fighter_state not in self._core
            or evidence_origin not in {"enrollment", "gallery_match"}
        ):
            return False
        normalized = _normalized_descriptor(descriptor)
        anchor = self.match(descriptor, parts, include_adaptive=False)
        if not anchor.accepted or anchor.state != fighter_state:
            return False
        if len(normalized) != len(self._core[fighter_state][0]):
            return False
        samples = self._adaptive.setdefault(fighter_state, [])
        samples.append(normalized)
        del samples[: max(0, len(samples) - self.max_adaptive_samples)]
        if parts:
            part_samples = self._adaptive_parts.setdefault(fighter_state, [])
            part_samples.append(dict(parts))
            del part_samples[: -self.max_adaptive_samples]
        return True
