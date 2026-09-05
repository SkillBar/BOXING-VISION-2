"""One fail-closed eligibility rule shared by UI, JSON and MP4 presenters."""
from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def result_eligibility(summary: Mapping[str, Any]) -> tuple[bool, list[str]]:
    quality = summary.get("quality")
    if not isinstance(quality, Mapping):
        return False, ["Качество идентичности ещё не проверено"]
    reasons: list[str] = []
    try:
        coverage = float(quality.get("identity_verified_coverage"))
    except (TypeError, ValueError):
        coverage = float("nan")
    if isinstance(quality.get("identity_verified_coverage"), bool) or not math.isfinite(coverage) or not 0.90 <= coverage <= 1.0:
        reasons.append("Недостаточное покрытие подтверждёнными личностями")
    if quality.get("identity_swap_suspected"):
        reasons.append("Нужно проверить возможную перестановку бойцов")
    pending = quality.get("required_review_count", 0)
    if not isinstance(pending, int) or isinstance(pending, bool) or pending != 0:
        reasons.append("Остались обязательные проверки")
    if quality.get("winner_visible") is False and not reasons:
        reasons.append("Итоговая оценка недоступна для этого анализа")
    metadata = summary.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("review_recompute_state") == "pending":
        reasons.append("Пересчёт проверки не завершён")
    return not reasons, reasons


def apply_result_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Do not leak disallowed winner/round points through an alternate view."""
    result = deepcopy(dict(summary))
    allowed, reasons = result_eligibility(result)
    quality = result.get("quality")
    if not isinstance(quality, dict):
        quality = {}
        result["quality"] = quality
    quality["winner_visible"] = allowed
    quality["result_block_reasons"] = reasons
    if not allowed:
        result.update(winner_id=None, winner_name=None, confidence=0.0,
                      winner={}, score_total={}, round_scores=[])
    return result
