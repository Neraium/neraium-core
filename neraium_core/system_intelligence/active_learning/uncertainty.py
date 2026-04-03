from __future__ import annotations

import math
from typing import Any


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def decompose_uncertainty(
    *,
    transition: dict[str, Any],
    trajectory: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    reliability: dict[str, Any],
    cross_system: dict[str, Any],
) -> dict[str, Any]:
    """Produce an interpretable uncertainty decomposition for active experiment design."""

    support = float(trajectory.get("support_count", 0.0))
    novelty = float(trajectory.get("novelty_score", 0.5))
    model_uncertainty = _clip01(0.6 * (1.0 / (1.0 + support / 4.0)) + 0.4 * novelty)

    sorted_likelihoods = sorted([float(h.get("likelihood", 0.0)) for h in hypotheses], reverse=True)
    top = sorted_likelihoods[0] if sorted_likelihoods else 0.0
    second = sorted_likelihoods[1] if len(sorted_likelihoods) > 1 else 0.0
    structural_ambiguity = _clip01(1.0 - (top - second))

    transfer = cross_system.get("transfer_adaptation") or {}
    transfer_conf = float(transfer.get("transfer_confidence", 0.0))
    domain_mismatch = _clip01(1.0 - transfer_conf)

    rec = reliability.get("intervention_recommendation") or {}
    calibrated = float(rec.get("recommendation_calibrated_confidence", 0.0))
    transition_unc = float(transition.get("uncertainty", 0.5))
    calibration_uncertainty = _clip01(0.6 * (1.0 - calibrated) + 0.4 * transition_unc)

    vector = {
        "model_uncertainty": round(model_uncertainty, 4),
        "structural_ambiguity": round(structural_ambiguity, 4),
        "domain_mismatch": round(domain_mismatch, 4),
        "calibration_uncertainty": round(calibration_uncertainty, 4),
    }
    dominant = max(vector, key=vector.get) if vector else "model_uncertainty"
    magnitude = _clip01(math.sqrt(sum(v * v for v in vector.values()) / max(1, len(vector))))

    return {
        "uncertainty_vector": vector,
        "dominant_uncertainty_type": dominant,
        "uncertainty_magnitude": round(magnitude, 4),
        "advisory": True,
        "status": "experimental",
    }
