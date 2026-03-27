from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForwardRiskAssessment:
    current_risk_level: str
    projected_near_term_trend: str
    trajectory: str
    risk_score: float
    projected_score: float
    evidence: dict[str, float]


def _risk_level(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def assess_forward_risk(
    *,
    composite_history: list[float],
    structural_score: float,
    regime_score: float,
    coupling_score: float,
) -> ForwardRiskAssessment:
    arr = np.asarray(composite_history[-20:], dtype=float)
    if arr.size < 3:
        score = float(max(structural_score, regime_score, coupling_score))
        return ForwardRiskAssessment(
            current_risk_level=_risk_level(score),
            projected_near_term_trend="uncertain",
            trajectory="stabilizing",
            risk_score=round(score, 4),
            projected_score=round(score, 4),
            evidence={"trend": 0.0, "acceleration": 0.0, "persistence": 0.0},
        )

    x = np.arange(arr.size, dtype=float)
    slope, intercept = np.polyfit(x, arr, 1)
    slope = float(slope)
    recent_diff = np.diff(arr)
    accel = float(np.mean(recent_diff[-4:])) if recent_diff.size else 0.0
    persistence = float(np.mean(arr[-6:] > 0.55)) if arr.size >= 6 else float(np.mean(arr > 0.55))

    current = float(0.60 * arr[-1] + 0.20 * structural_score + 0.20 * coupling_score)
    projected = float(arr[-1] + 5.0 * slope + 2.0 * accel + 0.12 * persistence + 0.08 * regime_score)
    projected = float(max(0.0, min(1.0, projected)))

    if slope > 0.015 or accel > 0.012:
        trend = "increasing"
        trajectory = "deteriorating"
    elif slope < -0.012 and accel <= 0.0:
        trend = "decreasing"
        trajectory = "stabilizing"
    else:
        trend = "flat"
        trajectory = "drifting"

    return ForwardRiskAssessment(
        current_risk_level=_risk_level(current),
        projected_near_term_trend=trend,
        trajectory=trajectory,
        risk_score=round(max(0.0, min(1.0, current)), 4),
        projected_score=round(projected, 4),
        evidence={"trend": round(slope, 6), "acceleration": round(accel, 6), "persistence": round(persistence, 6)},
    )
