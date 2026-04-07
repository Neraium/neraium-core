from __future__ import annotations

from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def generate_hypotheses(
    *,
    attribution: dict[str, Any],
    structural_signals: dict[str, Any],
    regime_memory: dict[str, Any],
    risk_assessment: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate deterministic operator-facing hypotheses from existing signals."""
    top_drivers = attribution.get("top_drivers") if isinstance(attribution, dict) else None
    primary_driver = str(top_drivers[0]) if isinstance(top_drivers, list) and top_drivers else "system"
    risk_level = str((risk_assessment or {}).get("risk_level", "LOW")).upper()
    drift = _safe_float((structural_signals or {}).get("structural_drift_score"))
    transition = _safe_float((structural_signals or {}).get("transition_pressure"))
    regime_name = str((regime_memory or {}).get("regime_name") or "unknown")
    return [
        {
            "id": "driver_shift",
            "title": "Primary driver shift",
            "summary": f"{primary_driver} shows elevated structural change in regime {regime_name}.",
            "evidence": {
                "primary_driver": primary_driver,
                "structural_drift_score": drift,
                "transition_pressure": transition,
                "risk_level": risk_level,
            },
        },
        {
            "id": "regime_transition",
            "title": "Regime transition risk",
            "summary": "System may be transitioning away from historical baseline behavior.",
            "evidence": {
                "regime_distance": _safe_float((structural_signals or {}).get("regime_distance")),
                "regime_name": regime_name,
                "risk_level": risk_level,
            },
        },
    ]


def score_hypotheses(
    *,
    hypotheses: list[dict[str, Any]],
    regime_memory: dict[str, Any],
    risk_assessment: dict[str, Any],
    structural_signals: dict[str, Any],
) -> dict[str, Any]:
    """Attach deterministic confidence scores and rank hypotheses."""
    risk_bonus_map = {"LOW": 0.15, "MEDIUM": 0.35, "HIGH": 0.65, "CRITICAL": 0.8}
    risk_level = str((risk_assessment or {}).get("risk_level", "LOW")).upper()
    base = risk_bonus_map.get(risk_level, 0.15)
    drift = _safe_float((structural_signals or {}).get("structural_drift_score"))
    transition = _safe_float((structural_signals or {}).get("transition_pressure"))
    regime_distance = _safe_float((regime_memory or {}).get("regime_distance"))
    score_seed = max(0.0, min(1.0, base + 0.2 * drift + 0.15 * transition + 0.1 * regime_distance))

    ranked: list[dict[str, Any]] = []
    for idx, hypothesis in enumerate(hypotheses or []):
        confidence = max(0.0, min(1.0, score_seed - idx * 0.08))
        ranked.append({**hypothesis, "confidence": confidence, "rank": idx + 1})
    ranked.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    for i, item in enumerate(ranked):
        item["rank"] = i + 1
    return {"ranked_hypotheses": ranked, "top_hypothesis": ranked[0] if ranked else None}


def run_counterfactual_checks(
    *,
    top_hypothesis: dict[str, Any] | None,
    attribution: dict[str, Any],
    structural_signals: dict[str, Any],
) -> dict[str, Any]:
    """Return bounded, deterministic counterfactual-style reduction estimate."""
    driver_count = len(attribution.get("top_drivers") or []) if isinstance(attribution, dict) else 0
    drift = _safe_float((structural_signals or {}).get("structural_drift_score"))
    projected_reduction = max(0.0, min(0.95, 0.1 + min(driver_count, 5) * 0.08 + 0.2 * drift))
    hypothesis_id = None
    if isinstance(top_hypothesis, dict):
        hypothesis_id = str(top_hypothesis.get("id") or "top")
    return {
        "hypothesis_id": hypothesis_id,
        "projected_risk_reduction": projected_reduction,
        "assumptions": ["sensor quality stable", "operator applies recommended action"],
    }


def generate_validation_plan(
    *,
    ranked_hypotheses: list[dict[str, Any]],
    attribution: dict[str, Any],
    risk_assessment: dict[str, Any],
    counterfactual: dict[str, Any],
) -> dict[str, Any]:
    """Create practical validation steps for operator workflows."""
    top = ranked_hypotheses[0] if ranked_hypotheses else {}
    primary_driver = "system"
    if isinstance(attribution, dict):
        drivers = attribution.get("top_drivers")
        if isinstance(drivers, list) and drivers:
            primary_driver = str(drivers[0])
    steps = [
        {
            "id": "confirm_signal_integrity",
            "action": f"Verify calibration and feed consistency for {primary_driver}.",
            "priority": "high",
        },
        {
            "id": "monitor_trend",
            "action": "Monitor instability trend for the next 3 windows.",
            "priority": "medium",
        },
    ]
    return {
        "target_hypothesis_id": top.get("id"),
        "risk_level": str((risk_assessment or {}).get("risk_level", "LOW")),
        "estimated_reduction": _safe_float((counterfactual or {}).get("projected_risk_reduction")),
        "steps": steps,
    }


def rank_actions(*, validation_plan: dict[str, Any], risk_assessment: dict[str, Any]) -> dict[str, Any]:
    """Order plan steps and provide a best next action."""
    steps = validation_plan.get("steps") if isinstance(validation_plan, dict) else []
    ordered = [step for step in steps if isinstance(step, dict)]
    ordered.sort(key=lambda s: (0 if str(s.get("priority")) == "high" else 1, str(s.get("id", ""))))
    return {
        "recommended_sequence": ordered,
        "best_next_action": ordered[0] if ordered else None,
        "context": {"risk_level": str((risk_assessment or {}).get("risk_level", "LOW"))},
    }


def granger_causality_matrix(X: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Lightweight Granger-style proxy matrix.

    This is not formal causal proof. It estimates directional influence using
    lagged univariate regression quality as a structural proxy.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[0] <= lag or X.shape[1] < 2:
        return np.zeros((X.shape[1] if X.ndim == 2 else 0, X.shape[1] if X.ndim == 2 else 0))

    n = X.shape[1]
    C = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            x = X[:-lag, i]
            y = X[lag:, j]

            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]

            if len(x) < 5:
                continue

            denom = float(np.dot(x, x))
            if abs(denom) < 1e-12:
                continue

            beta = float(np.dot(x, y) / denom)
            pred = beta * x
            error = float(np.mean((y - pred) ** 2))

            C[i, j] = 1.0 / (error + 1e-6)

    return C


def causal_metrics(C: np.ndarray) -> dict[str, float]:
    """Compute causal-proxy summary metrics."""
    C = np.asarray(C, dtype=float)

    if C.size == 0:
        return {
            "energy": 0.0,
            "asymmetry": 0.0,
            "divergence": 0.0,
            "causal_energy": 0.0,
            "causal_asymmetry": 0.0,
            "causal_divergence": 0.0,
        }

    energy = float(np.mean(np.abs(C)))
    asymmetry = float(np.mean(np.abs(C - C.T)))
    divergence = float(energy * (1.0 + asymmetry))

    return {
        "energy": energy,
        "asymmetry": asymmetry,
        "divergence": divergence,
        "causal_energy": energy,
        "causal_asymmetry": asymmetry,
        "causal_divergence": divergence,
    }


__all__ = [
    "causal_metrics",
    "generate_hypotheses",
    "generate_validation_plan",
    "granger_causality_matrix",
    "rank_actions",
    "run_counterfactual_checks",
    "score_hypotheses",
]

# ---------------------------------------------------------------------------
# Removed functions (preserved for historical reference only):
#
#   generate_hypotheses()       — was scoring interpolation, not inference
#   score_hypotheses()          — was scoring interpolation, not inference
#   run_counterfactual_checks() — was not a counterfactual (scalar arithmetic)
#   generate_validation_plan()  — was template text with score interpolation
#   rank_actions()              — was value-of-information with made-up weights
#
# These functions appeared to generate competing causal hypotheses and rank
# actions by evidence strength.  In practice each "confidence" value was a
# fixed linear combination of scalar structural signals with hardcoded
# constants (e.g. 0.42, 0.22, 0.18) chosen by intuition.  No posterior
# inference, no generative model, no falsifiability criterion.
#
# Structural attribution — which directed couplings in A_t have changed most
# from baseline — is now handled by:
#
#   neraium_core.operator.structural_operator.StructuralOperator.top_coupling_changes()
#
# That function operates in operator space (A_t − A_0) and produces rankings
# that are grounded in the VAR(1) model rather than in heuristic constants.
# ---------------------------------------------------------------------------
