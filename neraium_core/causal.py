from __future__ import annotations

from typing import Any

import numpy as np


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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_driver_scores(raw_scores: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for k, v in (raw_scores or {}).items():
        scores[str(k)] = max(0.0, _to_float(v, 0.0))
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 1e-12:
        return {k: 0.0 for k in scores}
    return {k: _clamp01(v / max_score) for k, v in scores.items()}


def _driver_scores_from_attribution(attr: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(attr, dict):
        return {}

    raw_scores = attr.get("driver_scores", {}) if isinstance(attr.get("driver_scores"), dict) else {}
    normalized = _normalize_driver_scores(raw_scores)
    if normalized:
        return normalized

    inferred: dict[str, float] = {}
    for item in (attr.get("top_drivers", []) if isinstance(attr, dict) else []):
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature", "")).strip()
        if not feature:
            continue
        inferred[feature] = max(
            _to_float(item.get("score", 0.0)),
            _to_float(item.get("contribution", 0.0)),
            _to_float(item.get("importance", 0.0)),
        )
    return _normalize_driver_scores(inferred)


def generate_hypotheses(
    *,
    attribution: dict[str, Any] | None,
    structural_signals: dict[str, Any] | None,
    regime_memory: dict[str, Any] | None,
    risk_assessment: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Generate deterministic, signal-grounded competing causal hypotheses."""
    attr = attribution or {}
    signals = structural_signals or {}
    regime = regime_memory or {}
    risk = risk_assessment or {}

    top_drivers = attr.get("top_drivers", []) if isinstance(attr, dict) else []
    driver_scores = _driver_scores_from_attribution(attr)

    drift = _to_float(signals.get("structural_drift_score", signals.get("drift", 0.0)))
    relational = _to_float(signals.get("relational_instability_score", signals.get("relational_drift", 0.0)))
    regime_distance = _to_float(signals.get("regime_distance", regime.get("regime_distance", 0.0)))
    regime_drift = _to_float(signals.get("regime_drift", 0.0))
    transition_pressure = _to_float(signals.get("transition_pressure", 0.0))
    localization = _clamp01(_to_float(signals.get("localization_score", 0.0)))
    temporal_distortion = _to_float(signals.get("temporal_distortion_score", 0.0))

    risk_level = str(risk.get("risk_level", "LOW")).upper()
    risk_weight = {"LOW": 0.2, "MEDIUM": 0.45, "ELEVATED": 0.6, "HIGH": 0.75}.get(risk_level, 0.3)

    driver_names = [str(d.get("feature", "")) for d in top_drivers if isinstance(d, dict) and d.get("feature")]
    primary_drivers = driver_names[:5]

    strongest_driver_score = 0.0
    if primary_drivers:
        strongest_driver_score = max(driver_scores.get(d, 0.0) for d in primary_drivers)

    # Hypothesis A: localized physical degradation.
    localized_support = _clamp01(0.42 * _clamp01(drift / 2.0) + 0.22 * _clamp01(relational / 2.0) + 0.18 * localization + 0.18 * risk_weight)
    localized_conflict = _clamp01(0.45 * _clamp01(temporal_distortion / 2.0) + 0.35 * _clamp01(regime_distance / 2.0) + 0.2 * _clamp01(1.0 - localization))
    physical_conf = _clamp01(localized_support * (1.0 - 0.45 * localized_conflict))

    hypotheses: list[dict[str, Any]] = [
        {
            "hypothesis_id": "hyp_physical_localized_degradation",
            "description": "Localized subsystem degradation is driving observed structural instability.",
            "type": "physical",
            "confidence": round(physical_conf, 4),
            "supporting_evidence": [
                f"structural_drift_score={drift:.4f}",
                f"relational_instability_score={relational:.4f}",
                f"localization_score={localization:.4f}",
            ],
            "conflicting_evidence": [
                f"temporal_distortion_score={temporal_distortion:.4f}",
                f"regime_distance={regime_distance:.4f}",
            ],
            "primary_drivers": primary_drivers,
        }
    ]

    # Hypothesis B: instrumentation/sensor drift.
    sensor_support = _clamp01(0.44 * _clamp01(temporal_distortion / 2.0) + 0.22 * strongest_driver_score + 0.22 * _clamp01(1.0 - localization) + 0.12 * _clamp01(regime_distance / 2.0))
    sensor_conflict = _clamp01(0.50 * _clamp01(relational / 2.0) + 0.25 * _clamp01(transition_pressure / 2.0) + 0.25 * localization)
    sensor_conf = _clamp01(sensor_support * (1.0 - 0.5 * sensor_conflict))
    hypotheses.append(
        {
            "hypothesis_id": "hyp_sensor_instrumentation_drift",
            "description": "Instrumentation drift or sensor fault is contributing to apparent structural change.",
            "type": "sensor",
            "confidence": round(sensor_conf, 4),
            "supporting_evidence": [
                f"temporal_distortion_score={temporal_distortion:.4f}",
                f"top_driver_strength={strongest_driver_score:.4f}",
                f"regime_distance={regime_distance:.4f}",
            ],
            "conflicting_evidence": [
                f"relational_instability_score={relational:.4f}",
                f"transition_pressure={transition_pressure:.4f}",
            ],
            "primary_drivers": primary_drivers[:3],
        }
    )

    # Hypothesis C: distributed systemic imbalance / coupling breakdown.
    systemic_support = _clamp01(0.28 * _clamp01(relational / 2.0) + 0.24 * _clamp01(regime_drift / 2.0) + 0.24 * _clamp01(transition_pressure / 2.0) + 0.24 * _clamp01(1.0 - localization))
    systemic_conflict = _clamp01(0.45 * strongest_driver_score + 0.30 * localization + 0.25 * _clamp01(temporal_distortion / 2.0))
    systemic_conf = _clamp01(systemic_support * (1.0 - 0.40 * systemic_conflict))
    hypotheses.append(
        {
            "hypothesis_id": "hyp_systemic_coupling_breakdown",
            "description": "Distributed coupling imbalance between components is causing system-wide instability.",
            "type": "systemic",
            "confidence": round(systemic_conf, 4),
            "supporting_evidence": [
                f"relational_instability_score={relational:.4f}",
                f"regime_drift={regime_drift:.4f}",
                f"transition_pressure={transition_pressure:.4f}",
            ],
            "conflicting_evidence": [
                f"localization_score={localization:.4f}",
                f"top_driver_strength={strongest_driver_score:.4f}",
            ],
            "primary_drivers": primary_drivers,
        }
    )

    # Stable deterministic ordering before downstream scoring/ranking.
    return sorted(hypotheses, key=lambda h: str(h.get("hypothesis_id", "")))


def score_hypotheses(
    *,
    hypotheses: list[dict[str, Any]] | None,
    regime_memory: dict[str, Any] | None,
    risk_assessment: dict[str, Any] | None,
    structural_signals: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score and rank hypotheses using evidence strength, regime consistency, risk alignment, and coherence."""
    regime = regime_memory or {}
    risk = risk_assessment or {}
    signals = structural_signals or {}

    regime_distance = _to_float(signals.get("regime_distance", regime.get("regime_distance", 0.0)))
    regime_drift = _to_float(signals.get("regime_drift", 0.0))
    transition_pressure = _to_float(signals.get("transition_pressure", 0.0))
    localization = _clamp01(_to_float(signals.get("localization_score", 0.0)))

    risk_level = str(risk.get("risk_level", "LOW")).upper()
    risk_scale = {"LOW": 0.2, "MEDIUM": 0.45, "ELEVATED": 0.6, "HIGH": 0.8}.get(risk_level, 0.3)

    ranked: list[dict[str, Any]] = []
    for hyp in (hypotheses or []):
        base_conf = _clamp01(_to_float(hyp.get("confidence", 0.0)))
        supports = hyp.get("supporting_evidence", [])
        conflicts = hyp.get("conflicting_evidence", [])

        evidence_strength = _clamp01(base_conf + 0.04 * min(5, len(supports)) - 0.05 * min(5, len(conflicts)))
        regime_consistency = _clamp01(1.0 - 0.35 * _clamp01(regime_distance / 2.0) + 0.25 * _clamp01(regime_drift / 2.0))
        risk_alignment = _clamp01(0.4 * risk_scale + 0.6 * _clamp01(transition_pressure / 2.0))

        hyp_type = str(hyp.get("type", ""))
        if hyp_type == "physical":
            coherence = _clamp01(0.65 * localization + 0.35 * (1.0 - _clamp01(transition_pressure / 2.0)))
        elif hyp_type == "sensor":
            coherence = _clamp01(0.55 * (1.0 - localization) + 0.45 * (1.0 - _clamp01(regime_drift / 2.0)))
        else:
            coherence = _clamp01(0.65 * (1.0 - localization) + 0.35 * _clamp01(transition_pressure / 2.0))

        ranking_score = _clamp01(0.40 * evidence_strength + 0.20 * regime_consistency + 0.25 * risk_alignment + 0.15 * coherence)

        enriched = dict(hyp)
        enriched["confidence"] = round(_clamp01(0.5 * base_conf + 0.5 * ranking_score), 4)
        enriched["ranking_score"] = round(ranking_score, 4)
        enriched["scoring_factors"] = {
            "evidence_strength": round(evidence_strength, 4),
            "regime_consistency": round(regime_consistency, 4),
            "risk_alignment": round(risk_alignment, 4),
            "coherence": round(coherence, 4),
        }
        ranked.append(enriched)

    ranked_sorted = sorted(
        ranked,
        key=lambda h: (
            -_to_float(h.get("ranking_score", 0.0)),
            -_to_float(h.get("confidence", 0.0)),
            str(h.get("hypothesis_id", "")),
        ),
    )
    return {
        "ranked_hypotheses": ranked_sorted,
        "top_hypothesis": ranked_sorted[0] if ranked_sorted else None,
    }


def run_counterfactual_checks(
    *,
    top_hypothesis: dict[str, Any] | None,
    attribution: dict[str, Any] | None,
    structural_signals: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run lightweight, proxy counterfactual checks without recomputing full models."""
    hyp = top_hypothesis or {}
    attr = attribution or {}
    signals = structural_signals or {}

    top_drivers = attr.get("top_drivers", []) if isinstance(attr, dict) else []
    driver_scores = _driver_scores_from_attribution(attr)

    drift = _to_float(signals.get("structural_drift_score", signals.get("drift", 0.0)))
    relational = _to_float(signals.get("relational_instability_score", signals.get("relational_drift", 0.0)))
    localization = _clamp01(_to_float(signals.get("localization_score", 0.0)))

    top_driver = None
    if top_drivers and isinstance(top_drivers[0], dict):
        top_driver = str(top_drivers[0].get("feature", "")) or None

    top_driver_weight = driver_scores.get(top_driver, 0.0) if top_driver else 0.0
    residual_if_removed = max(0.0, drift * (1.0 - 0.75 * top_driver_weight))
    relationship_removed = max(0.0, relational * (1.0 - 0.65 * (1.0 - localization)))

    checks = [
        {
            "check": "remove_top_driver",
            "target": top_driver,
            "baseline_instability": round(drift, 4),
            "estimated_instability_after_removal": round(residual_if_removed, 4),
            "persists": bool(residual_if_removed >= 0.6 * max(1e-6, drift)),
        },
        {
            "check": "remove_relationship_change",
            "target": "relationship_structure",
            "baseline_instability": round(relational, 4),
            "estimated_instability_after_removal": round(relationship_removed, 4),
            "stabilizes": bool(relationship_removed <= 0.5 * max(1e-6, relational)),
        },
        {
            "check": "localization_dependency",
            "target": "driver_subset",
            "localization_score": round(localization, 4),
            "depends_on_small_subset": bool(localization >= 0.6),
        },
    ]

    robust_signals = 0
    if checks[0]["persists"]:
        robust_signals += 1
    if not checks[1]["stabilizes"]:
        robust_signals += 1
    if not checks[2]["depends_on_small_subset"]:
        robust_signals += 1
    robustness = _clamp01(robust_signals / 3.0)

    hyp_label = str(hyp.get("hypothesis_id", "hypothesis"))
    if robustness >= 0.67:
        interpretation = f"{hyp_label} appears robust under lightweight counterfactual stress checks."
    elif robustness >= 0.34:
        interpretation = f"{hyp_label} has mixed robustness; targeted validation is required."
    else:
        interpretation = f"{hyp_label} is fragile under counterfactual checks; prioritize alternatives."

    return {
        "counterfactual_checks": checks,
        "robustness": round(robustness, 4),
        "interpretation": interpretation,
    }


def generate_validation_plan(
    *,
    ranked_hypotheses: list[dict[str, Any]] | None,
    attribution: dict[str, Any] | None,
    risk_assessment: dict[str, Any] | None,
    counterfactual: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Generate conservative confirm/falsify actions for top hypotheses."""
    hypotheses = ranked_hypotheses or []
    attr = attribution or {}
    risk = risk_assessment or {}
    cf = counterfactual or {}

    top_drivers = attr.get("top_drivers", []) if isinstance(attr, dict) else []
    primary_target = "structural_topology"
    if top_drivers and isinstance(top_drivers[0], dict):
        primary_target = str(top_drivers[0].get("feature", primary_target))

    risk_level = str(risk.get("risk_level", "LOW")).upper()
    risk_boost = {"LOW": 0.0, "MEDIUM": 0.08, "ELEVATED": 0.12, "HIGH": 0.16}.get(risk_level, 0.04)
    cf_robustness = _to_float(cf.get("robustness", 0.0))

    plan: list[dict[str, Any]] = []
    for idx, hyp in enumerate(hypotheses[:3], start=1):
        hyp_type = str(hyp.get("type", "systemic"))
        hyp_conf = _clamp01(_to_float(hyp.get("confidence", 0.0)))

        if hyp_type == "physical":
            action = "Inspect subsystem integrity and load-path consistency"
            target = primary_target
            true_outcome = "Localized stress/degradation signatures persist and align with structural drivers."
            false_outcome = "No localized degradation found; instability likely sensor/systemic."
            invasiveness = 0.45
            time_to_validation = 0.55
        elif hyp_type == "sensor":
            action = "Run calibration and cross-sensor consistency check"
            target = primary_target
            true_outcome = "Calibration drift or sensor inconsistency is confirmed."
            false_outcome = "Sensor calibration is stable; physical/systemic hypothesis gains support."
            invasiveness = 0.15
            time_to_validation = 0.25
        else:
            action = "Audit inter-component coupling and balancing constraints"
            target = "relationship_graph"
            true_outcome = "Cross-component coupling degradation or imbalance is observed."
            false_outcome = "Coupling remains stable; investigate localized drivers/instrumentation."
            invasiveness = 0.35
            time_to_validation = 0.45

        conf = _clamp01(0.75 * hyp_conf + 0.15 * cf_robustness + 0.10 + risk_boost)

        plan.append(
            {
                "action": action,
                "target": target,
                "expected_outcome_if_true": true_outcome,
                "expected_outcome_if_false": false_outcome,
                "priority": idx,
                "confidence": round(conf, 4),
                "uncertainty_reduction": round(_clamp01(0.55 * hyp_conf + 0.30 * (1.0 - idx * 0.2) + 0.15 * (1.0 - cf_robustness)), 4),
                "risk_relevance": round(_clamp01(0.55 * risk_boost + 0.45 * hyp_conf), 4),
                "cost_invasiveness": round(invasiveness, 4),
                "time_to_validation": round(time_to_validation, 4),
            }
        )

    return plan


def rank_actions(
    *,
    validation_plan: list[dict[str, Any]] | None,
    risk_assessment: dict[str, Any] | None,
) -> dict[str, Any]:
    """Rank actions by value-of-information and operational practicality."""
    risk = risk_assessment or {}
    risk_level = str(risk.get("risk_level", "LOW")).upper()
    risk_weight = {"LOW": 0.2, "MEDIUM": 0.5, "ELEVATED": 0.65, "HIGH": 0.8}.get(risk_level, 0.3)

    ranked: list[dict[str, Any]] = []
    for action in (validation_plan or []):
        uncertainty_reduction = _clamp01(_to_float(action.get("uncertainty_reduction", 0.0)))
        risk_relevance = _clamp01(max(_to_float(action.get("risk_relevance", 0.0)), risk_weight * 0.6))
        cost = _clamp01(_to_float(action.get("cost_invasiveness", 0.5)))
        time_to_validation = _clamp01(_to_float(action.get("time_to_validation", 0.5)))

        voi_score = _clamp01(
            0.40 * uncertainty_reduction
            + 0.30 * risk_relevance
            + 0.18 * (1.0 - cost)
            + 0.12 * (1.0 - time_to_validation)
        )
        enriched = dict(action)
        enriched["voi_score"] = round(voi_score, 4)
        ranked.append(enriched)

    ordered = sorted(
        ranked,
        key=lambda a: (
            -_to_float(a.get("voi_score", 0.0)),
            int(_to_float(a.get("priority", 99))),
            str(a.get("action", "")),
        ),
    )
    return {
        "recommended_sequence": ordered,
        "best_next_action": ordered[0] if ordered else None,
    }


__all__ = [
    "causal_metrics",
    "granger_causality_matrix",
    "generate_hypotheses",
    "score_hypotheses",
    "run_counterfactual_checks",
    "generate_validation_plan",
    "rank_actions",
]
