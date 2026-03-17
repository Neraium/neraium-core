from __future__ import annotations


def composite_score(components: dict[str, float], weights: dict[str, float] | None = None) -> float:
    default_weights = {
        "spectral_radius": 0.12,
        "inverse_spectral_gap": 0.12,
        "causal_divergence": 0.10,
        "graph_stability": 0.08,
        "lag1_autocorr_avg": 0.10,
        "baseline_relative_drift": 0.12,
        "regime_relative_drift": 0.10,
        "entropy": 0.08,
        "subsystem_instability": 0.10,
        "forecast_contribution": 0.08,
    }
    active = weights or default_weights
    return float(sum(components.get(k, 0.0) * w for k, w in active.items()))
