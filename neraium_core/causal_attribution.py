"""
Causal attribution layer: observational ranking of signals/nodes that contribute
most to current instability. Read-only; no interventions or operational directives.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _per_signal_correlation_drift_contribution(
    corr_baseline: np.ndarray,
    corr_recent: np.ndarray,
) -> np.ndarray:
    """
    Approximate each signal's contribution to total correlation drift via
    row/column Frobenius contribution. Observational only.
    """
    delta = np.asarray(corr_recent, dtype=float) - np.asarray(corr_baseline, dtype=float)
    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    n = delta.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    # Contribution of row i: norm of row i and column i (excluding diagonal once)
    contrib = np.zeros(n, dtype=float)
    for i in range(n):
        row_norm = float(np.linalg.norm(delta[i, :]))
        col_norm = float(np.linalg.norm(delta[:, i]))
        # Avoid double-counting diagonal
        diag_val = delta[i, i]
        contrib[i] = np.sqrt(row_norm**2 + col_norm**2 - diag_val**2)
    return contrib


def _causal_outbound_strength(causal_matrix: np.ndarray) -> np.ndarray:
    """Per-node outbound causal influence (sum of absolute causal weights)."""
    C = np.asarray(causal_matrix, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        return np.array([], dtype=float)
    outbound = np.sum(np.abs(C), axis=1)
    return np.nan_to_num(outbound, nan=0.0, posinf=0.0, neginf=0.0)


def causal_attribution(
    corr_baseline: np.ndarray,
    corr_recent: np.ndarray,
    causal_matrix: np.ndarray,
    sensor_names: list[str],
    *,
    top_k: int = 10,
    drift_weight: float = 0.5,
    causal_weight: float = 0.3,
    importance_weight: float = 0.2,
) -> dict[str, Any]:
    """
    Compute ranked attribution of which signals contribute most to current
    structural instability. Purely observational; no causal claims.

    Returns dict with:
      - top_drivers: list of sensor names ordered by descending driver score
      - driver_scores: dict mapping sensor name -> score in [0, 1] scale
    """
    n = corr_recent.shape[0] if corr_recent.size else 0
    if n == 0 or len(sensor_names) < n:
        return {
            "top_drivers": [],
            "driver_scores": {},
        }

    names = list(sensor_names)[:n]

    # 1) Per-signal correlation drift contribution (normalized to [0,1] scale)
    drift_contrib = _per_signal_correlation_drift_contribution(corr_baseline, corr_recent)
    if drift_contrib.size == 0:
        drift_contrib = np.zeros(n, dtype=float)
    drift_max = float(np.max(drift_contrib)) + 1e-12
    drift_norm = np.clip(drift_contrib / drift_max, 0.0, 1.0)

    # 2) Causal outbound strength (who influences others)
    causal_strength = _causal_outbound_strength(causal_matrix)
    if causal_strength.size != n:
        causal_strength = np.zeros(n, dtype=float)
    causal_max = float(np.max(causal_strength)) + 1e-12
    causal_norm = np.clip(causal_strength / causal_max, 0.0, 1.0)

    # 3) Structural importance (centrality in current correlation)
    importance = np.mean(np.abs(np.asarray(corr_recent, dtype=float)), axis=1)
    importance = np.nan_to_num(importance, nan=0.0)
    imp_max = float(np.max(importance)) + 1e-12
    imp_norm = np.clip(importance / imp_max, 0.0, 1.0)

    # Combined score: weighted sum
    combined = (
        drift_weight * drift_norm
        + causal_weight * causal_norm
        + importance_weight * imp_norm
    )
    combined = np.clip(combined, 0.0, 1.0)

    # Build driver_scores and top_drivers
    driver_scores = {names[i]: float(combined[i]) for i in range(n)}
    order = np.argsort(-combined)
    top_drivers = [names[int(i)] for i in order[:top_k]]

    return {
        "top_drivers": top_drivers,
        "driver_scores": driver_scores,
    }
