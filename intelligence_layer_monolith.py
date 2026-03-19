"""
Neraium SII intelligence layer — single file for copy/paste.
Dependencies: numpy only. Regime library defaults to ./regime_library.json

For correlation / drift / spectral / composite math only (no StructuralEngine),
see core_math_engine_monolith.py (rebuild with this script).

Usage:
    engine = StructuralEngine(baseline_window=50, recent_window=12)
    out = engine.process_frame({
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "s1",
        "asset_id": "a1",
        "sensor_values": {"s1": 0.4, "s2": 0.36, "s3": 0.42},
    })
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


# =============================================================================
# --- regime_store ---
import json
from pathlib import Path
from typing import Any


class RegimeStore:
    def __init__(self, path: str = "regime_library.json"):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"regimes": [], "baselines": {}}

        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"regimes": [], "baselines": {}}

    def save(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

# --- scoring ---
from typing import Mapping

import math


LEGACY_KEYS: dict[str, str] = {
    "drift": "relational_drift",
    "directional": "directional_divergence",
    "causal": "directional_divergence",
    "subsystem": "subsystem_instability",
    "subsystem_max": "subsystem_instability",
}


DEFAULT_COMPONENTS: dict[str, float] = {
    "relational_drift": 0.0,
    "regime_drift": 0.0,
    "spectral": 0.0,
    "directional_divergence": 0.0,
    "entropy": 0.0,
    "subsystem_instability": 0.0,
    "early_warning": 0.0,
}


DEFAULT_WEIGHTS: dict[str, float] = {
    "relational_drift": 1.0,
    "regime_drift": 0.8,
    "spectral": 0.8,
    "directional_divergence": 0.8,
    "entropy": 0.5,
    "subsystem_instability": 0.7,
    "early_warning": 0.6,
}


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def normalize_keys(values: Mapping[str, object]) -> dict[str, float]:
    normalized: dict[str, float] = {}

    def norm_key(key: object) -> str:
        return str(key).strip().lower()

    for key, value in values.items():
        k = norm_key(key)
        if k in DEFAULT_COMPONENTS or k in DEFAULT_WEIGHTS:
            normalized[k] = _coerce_float(value)

    for key, value in values.items():
        k = norm_key(key)
        mapped = LEGACY_KEYS.get(k, k)
        if mapped in normalized:
            continue
        if mapped in DEFAULT_COMPONENTS or mapped in DEFAULT_WEIGHTS:
            normalized[mapped] = _coerce_float(value)

    return normalized


def canonicalize_components(components: Mapping[str, object] | None = None) -> dict[str, float]:
    result = dict(DEFAULT_COMPONENTS)

    if not components:
        # Backward-compatible aliases
        result["drift"] = result["relational_drift"]
        result["directional"] = result["directional_divergence"]
        return result

    normalized = normalize_keys(components)
    result.update(normalized)
    # Backward-compatible aliases (tests and older callers may use these keys)
    result["drift"] = result.get("relational_drift", 0.0)
    result["directional"] = result.get("directional_divergence", 0.0)
    return result


def canonicalize_weights(weights: Mapping[str, object] | None = None) -> dict[str, float]:
    result = dict(DEFAULT_WEIGHTS)

    if not weights:
        return result

    normalized = normalize_keys(weights)
    result.update(normalized)
    return result


def available_components(
    components: Mapping[str, object],
    weights: Mapping[str, object] | None = None,
) -> dict[str, tuple[float, float]]:
    canonical_components = canonicalize_components(components)
    canonical_weights = canonicalize_weights(weights)

    active: dict[str, tuple[float, float]] = {}
    for key, value in canonical_components.items():
        weight = canonical_weights.get(key, 0.0)
        if weight > 0.0:
            active[key] = (value, weight)

    return active


# Winsorization cap for normalized composite (keeps scale compatible with decision thresholds)
DEFAULT_WINSORIZE_CAP = 3.0


def _winsorize(value: float, low: float = 0.0, high: float = DEFAULT_WINSORIZE_CAP) -> float:
    """Clip value to [low, high] for robust composite scoring."""
    if math.isnan(value) or math.isinf(value):
        return low
    return max(low, min(high, value))


def composite_instability_score(
    components: Mapping[str, object],
    weights: Mapping[str, object] | None = None,
    normalize: bool = True,
) -> float:
    active = available_components(components, weights)

    if not active:
        return 0.0

    weighted_sum = sum(value * weight for value, weight in active.values())
    weight_sum = sum(weight for _, weight in active.values())

    if not normalize or weight_sum <= 0.0:
        return float(weighted_sum)

    return float(weighted_sum / weight_sum)


def composite_instability_score_normalized(
    components: Mapping[str, object],
    weights: Mapping[str, object] | None = None,
    winsorize_cap: float = DEFAULT_WINSORIZE_CAP,
) -> float:
    """
    Composite instability score with robust per-component winsorization.
    Each component value is clipped to [0, winsorize_cap] before weighted average,
    so the result stays on a scale compatible with existing decision thresholds.
    """
    active = available_components(components, weights)

    if not active:
        return 0.0

    weighted_sum = sum(
        _winsorize(value, high=winsorize_cap) * weight for value, weight in active.values()
    )
    weight_sum = sum(weight for _, weight in active.values())

    if weight_sum <= 0.0:
        return 0.0

    return float(weighted_sum / weight_sum)

# --- geometry ---
from typing import Any

import numpy as np


ArrayLike = Any


def _as_2d_array(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array-like structure")
    if array.shape[0] < 2:
        raise ValueError("At least two observations are required")
    return array


def normalize_window(observations: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window z-score normalization with missing and zero-variance guards."""
    data = _as_2d_array(observations)
    means = np.nanmean(data, axis=0)
    means = np.nan_to_num(means, nan=0.0)

    centered = data - means
    std = np.nanstd(data, axis=0)
    std = np.nan_to_num(std, nan=0.0)
    safe_std = np.where(std <= 1e-12, 1.0, std)

    z = centered / safe_std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z, means, std


def correlation_matrix(observations: ArrayLike) -> np.ndarray:
    """Compute correlation geometry R_t from row-wise observations."""
    data = _as_2d_array(observations)
    corr = np.corrcoef(data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def structural_drift(current_corr: ArrayLike, baseline_corr: ArrayLike, norm: str = "fro") -> float:
    """Baseline-relative structural drift D_t = ||R_t - R_0||."""
    current = np.asarray(current_corr, dtype=float)
    baseline = np.asarray(baseline_corr, dtype=float)
    if current.shape != baseline.shape or current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("current and baseline correlation matrices must be equally-shaped square matrices")

    delta = np.nan_to_num(current - baseline, nan=0.0, posinf=0.0, neginf=0.0)
    if norm == "mae":
        return float(np.mean(np.abs(delta)))
    if norm == "fro":
        return float(np.linalg.norm(delta, ord="fro"))
    raise ValueError("norm must be 'fro' or 'mae'")


def signal_structural_importance(corr: ArrayLike) -> np.ndarray:
    matrix = np.asarray(corr, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation matrix must be square")
    n = matrix.shape[0]
    return np.mean(np.abs(matrix), axis=1) if n else np.array([], dtype=float)


def relational_structure(corr: ArrayLike) -> dict[str, np.ndarray | float]:
    """Extract compact relational observables from a correlation matrix."""
    matrix = np.asarray(corr, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation matrix must be square")

    magnitude = np.abs(matrix)
    np.fill_diagonal(magnitude, 0.0)
    centrality = magnitude.mean(axis=1)
    relational_energy = float(magnitude.sum() / max(matrix.shape[0] * (matrix.shape[0] - 1), 1))

    return {
        "centrality": centrality,
        "relational_energy": relational_energy,
    }


# --- early_warning ---
from typing import Any

import numpy as np


ArrayLike = Any


def early_warning_metrics(observations: ArrayLike) -> dict[str, float]:
    data = np.asarray(observations, dtype=float)
    if data.ndim != 2:
        raise ValueError("Expected 2D observations")
    safe = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    variances = np.var(safe, axis=0)
    if safe.shape[0] < 2:
        lag1 = np.zeros(safe.shape[1], dtype=float)
    else:
        lag1 = []
        for idx in range(safe.shape[1]):
            lag1.append(np.corrcoef(safe[:-1, idx], safe[1:, idx])[0, 1])
        lag1 = np.nan_to_num(np.array(lag1, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "variance": float(np.mean(variances) if variances.size else 0.0),
        "lag1_autocorrelation": float(np.mean(lag1) if lag1.size else 0.0),
    }


# --- forecasting ---
from collections.abc import Sequence

import numpy as np


def instability_trend_regression(series: Sequence[float]) -> float:
    """Linear trend estimate for instability series."""
    y = np.asarray(list(series), dtype=float)

    if len(y) < 5:
        return 0.0

    x = np.arange(len(y), dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def time_to_instability_regression(
    series: Sequence[float],
    threshold: float = 1.5,
) -> float | None:
    """Project time to threshold crossing from linear regression."""
    y = np.asarray(list(series), dtype=float)

    if len(y) < 5:
        return None

    x = np.arange(len(y), dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

    current = float(y[-1])
    if slope <= 0:
        return None

    tti = (float(threshold) - current) / float(slope)
    return float(max(0.0, tti))


# Backward-compatible wrappers
def instability_trend(series: Sequence[float]) -> float:
    return instability_trend_regression(series)


def time_to_instability(series: Sequence[float], threshold: float = 1.5) -> float | None:
    return time_to_instability_regression(series, threshold=threshold)

# --- forecast_models ---
from collections.abc import Sequence
from typing import Optional

import numpy as np


def _fit_ar1(series: Sequence[float]) -> tuple[float, float] | None:
    """
    Fit a simple AR(1) model: y_t = a + b*y_{t-1}.
    Returns (a, b) or None if insufficient data.
    """
    y = np.asarray(list(series), dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 5:
        return None

    x_prev = y[:-1]
    y_curr = y[1:]

    # Least squares for y_curr = a + b*x_prev
    A = np.vstack([np.ones_like(x_prev), x_prev]).T
    (a, b), *_ = np.linalg.lstsq(A, y_curr, rcond=None)
    a = float(a)
    b = float(b)
    return a, b


def forecast_next(series: Sequence[float]) -> float:
    """One-step AR(1) forecast."""
    fit = _fit_ar1(series)
    y = np.asarray(list(series), dtype=float)
    y = y[np.isfinite(y)]
    if fit is None or y.size == 0:
        return float(y[-1]) if y.size else 0.0

    a, b = fit
    return float(a + b * float(y[-1]))


def time_to_threshold_ar1(series: Sequence[float], threshold: float = 1.5, max_steps: int = 200) -> Optional[float]:
    """
    Estimate time-to-threshold by iterating AR(1) forward until prediction >= threshold.
    Returns None if never crossed within max_steps or if AR(1) is not fit.
    """
    fit = _fit_ar1(series)
    y = np.asarray(list(series), dtype=float)
    y = y[np.isfinite(y)]
    if fit is None or y.size == 0:
        return None

    a, b = fit
    current = float(y[-1])
    if current >= threshold:
        return 0.0

    # Iterate until crossing or stability fails
    t = 0
    next_val = current
    while t < max_steps:
        next_val = float(a + b * next_val)
        t += 1
        if next_val >= threshold:
            return float(t)

        # If AR(1) is effectively decaying and won't cross, bail early
        if b <= 0:
            # Sequence will oscillate/decay without upward trend; safe to stop
            break

    return None


__all__ = ["forecast_next", "time_to_threshold_ar1"]



# --- entropy ---
from typing import Any

import numpy as np


ArrayLike = Any


def interaction_entropy(matrix: ArrayLike) -> float:
    values = np.abs(np.asarray(matrix, dtype=float)).ravel()
    total = float(values.sum())
    if total <= 0:
        return 0.0

    probs = values / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


# --- spectral ---
from typing import Any

import numpy as np


ArrayLike = Any


def eigendecomposition(matrix: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Matrix must be square")
    if values.size == 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)
    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(safe_values)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def spectral_radius(matrix: ArrayLike) -> float:
    eigenvalues, _ = eigendecomposition(matrix)
    return float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0


def spectral_gap(matrix: ArrayLike) -> float:
    eigenvalues, _ = eigendecomposition(matrix)
    if eigenvalues.size < 2:
        return 0.0
    return float(eigenvalues[0] - eigenvalues[1])


def dominant_mode_loading(matrix: ArrayLike) -> dict[str, list[float] | float]:
    eigenvalues, eigenvectors = eigendecomposition(matrix)
    if eigenvalues.size == 0:
        return {"dominant_eigenvalue": 0.0, "dominant_eigenvector": []}
    return {
        "dominant_eigenvalue": float(eigenvalues[0]),
        "dominant_eigenvector": [float(v) for v in eigenvectors[:, 0]],
    }


# --- graph ---
from typing import Any

import numpy as np


ArrayLike = Any


def thresholded_adjacency(corr: ArrayLike, threshold: float = 0.6) -> np.ndarray:
    matrix = np.asarray(corr, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation matrix must be square")
    adj = (np.abs(matrix) >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def graph_metrics(adjacency: ArrayLike, corr: ArrayLike | None = None) -> dict[str, float]:
    adj = np.asarray(adjacency, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    n = adj.shape[0]
    degree = adj.sum(axis=1)
    max_edges = n * (n - 1)
    density = float(adj.sum() / max_edges) if max_edges else 0.0

    triangles = float(np.trace(np.linalg.matrix_power(adj, 3)) / 6.0) if n >= 3 else 0.0
    triplets = float(np.sum(degree * (degree - 1)) / 2.0)
    clustering = float((3.0 * triangles / triplets) if triplets > 0 else 0.0)

    connected = 0.0
    if n:
        reachability = np.linalg.matrix_power(adj + np.eye(n), max(n - 1, 1))
        connected = float(np.all(reachability > 0))

    metrics = {
        "mean_degree": float(np.mean(degree) if n else 0.0),
        "density": density,
        "clustering": clustering,
        "connectivity": connected,
    }

    if corr is not None:
        corr_matrix = np.asarray(corr, dtype=float)
        if corr_matrix.shape != adj.shape:
            raise ValueError("corr must have the same shape as adjacency")
        metrics["mean_absolute_connectivity"] = float(np.mean(np.abs(corr_matrix - np.eye(n))) if n else 0.0)

    return metrics


# --- directional ---
from typing import Any

import numpy as np


ArrayLike = Any


def lagged_correlation_matrix(observations: ArrayLike, lag: int = 1) -> np.ndarray:
    """Structural directional proxy C_ij = corr(x_i(t), x_j(t+lag))."""
    data = np.asarray(observations, dtype=float)
    if data.ndim != 2:
        raise ValueError("Expected 2D observations")
    if lag <= 0 or data.shape[0] <= lag:
        raise ValueError("Lag must be positive and smaller than the number of observations")

    current = np.nan_to_num(data[:-lag], nan=0.0, posinf=0.0, neginf=0.0)
    future = np.nan_to_num(data[lag:], nan=0.0, posinf=0.0, neginf=0.0)
    n_features = data.shape[1]
    result = np.zeros((n_features, n_features), dtype=float)

    for i in range(n_features):
        for j in range(n_features):
            result[i, j] = np.corrcoef(current[:, i], future[:, j])[0, 1]

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def directional_metrics(matrix: ArrayLike) -> dict[str, float]:
    lagged = np.asarray(matrix, dtype=float)
    if lagged.ndim != 2 or lagged.shape[0] != lagged.shape[1]:
        raise ValueError("Directional matrix must be square")

    abs_matrix = np.abs(lagged)
    energy = float(np.mean(abs_matrix))
    asymmetry = float(np.mean(np.abs(lagged - lagged.T)))
    divergence = float(energy * (1.0 + asymmetry))

    return {
        "causal_energy": energy,
        "causal_asymmetry": asymmetry,
        "causal_divergence": divergence,
        "divergence": divergence,
        "proxy_only": 1.0,
    }


# --- causal_proxy (casual.py) ---
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

# --- causal_graph ---
import numpy as np

# Compatibility module: the codebase contains `casual_graph` (typo) but other
# modules import `causal_graph`. Provide the expected API.


def causal_adjacency(C: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    A = np.array(C, dtype=float, copy=True)
    A[np.abs(A) < threshold] = 0.0
    np.fill_diagonal(A, 0.0)
    return A


def causal_graph_metrics(C: np.ndarray, threshold: float = 0.1) -> dict[str, float | list[int]]:
    A = causal_adjacency(C, threshold=threshold)

    if A.size == 0:
        return {
            "density": 0.0,
            "asymmetry": 0.0,
            "dominant_sources": [],
        }

    density = float(np.mean(np.abs(A) > 0))
    asymmetry = float(np.mean(np.abs(A - A.T)))
    outbound = np.sum(np.abs(A), axis=1)
    dominant_sources = np.argsort(-outbound)[:3].tolist()

    return {
        "density": density,
        "asymmetry": asymmetry,
        "dominant_sources": dominant_sources,
    }


__all__ = ["causal_graph_metrics"]



# --- causal_attribution ---
"""
Causal attribution layer: observational ranking of signals/nodes that contribute
most to current instability. Read-only; no interventions or operational directives.
"""
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


# --- subsystems ---
import numpy as np
from numpy.linalg import eigvals, eig


def spectral_clustering_subsystems(corr: np.ndarray, k: int = 3) -> list[list[int]]:
    """
    Lightweight subsystem discovery using eigenvector embedding.

    This is a practical approximation, not a full clustering framework.
    """
    corr = np.asarray(corr, dtype=float)

    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] < 2:
        return []

    vals, vecs = eig(corr)
    idx = np.argsort(-np.abs(vals))[: min(k, corr.shape[0])]
    embedding = np.real(vecs[:, idx])

    labels = np.argmax(np.abs(embedding), axis=1)

    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    return list(clusters.values())


def discover_subsystems(corr: np.ndarray, k: int = 3) -> list[list[int]]:
    """
    Backward-compatible subsystem discovery wrapper.

    The test suite expects:
    - for a single-sensor correlation matrix, no subsystems are discovered.
    """
    clusters = spectral_clustering_subsystems(corr, k=k)
    # Only subsystems with at least 2 sensors are meaningful for spectral measures.
    return [cluster for cluster in clusters if len(cluster) >= 2]


def subsystem_spectral_measures(corr: np.ndarray, k: int = 3) -> dict[str, object]:
    """
    Compute subsystem-local dominant spectral instability.

    Returns both the discovered clusters and the max subsystem instability.
    """
    corr = np.asarray(corr, dtype=float)

    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] < 2:
        return {
            "clusters": [],
            "subsystem_instability": 0.0,
            "max_instability": 0.0,
            "subsystem_count": 0,
        }

    clusters = spectral_clustering_subsystems(corr, k=k)
    instabilities: list[float] = []

    for cluster in clusters:
        if len(cluster) < 2:
            continue

        sub = corr[np.ix_(cluster, cluster)]
        vals = eigvals(sub)
        instabilities.append(float(np.max(np.abs(vals))))

    max_instability = float(max(instabilities)) if instabilities else 0.0

    return {
        "clusters": clusters,
        "subsystem_instability": max_instability,
        "max_instability": max_instability,
        "subsystem_count": len(clusters),
    }

# --- regime ---
from typing import Any

import numpy as np


def build_regime_signature(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Build a regime signature from per-signal mean and std."""
    return np.concatenate([np.asarray(mean, dtype=float), np.asarray(std, dtype=float)])


def regime_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two regime signatures."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def assign_regime(signature: np.ndarray, regimes: list[dict[str, Any]]) -> dict[str, float | str] | None:
    """Find the nearest known regime."""
    if not regimes:
        return None

    distances: list[tuple[float, str]] = []
    sig = np.asarray(signature, dtype=float)
    for regime in regimes:
        centroid = np.asarray(regime["signature"], dtype=float)
        # Sensor sets/window configuration can change signature dimensionality over
        # time. Skip incompatible regimes instead of failing with broadcasting.
        if centroid.shape != sig.shape:
            continue
        distances.append((regime_distance(sig, centroid), str(regime["name"])))

    distances.sort(key=lambda x: x[0])
    if not distances:
        return None
    nearest_distance, nearest_name = distances[0]
    return {"name": nearest_name, "distance": float(nearest_distance)}


def update_regime_library(
    signature: np.ndarray,
    regimes: list[dict[str, Any]],
    threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Update the regime library.

    If no regime exists, bootstrap one.
    If the nearest regime is farther than threshold, add a new regime.
    """
    if not regimes:
        regimes.append({"name": "regime_0", "signature": signature.tolist()})
        return regimes

    assigned = assign_regime(signature, regimes)
    if assigned is None or float(assigned["distance"]) > threshold:
        regimes.append(
            {
                "name": f"regime_{len(regimes)}",
                "signature": signature.tolist(),
            }
        )

    return regimes

# --- data_quality ---
"""
Data quality gating before analytics.
Prevents poor data quality from masquerading as structural instability.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DataQualityReport:
    """Result of data quality checks on baseline and recent windows."""

    missingness_rate: float = 0.0
    stale_sensors: List[str] = field(default_factory=list)
    flatlined_sensors: List[str] = field(default_factory=list)
    timestamp_irregularity: float = 0.0
    sensor_churn: float = 0.0
    outlier_burst_density: float = 0.0
    statuses: List[str] = field(default_factory=list)
    sensor_coverage: float = 1.0
    variability_coverage: float = 1.0
    valid_signal_count: int = 0
    total_sensors: int = 0
    gate_passed: bool = True

    def to_dict(self) -> dict:
        return {
            "missingness_rate": self.missingness_rate,
            "stale_sensors": self.stale_sensors,
            "flatlined_sensors": self.flatlined_sensors,
            "timestamp_irregularity": self.timestamp_irregularity,
            "sensor_churn": self.sensor_churn,
            "outlier_burst_density": self.outlier_burst_density,
            "statuses": self.statuses,
            "sensor_coverage": self.sensor_coverage,
            "variability_coverage": self.variability_coverage,
            "valid_signal_count": self.valid_signal_count,
            "total_sensors": self.total_sensors,
            "gate_passed": self.gate_passed,
        }


# Status constants for downstream gating
DATA_QUALITY_LIMITED = "DATA_QUALITY_LIMITED"
INSUFFICIENT_VARIABILITY = "INSUFFICIENT_VARIABILITY"
LOW_SENSOR_COVERAGE = "LOW_SENSOR_COVERAGE"
TIMESTAMP_IRREGULAR = "TIMESTAMP_IRREGULAR"
HIGH_SENSOR_CHURN = "HIGH_SENSOR_CHURN"

# Default thresholds (configurable via caller)
DEFAULT_MIN_SENSORS = 2
DEFAULT_MAX_MISSINGNESS = 0.5
DEFAULT_MIN_VARIABILITY_COVERAGE = 0.25
DEFAULT_FLATLINE_STD_THRESHOLD = 1e-12
DEFAULT_STALE_NAN_RATE = 0.8
DEFAULT_MAX_CHURN = 0.8
DEFAULT_MAX_IRREGULARITY = 0.5


def compute_data_quality(
    baseline_matrix: np.ndarray,
    recent_matrix: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    timestamps_baseline: Optional[List[float]] = None,
    timestamps_recent: Optional[List[float]] = None,
    *,
    min_sensors: int = DEFAULT_MIN_SENSORS,
    max_missingness: float = DEFAULT_MAX_MISSINGNESS,
    min_variability_coverage: float = DEFAULT_MIN_VARIABILITY_COVERAGE,
    flatline_std_threshold: float = DEFAULT_FLATLINE_STD_THRESHOLD,
    stale_nan_rate: float = DEFAULT_STALE_NAN_RATE,
    max_churn: float = DEFAULT_MAX_CHURN,
    max_irregularity: float = DEFAULT_MAX_IRREGULARITY,
) -> DataQualityReport:
    """
    Compute data quality metrics and statuses.
    Returns a report; gate_passed is False if quality is too poor for production analytics.
    """
    baseline = np.asarray(baseline_matrix, dtype=float)
    recent = np.asarray(recent_matrix, dtype=float)
    if baseline.ndim != 2 or recent.ndim != 2:
        return DataQualityReport(
            statuses=[DATA_QUALITY_LIMITED],
            gate_passed=False,
        )
    n_sensors = baseline.shape[1]
    if sensor_names is None:
        sensor_names = [f"s_{i}" for i in range(n_sensors)]
    else:
        sensor_names = list(sensor_names)[:n_sensors]

    report = DataQualityReport(total_sensors=n_sensors)

    # Missingness per sensor (recent window)
    nan_count = np.sum(np.isnan(recent), axis=0)
    total_obs = recent.shape[0] * recent.shape[1]
    report.missingness_rate = float(np.sum(nan_count) / total_obs) if total_obs else 0.0

    # Stale sensors: mostly NaN in recent
    obs_per_sensor = recent.shape[0]
    stale = (nan_count / max(1, obs_per_sensor)) >= stale_nan_rate
    report.stale_sensors = [sensor_names[i] for i in range(n_sensors) if i < len(sensor_names) and stale[i]]

    # Flatlined: near-zero std in recent (using nanstd)
    safe_recent = np.nan_to_num(recent, nan=0.0)
    std_recent = np.nanstd(recent, axis=0)
    std_recent = np.nan_to_num(std_recent, nan=0.0)
    flatlined = std_recent <= flatline_std_threshold
    report.flatlined_sensors = [sensor_names[i] for i in range(n_sensors) if i < len(sensor_names) and flatlined[i]]

    # Valid mask: nonzero variance in recent or baseline
    std_baseline = np.nanstd(baseline, axis=0)
    std_baseline = np.nan_to_num(std_baseline, nan=0.0)
    valid_mask = (std_recent > flatline_std_threshold) | (std_baseline > flatline_std_threshold)
    report.valid_signal_count = int(np.sum(valid_mask))

    # Variability coverage: fraction of sensors with usable variance
    report.variability_coverage = report.valid_signal_count / max(1, n_sensors)
    if report.variability_coverage < min_variability_coverage:
        report.statuses.append(INSUFFICIENT_VARIABILITY)
    if report.valid_signal_count < min_sensors:
        report.statuses.append(LOW_SENSOR_COVERAGE)
    report.sensor_coverage = 1.0 - (len(report.stale_sensors) / max(1, n_sensors))
    if report.sensor_coverage < 0.5:
        report.statuses.append(LOW_SENSOR_COVERAGE)
    if report.missingness_rate > max_missingness:
        report.statuses.append(DATA_QUALITY_LIMITED)

    # Timestamp irregularity (if timestamps provided)
    if timestamps_baseline is not None and len(timestamps_baseline) >= 2:
        ts = np.array(timestamps_baseline, dtype=float)
        gaps = np.diff(ts)
        if np.all(gaps > 0):
            cv = float(np.std(gaps) / (np.mean(gaps) + 1e-12))
            report.timestamp_irregularity = min(1.0, cv)
            if report.timestamp_irregularity > max_irregularity:
                report.statuses.append(TIMESTAMP_IRREGULAR)
    if timestamps_recent is not None and len(timestamps_recent) >= 2 and TIMESTAMP_IRREGULAR not in report.statuses:
        ts = np.array(timestamps_recent, dtype=float)
        gaps = np.diff(ts)
        if np.all(gaps > 0):
            cv = float(np.std(gaps) / (np.mean(gaps) + 1e-12))
            report.timestamp_irregularity = max(report.timestamp_irregularity, min(1.0, cv))
            if report.timestamp_irregularity > max_irregularity:
                report.statuses.append(TIMESTAMP_IRREGULAR)

    # Sensor churn: Jaccard distance of "valid" sensors between baseline and recent
    valid_baseline = (std_baseline > flatline_std_threshold).astype(int)
    valid_recent = (std_recent > flatline_std_threshold).astype(int)
    intersection = np.sum(valid_baseline & valid_recent)
    union = np.sum(valid_baseline | valid_recent)
    report.sensor_churn = 1.0 - (intersection / union) if union else 0.0
    if report.sensor_churn > max_churn:
        report.statuses.append(HIGH_SENSOR_CHURN)

    # Outlier burst density: proportion of consecutive runs of robust z > 2
    if safe_recent.size >= 3:
        med = np.median(safe_recent)
        mad = np.median(np.abs(safe_recent - med)) + 1e-12
        z = np.abs((safe_recent - med) / mad)
        over = (z > 2.0).astype(float)
        runs = np.diff(over, axis=0)
        burst_starts = np.sum(runs == 1) if runs.size else 0
        report.outlier_burst_density = burst_starts / max(1, recent.shape[0] * recent.shape[1])
    else:
        report.outlier_burst_density = 0.0

    if not report.statuses:
        report.statuses = []
    report.gate_passed = (
        report.valid_signal_count >= min_sensors
        and report.missingness_rate <= max_missingness
        and report.variability_coverage >= min_variability_coverage
    )
    return report


def data_quality_summary(report: DataQualityReport) -> dict:
    """
    Compact summary for experiment-friendly output and downstream confidence.
    Includes active_sensor_count, missing_sensor_count, and degradation flags.
    """
    # Count sensors that are stale or flatlined (union so we count each once)
    stale_set = set(report.stale_sensors)
    flat_set = set(report.flatlined_sensors)
    missing_sensor_count = len(stale_set | flat_set)
    return {
        "gate_passed": report.gate_passed,
        "missingness_rate": report.missingness_rate,
        "valid_signal_count": report.valid_signal_count,
        "total_sensors": report.total_sensors,
        "stale_sensor_count": len(report.stale_sensors),
        "flatlined_sensor_count": len(report.flatlined_sensors),
        "missing_sensor_count": missing_sensor_count,
        "statuses": list(report.statuses),
        "sensor_coverage": report.sensor_coverage,
        "variability_coverage": report.variability_coverage,
    }


# Threshold above which we do not attempt fallback analytics (too much data missing)
DEFAULT_DEGRADED_MAX_MISSINGNESS = 0.85


def should_use_degraded_analytics(
    report: DataQualityReport,
    *,
    max_missingness_for_fallback: float = DEFAULT_DEGRADED_MAX_MISSINGNESS,
    min_sensors_for_fallback: int = 1,
) -> bool:
    """
    True if we should still produce meaningful output with degraded confidence
    (e.g. when gate failed but data is not catastrophically missing).
    """
    if report.gate_passed:
        return False
    if report.missingness_rate > max_missingness_for_fallback:
        return False
    if report.valid_signal_count < min_sensors_for_fallback:
        return False
    return report.total_sensors >= 1


def impute_missing_simple(
    matrix: np.ndarray,
    *,
    axis: int = 0,
    method: str = "column_mean",
) -> np.ndarray:
    """
    Simple observational imputation for missing values so analytics can still run
    with degraded confidence. Use only when gate failed but degraded path is allowed.

    method: "column_mean" fills NaNs with column mean; "zero" fills with 0.
    """
    data = np.asarray(matrix, dtype=float, copy=True)
    if not np.any(np.isnan(data)):
        return data
    if method == "zero":
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if method == "column_mean":
        if axis == 0:
            col_mean = np.nanmean(data, axis=0)
            col_mean = np.nan_to_num(col_mean, nan=0.0)
            for j in range(data.shape[1]):
                mask = np.isnan(data[:, j])
                if np.any(mask):
                    data[mask, j] = col_mean[j]
        else:
            row_mean = np.nanmean(data, axis=1)
            row_mean = np.nan_to_num(row_mean, nan=0.0)
            for i in range(data.shape[0]):
                mask = np.isnan(data[i, :])
                if np.any(mask):
                    data[i, mask] = row_mean[i]
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


# --- decision_layer ---
from typing import Any


def _risk_level(score: float) -> str:
    if score >= 2.5:
        return "HIGH"
    if score >= 1.5:
        return "ELEVATED"
    if score >= 0.7:
        return "MODERATE"
    return "LOW"


def _signal_strength(score: float, trend: float) -> str:
    if score >= 2.5 and trend > 0:
        return "high"
    if score >= 1.5:
        return "medium"
    return "low"


def _confidence(components: dict[str, float]) -> str:
    active = [v for v in components.values() if abs(v) > 1e-6]

    if len(active) >= 5:
        return "high"
    if len(active) >= 3:
        return "medium"
    return "low"


def _confidence_categorical_from_score(score: float) -> str:
    """Map a [0, 1] confidence score to categorical for backward compatibility."""
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _phase(score: float, trend: float) -> str:
    if score < 0.7:
        return "stable"
    if trend > 0.05:
        return "degrading"
    if trend < -0.05:
        return "recovering"
    return "transitional"


def _interpret_state(
    relational_drift: float,
    regime_drift: float,
    directional: float,
    spectral: float,
    early_warning: float,
    entropy: float,
    trend: float,
    persistence: dict[str, float] | None,
) -> str:
    """
    Interpret structural state with clear separation of conditions.

    - REGIME_SHIFT_OBSERVED: relational geometry has moved into a different regime
      without strong directional/coupling breakdown or sustained instability.
    - COUPLING_INSTABILITY_OBSERVED: directional/interaction (and spectral) breakdown
      dominates; depends more strongly on directional + spectral evidence.
    - STRUCTURAL_INSTABILITY_OBSERVED: relational drift + entropy + regime drift
      with sustained, multi-indicator confirmation.
    """
    persistence = persistence or {}
    history_len = float(persistence.get("history_len", 0.0))
    consecutive_elevated = float(persistence.get("consecutive_elevated", 0.0))
    consecutive_high = float(persistence.get("consecutive_high", 0.0))
    rolling_mean = float(persistence.get("rolling_mean", 0.0))

    # Warmup/hysteresis: avoid jumping straight into strong instability classes.
    if history_len < 8:
        if relational_drift > 1.2:
            return "REGIME_SHIFT_OBSERVED"
        return "NOMINAL_STRUCTURE"

    motion = relational_drift > 1.2
    regime_departure = regime_drift >= 0.8
    # Coupling instability: depends more strongly on directional and spectral interaction breakdown.
    directional_breakdown = directional > 1.0
    spectral_breakdown = spectral > 1.2
    coupling_instability = directional_breakdown or spectral_breakdown

    bounded_persistence = consecutive_high < 2 and consecutive_elevated < 4 and rolling_mean < 2.0
    no_degradation_trend = abs(trend) <= 0.06
    sustained = consecutive_high >= 2 or consecutive_elevated >= 5 or rolling_mean >= 2.2
    # Slightly lower bar for coupling: directional/spectral breakdown can persist with less elevation.
    sustained_coupling = consecutive_high >= 1 or consecutive_elevated >= 3 or rolling_mean >= 1.7

    # Coupling instability first: sustained directional/spectral breakdown (interaction-focused).
    if coupling_instability and sustained_coupling:
        return "COUPLING_INSTABILITY_OBSERVED"

    # Regime shift only: structure moved, no strong coupling/directional breakdown, bounded.
    if motion and not coupling_instability and bounded_persistence and no_degradation_trend:
        return "REGIME_SHIFT_OBSERVED"

    # Constrained coherence: motion with correction-like activity but no sustained breakdown.
    correction_present = early_warning > 0.9 or coupling_instability
    if motion and correction_present and bounded_persistence and no_degradation_trend:
        return "COHERENCE_UNDER_CONSTRAINT"

    # Structural instability: depends more strongly on relational drift + entropy + regime drift.
    entropy_elevated = entropy > 0.8
    structural_evidence = (motion and regime_departure) or (motion and entropy_elevated)
    multi_indicator = structural_evidence or (regime_departure and entropy_elevated) or (
        motion and coupling_instability and early_warning > 1.1
    )
    degrading = trend > 0.06

    if sustained and multi_indicator and (degrading or regime_departure):
        return "STRUCTURAL_INSTABILITY_OBSERVED"

    return "NOMINAL_STRUCTURE"


def _operator_message(
    state: str,
    trend: float,
    time_to_instability: float | None,
) -> str:
    """
    Strictly observational language.
    No control, no directives, no operational commands.
    """

    if state == "STRUCTURAL_INSTABILITY_OBSERVED":
        if time_to_instability is not None:
            return (
                "Observed structural relationships are diverging from previously seen "
                "system patterns. Current configuration exhibits elevated instability "
                "characteristics under current analysis, with continued progression "
                f"projected over approximately {round(time_to_instability, 1)} time units."
            )
        return (
            "Observed structural relationships are diverging from previously seen "
            "system patterns. Current configuration exhibits elevated instability "
            "characteristics under current analysis."
        )

    if state == "REGIME_SHIFT_OBSERVED":
        return (
            "Observed system relationships indicate a transition into a different "
            "structural regime. Current behavior differs from prior baseline but "
            "remains internally consistent under current analysis."
        )

    if state == "COUPLING_INSTABILITY_OBSERVED":
        return (
            "Observed coupling and directional interactions between signals show "
            "elevated variability. System coordination patterns appear less stable "
            "than baseline under current analysis."
        )

    if state == "COHERENCE_UNDER_CONSTRAINT":
        return (
            "Observed structure is moving under apparent correction activity. "
            "Current relationships remain bounded and internally coherent under current analysis, "
            "without clear evidence of coordinated structural breakdown."
        )

    if trend > 0.0:
        return (
            "Observed structural patterns remain broadly consistent with previously "
            "seen behavior, with limited upward movement in current instability signals."
        )

    return (
        "Observed structural patterns are consistent with previously seen baseline "
        "behavior under current analysis. No significant structural deviation detected."
    )


def decision_output(
    composite_score: float,
    components: dict[str, Any],
    forecast: dict[str, Any],
    *,
    confidence_score: float | None = None,
    classification_stability: float | None = None,
) -> dict[str, Any]:
    """
    Convert structural analytics into operator-safe decision output.

    This layer is observational only.
    It does not prescribe action or imply control authority.
    """
    relational_drift = float(components.get("relational_drift", 0.0))
    regime_drift = float(components.get("regime_drift", 0.0))
    directional = float(components.get("directional_divergence", 0.0))
    spectral = float(components.get("spectral", 0.0))
    early_warning = float(components.get("early_warning", 0.0))
    entropy = float(components.get("entropy", 0.0))

    trend = float(forecast.get("trend", 0.0))
    persistence = forecast.get("persistence") if isinstance(forecast.get("persistence"), dict) else None
    time_to_instability = forecast.get("ar1_time_to_instability")
    if time_to_instability is None:
        time_to_instability = forecast.get("time_to_instability")

    state = _interpret_state(
        relational_drift=relational_drift,
        regime_drift=regime_drift,
        directional=directional,
        spectral=spectral,
        early_warning=early_warning,
        entropy=entropy,
        trend=trend,
        persistence=persistence,
    )

    risk_level = _risk_level(composite_score)
    signal_strength = _signal_strength(composite_score, trend)
    conf_val = confidence_score if confidence_score is not None else components.get("_confidence_score")
    if conf_val is not None:
        try:
            confidence = _confidence_categorical_from_score(float(conf_val))
        except (TypeError, ValueError):
            confidence = _confidence({k: v for k, v in components.items() if not k.startswith("_")})
    else:
        confidence = _confidence({k: v for k, v in components.items() if not k.startswith("_")})
    phase = _phase(composite_score, trend)

    signal_emitted = composite_score >= 1.5 or state in {
        "REGIME_SHIFT_OBSERVED",
        "STRUCTURAL_INSTABILITY_OBSERVED",
        "COUPLING_INSTABILITY_OBSERVED",
    }

    operator_message = _operator_message(
        state=state,
        trend=trend,
        time_to_instability=time_to_instability,
    )

    out: dict[str, Any] = {
        "phase": phase,
        "risk_level": risk_level,
        "signal_emitted": signal_emitted,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "operator_message": operator_message,
        "interpreted_state": state,
    }
    if classification_stability is not None:
        out["classification_stability"] = round(classification_stability, 4)
    return out


def evaluate_signal(timeseries: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """
    Backward-compatible decision helper used by the test suite.

    This function evaluates a short time-series summary and returns an
    operator-safe decision-like payload. It is observational only and does
    not prescribe control actions.
    """
    if not timeseries:
        return {
            "signal_emitted": False,
            "signal_strength": "low",
            "confidence": "low",
            "operator_message": "No material structural instability detected.",
            "reason": [],
        }

    peak_instability = float(config.get("peak_instability", 1.5))

    phases = [str(row.get("phase", "") or "").lower().strip() for row in timeseries]
    all_stable = bool(phases) and all(p == "stable" for p in phases)

    values = []
    for row in timeseries:
        try:
            values.append(float(row.get("composite_instability", 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)

    latest_instability = float(values[-1])
    max_instability = max(values) if values else 0.0

    # Consistency rule: require a small run of elevated instability at the end
    # of the series, rather than a single noisy spike.
    required_cycles = min(3, len(values))
    high_cut = peak_instability * 0.85

    consecutive_high = 0
    for v in reversed(values):
        if v >= high_cut:
            consecutive_high += 1
        else:
            break

    consistency_ok = consecutive_high >= required_cycles

    # If everything is explicitly stable, keep the message minimal.
    if all_stable:
        return {
            "signal_emitted": False,
            "signal_strength": "low",
            "confidence": "low",
            "operator_message": "No material structural instability detected.",
            "reason": [],
        }

    if max_instability < peak_instability:
        return {
            "signal_emitted": False,
            "signal_strength": "low",
            "confidence": "low",
            "operator_message": "No material structural instability detected.",
            "reason": [],
        }

    if consistency_ok:
        # Strength tier based on how close we are to the configured peak.
        if latest_instability >= peak_instability:
            signal_strength = "high"
        else:
            signal_strength = "medium"

        confidence = "high" if consecutive_high >= required_cycles else "medium"
        return {
            "signal_emitted": True,
            "signal_strength": signal_strength,
            "confidence": confidence,
            "operator_message": (
                "Elevated structural instability characteristics observed; "
                "human review for confirmation is appropriate."
            ),
            "reason": [],
        }

    # Suppress signal when the configured peak was hit but the evidence was not consistent.
    return {
        "signal_emitted": False,
        "signal_strength": "low",
        "confidence": "low",
        "reason": [
            "Signal suppressed because it did not satisfy consistency requirements.",
        ],
        "operator_message": (
            "Observed instability did not satisfy consistency requirements for emission."
        ),
    }

# --- staged_pipeline ---
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def bounded_z(raw: float, mean: float, std: float, cap: float = 4.0) -> float:
    denom = max(1e-6, float(std))
    z = (float(raw) - float(mean)) / denom
    return clamp(float(z), 0.0, cap)


def corr_from_matrix(m: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(m.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def flatten_upper_tri(m: np.ndarray) -> np.ndarray:
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return np.array([], dtype=float)
    idx = np.triu_indices(m.shape[0], k=1)
    return m[idx]


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return float(default)
        return f
    except (TypeError, ValueError):
        return float(default)


@dataclass
class NodeBaselineProfile:
    corr_baseline: np.ndarray | None = None
    corr_drift_mean: float = 0.0
    corr_drift_std: float = 0.1
    relational_mean: float = 0.0
    relational_std: float = 0.1
    temporal_gap_mean: float = 1.0
    temporal_gap_std: float = 0.1
    instability_mean: float = 0.0
    instability_std: float = 0.1
    finalized: bool = False


@dataclass
class RegimeMemory:
    centroids: list[np.ndarray] = field(default_factory=list)
    threshold: float = 2.0

    def nearest_distance(self, signature: np.ndarray) -> float:
        if not self.centroids:
            return 0.0
        dists = [float(np.linalg.norm(signature - c)) for c in self.centroids if c.shape == signature.shape]
        if not dists:
            return 0.0
        return float(min(dists))

    def update(self, signature: np.ndarray) -> float:
        if not self.centroids:
            self.centroids.append(signature.copy())
            return 0.0
        nearest = self.nearest_distance(signature)
        if nearest > self.threshold:
            self.centroids.append(signature.copy())
        return nearest


@dataclass
class NodeRuntime:
    node: str
    variant: str
    sensor_names: list[str]
    baseline_window: int
    recent_window: int
    values_history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=500))
    timestamp_history: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    score_history: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    interpreted_history: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    baseline_profile: NodeBaselineProfile = field(default_factory=NodeBaselineProfile)
    regime_memory: RegimeMemory = field(default_factory=RegimeMemory)
    _baseline_corr_drift: list[float] = field(default_factory=list)
    _baseline_relational: list[float] = field(default_factory=list)
    _baseline_gap: list[float] = field(default_factory=list)
    _baseline_instability: list[float] = field(default_factory=list)

    def push(self, ts: float, sensors: dict[str, float]) -> np.ndarray:
        vec = np.array([safe_float(sensors.get(s), np.nan) for s in self.sensor_names], dtype=float)
        self.values_history.append(vec)
        self.timestamp_history.append(ts)
        return vec

    def recent_matrix(self) -> np.ndarray | None:
        if len(self.values_history) < self.recent_window:
            return None
        m = np.vstack(list(self.values_history)[-self.recent_window :])
        if np.isnan(m).any():
            col_mean = np.nanmean(m, axis=0)
            col_mean = np.nan_to_num(col_mean, nan=0.0)
            inds = np.where(np.isnan(m))
            m[inds] = np.take(col_mean, inds[1])
        return m

    def baseline_matrix(self) -> np.ndarray | None:
        if len(self.values_history) < self.baseline_window:
            return None
        m = np.vstack(list(self.values_history)[: self.baseline_window])
        if np.isnan(m).any():
            col_mean = np.nanmean(m, axis=0)
            col_mean = np.nan_to_num(col_mean, nan=0.0)
            inds = np.where(np.isnan(m))
            m[inds] = np.take(col_mean, inds[1])
        return m

    def recent_timestamps(self) -> list[float] | None:
        if len(self.timestamp_history) < self.recent_window:
            return None
        return list(self.timestamp_history)[-self.recent_window :]

    def baseline_timestamps(self) -> list[float] | None:
        if len(self.timestamp_history) < self.baseline_window:
            return None
        return list(self.timestamp_history)[: self.baseline_window]


class DataQualityStage:
    @staticmethod
    def evaluate(
        baseline: np.ndarray,
        recent: np.ndarray,
        ts_base: list[float] | None,
        ts_recent: list[float] | None,
    ) -> dict[str, float | bool | list[str]]:
        _ = baseline
        _ = ts_base
        n_total = int(recent.size)
        miss = int(np.isnan(recent).sum())
        missingness_rate = float(miss / max(1, n_total))
        std_recent = np.nanstd(recent, axis=0)
        std_recent = np.nan_to_num(std_recent, nan=0.0)
        flatlined = int(np.sum(std_recent <= 1e-12))
        valid_signal_count = int(np.sum(std_recent > 1e-12))
        total_sensors = int(recent.shape[1])
        sensor_coverage = float(valid_signal_count / max(1, total_sensors))
        timestamp_irregularity = 0.0
        if ts_recent is not None and len(ts_recent) >= 3:
            gaps = np.diff(np.array(ts_recent, dtype=float))
            if np.all(gaps > 0):
                timestamp_irregularity = float(np.std(gaps) / (np.mean(gaps) + 1e-9))
        timestamp_irregularity = float(clamp(timestamp_irregularity, 0.0, 1.0))
        statuses: list[str] = []
        if missingness_rate > 0.5:
            statuses.append("DATA_QUALITY_LIMITED")
        if sensor_coverage < 0.5:
            statuses.append("LOW_SENSOR_COVERAGE")
        if timestamp_irregularity > 0.5:
            statuses.append("TIMESTAMP_IRREGULAR")
        gate_passed = bool(missingness_rate <= 0.5 and sensor_coverage >= 0.5 and valid_signal_count >= 2)
        return {
            "missingness_rate": missingness_rate,
            "timestamp_irregularity": timestamp_irregularity,
            "flatlined_sensor_count": flatlined,
            "valid_signal_count": valid_signal_count,
            "total_sensors": total_sensors,
            "sensor_coverage": sensor_coverage,
            "statuses": statuses,
            "gate_passed": gate_passed,
        }


class FeatureExtractionStage:
    @staticmethod
    def extract(baseline: np.ndarray, recent: np.ndarray) -> dict[str, Any]:
        base_mean = np.mean(baseline, axis=0)
        rec_mean = np.mean(recent, axis=0)
        base_std = np.std(baseline, axis=0)
        rec_std = np.std(recent, axis=0)
        corr_base = corr_from_matrix(baseline)
        corr_recent = corr_from_matrix(recent)
        rel_vec_base = flatten_upper_tri(corr_base)
        rel_vec_recent = flatten_upper_tri(corr_recent)
        signature = np.concatenate([rec_mean, rec_std, rel_vec_recent])
        return {
            "base_mean": base_mean,
            "rec_mean": rec_mean,
            "base_std": base_std,
            "rec_std": rec_std,
            "corr_base": corr_base,
            "corr_recent": corr_recent,
            "rel_vec_base": rel_vec_base,
            "rel_vec_recent": rel_vec_recent,
            "signature": signature,
        }


class StructuralDriftStage:
    @staticmethod
    def score(features: dict[str, Any], baseline_profile: NodeBaselineProfile) -> tuple[float, float]:
        corr_recent = features["corr_recent"]
        corr_ref = baseline_profile.corr_baseline if baseline_profile.corr_baseline is not None else features["corr_base"]
        raw = float(np.linalg.norm(corr_recent - corr_ref, ord="fro"))
        normalized = raw
        if baseline_profile.finalized:
            normalized = bounded_z(raw, baseline_profile.corr_drift_mean, baseline_profile.corr_drift_std, cap=4.0)
        return raw, normalized


class RelationalInstabilityStage:
    @staticmethod
    def score(features: dict[str, Any], baseline_profile: NodeBaselineProfile) -> tuple[float, float]:
        delta = features["rel_vec_recent"] - features["rel_vec_base"]
        raw = float(np.mean(np.abs(delta))) if delta.size else 0.0
        normalized = raw
        if baseline_profile.finalized:
            normalized = bounded_z(raw, baseline_profile.relational_mean, baseline_profile.relational_std, cap=4.0)
        return raw, normalized


class TemporalCoherenceStage:
    @staticmethod
    def score(ts_recent: list[float] | None, baseline_profile: NodeBaselineProfile) -> tuple[float, float]:
        if ts_recent is None or len(ts_recent) < 3:
            return 0.0, 0.0
        gaps = np.diff(np.array(ts_recent, dtype=float))
        if not np.all(gaps > 0):
            return 1.0, 3.0
        cv = float(np.std(gaps) / (np.mean(gaps) + 1e-9))
        raw = float(clamp(cv, 0.0, 5.0))
        normalized = raw
        if baseline_profile.finalized:
            normalized = bounded_z(raw, baseline_profile.temporal_gap_mean, baseline_profile.temporal_gap_std, cap=4.0)
        return raw, normalized


class RegimeStage:
    @staticmethod
    def distance(runtime: NodeRuntime, signature: np.ndarray) -> float:
        return runtime.regime_memory.update(signature)


class ConfidenceStage:
    @staticmethod
    def score(
        dq: dict[str, Any],
        component_scores: dict[str, float],
        score_history: deque[float],
        baseline_profile: NodeBaselineProfile,
    ) -> float:
        quality = (1.0 - float(dq["missingness_rate"])) * float(dq["sensor_coverage"]) * (
            1.0 - float(dq["timestamp_irregularity"])
        )
        quality = clamp(quality, 0.0, 1.0)
        if len(score_history) >= 6:
            recent = np.array(list(score_history)[-6:], dtype=float)
            persistence = float(np.mean(recent > 1.0))
            volatility = float(np.std(recent))
        else:
            persistence = 0.0
            volatility = 0.0
        vals = np.array(list(component_scores.values()), dtype=float)
        spread = float(np.std(vals) / (np.mean(vals) + 1e-6)) if vals.size else 1.0
        agreement = clamp(1.0 - 0.5 * spread, 0.0, 1.0)
        if baseline_profile.finalized:
            baseline_std = max(0.05, baseline_profile.instability_std)
            distance_factor = clamp(float(np.mean(vals)) / (2.0 * baseline_std + 1e-6), 0.0, 1.0)
        else:
            distance_factor = 0.3
        conf = (
            0.35 * quality
            + 0.20 * agreement
            + 0.20 * clamp(1.0 - volatility, 0.0, 1.0)
            + 0.15 * clamp(persistence, 0.0, 1.0)
            + 0.10 * distance_factor
        )
        return clamp(conf, 0.0, 1.0)

    @staticmethod
    def categorical(conf: float) -> str:
        if conf >= 0.70:
            return "high"
        if conf >= 0.40:
            return "medium"
        return "low"


class LocalizationStage:
    @staticmethod
    def compute(anomaly_evidence_by_node: dict[str, float]) -> dict[str, float]:
        vals = np.array([max(0.0, float(v)) for v in anomaly_evidence_by_node.values()], dtype=float)
        s = float(np.sum(vals))
        if s <= 1e-9:
            return {k: 0.0 for k in anomaly_evidence_by_node.keys()}
        shares = {k: float(v) / s for k, v in anomaly_evidence_by_node.items()}
        concentration = float(np.max(vals) / (s + 1e-9))
        return {k: clamp(shares[k] * concentration * 2.0, 0.0, 1.0) for k in anomaly_evidence_by_node.keys()}


class DecisionStage:
    @staticmethod
    def interpreted_state(
        structural: float,
        relational: float,
        regime_distance: float,
        temporal_distortion: float,
        localization: float,
        trend: float,
    ) -> str:
        motion = structural > 1.2 or relational > 1.0
        strong_coupling_break = relational > 1.4
        regime_shift = regime_distance > 0.8
        sustained_degrading = trend > 0.03
        if strong_coupling_break and localization > 0.25:
            return "COUPLING_INSTABILITY_OBSERVED"
        if motion and regime_shift and sustained_degrading and localization > 0.20:
            return "STRUCTURAL_INSTABILITY_OBSERVED"
        if motion and regime_shift and not sustained_degrading:
            return "REGIME_SHIFT_OBSERVED"
        if motion and temporal_distortion > 1.0 and localization < 0.20:
            return "COHERENCE_UNDER_CONSTRAINT"
        return "NOMINAL_STRUCTURE"

    @staticmethod
    def state_from_score(instability: float, confidence: float, localization: float) -> str:
        loc_gate = 0.40 + 0.60 * localization
        conf_gate = 0.55 + 0.45 * confidence
        adjusted = instability * loc_gate * conf_gate
        if localization < 0.16 and confidence < 0.55 and adjusted < 2.6:
            return "STABLE"
        if adjusted >= 2.0:
            return "ALERT"
        if adjusted >= 1.0:
            return "WATCH"
        return "STABLE"


class AttributionStage:
    @staticmethod
    def explain(components: dict[str, float], state: str) -> tuple[str, dict[str, float]]:
        total = sum(max(0.0, v) for v in components.values()) + 1e-9
        contrib = {k: max(0.0, v) / total for k, v in components.items()}
        ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        top = [k for k, _ in ranked[:3]]
        msg = f"{state}: dominated by {', '.join(top)}." if top else f"{state}: no dominant structural drivers."
        return msg, contrib



# --- StructuralEngine (alignment.py) ---
# How slowly the rolling baseline adapts (only when nominal); avoid absorbing instability.
DEFAULT_BASELINE_ADAPTATION_ALPHA = 0.92
# Composite below this and nominal state required to update rolling baseline.
BASELINE_UPDATE_MAX_COMPOSITE = 0.85
# Number of recent interpreted states to compute classification stability.
CLASSIFICATION_STABILITY_WINDOW = 15


class StructuralEngine:
    def __init__(
        self,
        baseline_window: int = 50,
        recent_window: int = 12,
        window_stride: int = 1,
        regime_store_path: str = "regime_library.json",
        baseline_adaptation_alpha: float = DEFAULT_BASELINE_ADAPTATION_ALPHA,
    ):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.window_stride = max(1, window_stride)
        self.frames = deque(maxlen=500)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None
        self.score_history: deque[float] = deque(maxlen=120)
        self.baseline_adaptation_alpha = baseline_adaptation_alpha
        # Rolling baseline: updated only when system is nominal and composite low.
        self._rolling_baseline_corr: Optional[np.ndarray] = None
        # Recent interpreted states for classification stability.
        self._state_history: deque[str] = deque(maxlen=CLASSIFICATION_STABILITY_WINDOW)
        self._stage_baseline_profile = NodeBaselineProfile()

        self.regime_store = RegimeStore(regime_store_path)
        persisted = self.regime_store.load()
        self.regime_signatures: list[dict[str, object]] = list(persisted.get("regimes", []))
        self.regime_baselines: dict[str, dict[str, object]] = dict(persisted.get("baselines", {}))

    def _persist_regime_state(self) -> None:
        self.regime_store.save(
            {
                "regimes": self.regime_signatures,
                "baselines": self.regime_baselines,
            }
        )

    def _persistence_features(self) -> dict[str, float]:
        """
        Lightweight persistence/hysteresis helpers derived from composite history.

        This does not change analytics; it provides decision-layer context so
        transient motion does not escalate into persistent instability.
        """
        values = [float(v) for v in self.score_history]
        if not values:
            return {
                "history_len": 0.0,
                "rolling_mean": 0.0,
                "rolling_std": 0.0,
                "consecutive_elevated": 0.0,
                "consecutive_high": 0.0,
            }

        window = values[-min(len(values), 12) :]
        rolling_mean = float(np.mean(window)) if window else 0.0
        rolling_std = float(np.std(window)) if window else 0.0

        consecutive_elevated = 0
        consecutive_high = 0
        for v in reversed(values):
            if v >= 1.5:
                consecutive_elevated += 1
            else:
                break
        for v in reversed(values):
            if v >= 2.5:
                consecutive_high += 1
            else:
                break

        return {
            "history_len": float(len(values)),
            "rolling_mean": float(rolling_mean),
            "rolling_std": float(rolling_std),
            "consecutive_elevated": float(consecutive_elevated),
            "consecutive_high": float(consecutive_high),
        }

    def _vector_from_frame(self, frame: Dict) -> np.ndarray:
        sensor_values = frame["sensor_values"]

        if not self.sensor_order:
            self.sensor_order = sorted(sensor_values.keys())

        values = []
        for name in self.sensor_order:
            v = sensor_values.get(name)
            try:
                values.append(float(v) if v is not None else np.nan)
            except (TypeError, ValueError):
                values.append(np.nan)

        return np.array(values, dtype=float)

    def _get_recent_window(self) -> Optional[np.ndarray]:
        if len(self.frames) < self.recent_window:
            return None

        vectors = np.vstack([f["_vector"] for f in list(self.frames)[-self.recent_window:]])
        vectors = vectors[:: self.window_stride]

        if vectors.shape[0] < 2:
            return None

        return vectors

    def _get_baseline_window(self) -> Optional[np.ndarray]:
        if len(self.frames) < self.baseline_window:
            return None

        vectors = np.vstack([f["_vector"] for f in list(self.frames)[: self.baseline_window]])
        vectors = vectors[:: self.window_stride]

        if vectors.shape[0] < 2:
            return None

        return vectors

    def _get_recent_timestamps(self) -> Optional[list[float]]:
        if len(self.frames) < self.recent_window:
            return None
        ts_vals: list[float] = []
        for f in list(self.frames)[-self.recent_window:]:
            try:
                ts_vals.append(float(f.get("timestamp")))
            except (TypeError, ValueError):
                continue
        return ts_vals if len(ts_vals) >= 2 else None

    def _get_baseline_timestamps(self) -> Optional[list[float]]:
        if len(self.frames) < self.baseline_window:
            return None
        ts_vals: list[float] = []
        for f in list(self.frames)[: self.baseline_window]:
            try:
                ts_vals.append(float(f.get("timestamp")))
            except (TypeError, ValueError):
                continue
        return ts_vals if len(ts_vals) >= 2 else None

    def _system_health(self, drift_score: float, stability_score: float) -> int:
        health = 100.0 - min(drift_score * 20.0, 85.0)
        health += stability_score * 20.0
        return int(round(max(0.0, min(100.0, health))))

    def _alert_state(self, drift_score: float) -> str:
        if drift_score > 3.0:
            return "ALERT"
        if drift_score > 1.5:
            return "WATCH"
        return "STABLE"

    def _drift_alert(self, drift_score: float) -> bool:
        return drift_score > 1.5

    def process_frame(self, frame: Dict) -> Dict:
        vector = self._vector_from_frame(frame)

        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)

        result = {
            "timestamp": frame["timestamp"],
            "site_id": frame["site_id"],
            "asset_id": frame["asset_id"],
            "state": "STABLE",
            "structural_drift_score": 0.0,
            "relational_stability_score": 1.0,
            "system_health": 100,
            "drift_alert": False,
            "sensor_relationships": self.sensor_order,
            "regime_name": None,
            "regime_distance": None,
            "regime_drift": 0.0,
            "latest_drift": 0.0,
            "latest_instability": 0.0,
            "relational_instability_score": 0.0,
            "temporal_distortion_score": 0.0,
            "localization_score": 0.0,
            "causal_attribution": {"top_drivers": [], "driver_scores": {}},
            "dominant_driver": None,
            "explanation": "Warmup: awaiting sufficient window history.",
            "baseline_mode": None,
            "data_quality_summary": {},
            "active_sensor_count": 0,
            "missing_sensor_count": 0,
        }

        baseline_window = self._get_baseline_window()
        recent_window = self._get_recent_window()

        if baseline_window is None or recent_window is None:
            self.latest_result = result
            return result

        data_quality_report = compute_data_quality(
            baseline_window,
            recent_window,
            sensor_names=self.sensor_order,
            timestamps_baseline=self._get_baseline_timestamps(),
            timestamps_recent=self._get_recent_timestamps(),
        )
        result["data_quality"] = data_quality_report.to_dict()
        dq_summary = data_quality_summary(data_quality_report)
        result["data_quality_summary"] = dq_summary
        result["active_sensor_count"] = dq_summary["valid_signal_count"]
        result["missing_sensor_count"] = dq_summary["missing_sensor_count"]

        use_degraded = (not data_quality_report.gate_passed) and should_use_degraded_analytics(
            data_quality_report
        )
        # Optional imputation when gate failed but we still want meaningful degraded output.
        if not data_quality_report.gate_passed and use_degraded:
            baseline_window = impute_missing_simple(baseline_window, method="column_mean")
            recent_window = impute_missing_simple(recent_window, method="column_mean")

        z_baseline, baseline_mean, baseline_std = normalize_window(baseline_window)
        z_recent, recent_mean, recent_std = normalize_window(recent_window)

        valid_mask = (np.nan_to_num(recent_std) > 1e-12) | (np.nan_to_num(baseline_std) > 1e-12)
        valid_signal_count = int(np.sum(valid_mask))

        warning = early_warning_metrics(np.nan_to_num(recent_window, nan=0.0))

        signature = build_regime_signature(recent_mean, recent_std)
        assigned_regime = assign_regime(signature, self.regime_signatures)
        self.regime_signatures = update_regime_library(signature, self.regime_signatures)
        assigned_regime = assign_regime(signature, self.regime_signatures)

        regime_name = assigned_regime["name"] if assigned_regime else None
        regime_distance = float(assigned_regime["distance"]) if assigned_regime else None

        analytics: dict[str, object] = {
            "early_warning": warning,
            "relational_metrics_skipped": valid_signal_count < 2,
            "regime_signature": {
                "current": [float(v) for v in signature],
                "nearest": assigned_regime,
                "assigned_name": regime_name,
                "library_size": len(self.regime_signatures),
            },
        }

        components = canonicalize_components(
            {
                "drift": 0.0,
                "regime_drift": 0.0,
                "early_warning": warning["variance"] + max(0.0, warning["lag1_autocorrelation"]),
            }
        )

        if valid_signal_count >= 2:
            z_base_valid = z_baseline[:, valid_mask]
            z_recent_valid = z_recent[:, valid_mask]
            stage_features = FeatureExtractionStage.extract(z_base_valid, z_recent_valid)

            corr_baseline = correlation_matrix(z_base_valid)
            corr_recent = correlation_matrix(z_recent_valid)

            # Adaptive baseline: use rolling baseline when available to avoid static reference.
            baseline_corr_used = corr_baseline
            baseline_mode = "fixed"
            if (
                self._rolling_baseline_corr is not None
                and self._rolling_baseline_corr.shape == corr_recent.shape
            ):
                baseline_corr_used = self._rolling_baseline_corr
                baseline_mode = "rolling"

            self._stage_baseline_profile.corr_baseline = np.array(baseline_corr_used, dtype=float, copy=True)
            stage_structural_raw, _ = StructuralDriftStage.score(stage_features, self._stage_baseline_profile)
            stage_relational_raw, _ = RelationalInstabilityStage.score(stage_features, self._stage_baseline_profile)
            temporal_raw, _ = TemporalCoherenceStage.score(self._get_recent_timestamps(), self._stage_baseline_profile)
            # Preserve production sensitivity by keeping legacy drift geometry while
            # binding stage outputs into runtime diagnostics.
            drift_score = structural_drift(corr_recent, baseline_corr_used, norm="fro")
            rel_delta_legacy = flatten_upper_tri(corr_recent) - flatten_upper_tri(baseline_corr_used)
            relational_raw = float(np.mean(np.abs(rel_delta_legacy))) if rel_delta_legacy.size else 0.0
            relational_raw = max(relational_raw, stage_relational_raw, 0.5 * stage_structural_raw)
            stability_score = 1.0 / (1.0 + drift_score)

            regime_drift = 0.0
            if regime_name is not None:
                if regime_name not in self.regime_baselines:
                    self.regime_baselines[regime_name] = {
                        "signature": signature.tolist(),
                        "correlation": corr_recent.tolist(),
                        "count": 1,
                    }
                else:
                    regime_corr = np.asarray(self.regime_baselines[regime_name]["correlation"], dtype=float)
                    regime_drift = structural_drift(corr_recent, regime_corr, norm="fro")
                    # Regime-specific baseline: EMA update so we gradually adapt inside stable regime.
                    alpha = 0.88
                    updated = alpha * regime_corr + (1.0 - alpha) * corr_recent
                    self.regime_baselines[regime_name]["correlation"] = updated.tolist()
                    self.regime_baselines[regime_name]["count"] = int(
                        self.regime_baselines[regime_name].get("count", 0)
                    ) + 1

                self._persist_regime_state()

            signal_importance = signal_structural_importance(corr_recent)
            adjacency = thresholded_adjacency(corr_recent, threshold=0.6)
            graph = graph_metrics(adjacency, corr=corr_recent)

            directional = directional_metrics(lagged_correlation_matrix(z_recent_valid, lag=1))

            causal_matrix = granger_causality_matrix(z_recent_valid)
            causal = causal_metrics(causal_matrix)
            causal_graph = causal_graph_metrics(causal_matrix, threshold=0.1)

            valid_sensor_names = [self.sensor_order[i] for i in range(len(valid_mask)) if valid_mask[i]]
            attr = causal_attribution(
                baseline_corr_used,
                corr_recent,
                causal_matrix,
                valid_sensor_names,
                top_k=10,
            )
            result["causal_attribution"] = attr
            result["dominant_driver"] = attr["top_drivers"][0] if attr["top_drivers"] else None

            subsystem = subsystem_spectral_measures(corr_recent)

            spectral = {
                "radius": spectral_radius(corr_recent),
                "gap": spectral_gap(corr_recent),
                **dominant_mode_loading(corr_recent),
            }

            raw_components = {
                "drift": drift_score,
                "relational_drift": relational_raw,
                "regime_drift": regime_drift,
                "spectral": spectral["radius"],
                "directional": max(
                    float(directional.get("divergence", 0.0)),
                    float(causal.get("causal_divergence", 0.0)),
                ),
                "entropy": interaction_entropy(corr_recent),
                "subsystem_instability": float(subsystem["max_instability"]),
                "temporal_distortion": temporal_raw,
            }

            # Merge order matters: preserve early_warning computed from the
            # latest signal window, while ensuring freshly computed relational
            # drift / regime drift / spectral / divergence / entropy /
            # subsystem instability are not clobbered by stale base defaults.
            base_components = components
            raw_canonical = canonicalize_components(raw_components)
            raw_canonical["early_warning"] = float(base_components.get("early_warning", 0.0))

            base_components.update(raw_canonical)
            components = base_components

            result.update(
                {
                    "structural_drift_score": round(drift_score, 4),
                    "relational_stability_score": round(stability_score, 4),
                    "system_health": self._system_health(drift_score, stability_score),
                    "state": self._alert_state(drift_score),
                    "drift_alert": self._drift_alert(drift_score),
                    "regime_name": regime_name,
                    "regime_distance": round(regime_distance, 4) if regime_distance is not None else None,
                    "regime_drift": round(float(regime_drift), 4),
                    "latest_drift": round(float(drift_score), 4),
                    "baseline_mode": baseline_mode,
                }
            )
            regime_memory_state = {
                "regime_name": regime_name,
                "library_size": len(self.regime_signatures),
                "baseline_count": (
                    int(self.regime_baselines.get(regime_name, {}).get("count", 0))
                    if regime_name
                    else None
                ),
            }
            result["regime_memory_state"] = regime_memory_state

            analytics.update(
                {
                    "correlation_geometry": {
                        "baseline": corr_baseline.tolist(),
                        "current": corr_recent.tolist(),
                    },
                    "signal_structural_importance": [float(v) for v in signal_importance],
                    "graph": graph,
                    "directional": directional,
                    "causal": causal,
                    "causal_graph": causal_graph,
                    "subsystems": subsystem,
                    "spectral": spectral,
                    "entropy": float(interaction_entropy(corr_recent)),
                    "regime_drift": float(regime_drift),
                }
            )
        else:
            result["regime_memory_state"] = {
                "regime_name": regime_name,
                "library_size": len(self.regime_signatures),
                "baseline_count": None,
            }

        # Per-component confidence: down-weight or fully suppress evidence when the
        # data quality gate indicates unreliable inputs. Production alerts should
        # be driven by Tier-1 components only.
        tier1_components = {"relational_drift", "regime_drift", "spectral", "early_warning"}

        # Evidence quality in [0, 1]
        missingness_factor = max(0.0, 1.0 - float(data_quality_report.missingness_rate))
        variability_factor = max(0.0, min(1.0, float(data_quality_report.variability_coverage)))
        coverage_factor = max(0.0, min(1.0, float(data_quality_report.sensor_coverage)))
        sample_factor = 0.0
        if data_quality_report.total_sensors > 0:
            sample_factor = float(data_quality_report.valid_signal_count) / float(max(1, data_quality_report.total_sensors))
        sample_factor = max(0.0, min(1.0, sample_factor))

        evidence_conf = (
            missingness_factor
            * (0.4 + 0.6 * variability_factor)
            * (0.4 + 0.6 * coverage_factor)
            * (0.5 + 0.5 * sample_factor)
        )
        if not bool(data_quality_report.gate_passed):
            evidence_conf *= 0.25
        if use_degraded:
            evidence_conf *= 0.5  # Explicit degraded confidence when using fallback analytics
        evidence_conf = max(0.0, min(1.0, evidence_conf))

        correlation_ready = valid_signal_count >= 2

        # Classification stability: how consistent recent interpreted states have been.
        state_history_list = list(self._state_history)
        if len(state_history_list) >= 2:
            counts = Counter(state_history_list)
            most_common_count = max(counts.values()) if counts else 0
            classification_stability = float(most_common_count) / float(len(state_history_list))
        else:
            classification_stability = 1.0

        # Metric disagreement: high std across components slightly reduces confidence.
        comp_vals = [float(components.get(k, 0.0)) for k in tier1_components if k in components]
        if comp_vals:
            mean_c = sum(comp_vals) / len(comp_vals)
            std_c = (sum((x - mean_c) ** 2 for x in comp_vals) / len(comp_vals)) ** 0.5
            disagreement = std_c / (mean_c + 1e-6)
            disagreement_factor = max(0.7, 1.0 - disagreement * 0.15)
        else:
            disagreement_factor = 1.0

        stabilized_confidence = evidence_conf * (0.6 + 0.4 * classification_stability) * disagreement_factor
        stabilized_confidence = max(0.0, min(1.0, stabilized_confidence))

        # Regime baseline confidence depends on how much history exists for the
        # assigned regime. If we don't yet have baseline correlation samples,
        # the regime drift evidence is treated as unreliable.
        regime_count = 0
        if regime_name is not None:
            entry = self.regime_baselines.get(regime_name)
            if isinstance(entry, dict):
                try:
                    regime_count = int(entry.get("count", 0) or 0)
                except (TypeError, ValueError):
                    regime_count = 0

        regime_factor = min(1.0, float(regime_count) / 5.0) if regime_count > 0 else 0.0

        component_confidence: dict[str, float] = {k: 0.0 for k in components.keys()}

        # Tier-1
        component_confidence["relational_drift"] = evidence_conf if correlation_ready else 0.0
        component_confidence["spectral"] = evidence_conf if correlation_ready else 0.0
        component_confidence["early_warning"] = evidence_conf
        component_confidence["regime_drift"] = evidence_conf * regime_factor if correlation_ready else 0.0

        # Suppress non-Tier-1 components explicitly (keeps production composite Tier-1 only)
        for k in list(component_confidence.keys()):
            if k not in tier1_components:
                component_confidence[k] = 0.0

        analytics["component_confidence"] = component_confidence

        # Confidence-weighted composite: use confidence as a scaling on component weights
        # so that unreliable evidence doesn't dilute the Tier-1 score.
        base_weights = canonicalize_weights()
        weights_for_composite: dict[str, float] = {}
        for k, w in base_weights.items():
            weights_for_composite[k] = float(w) * float(component_confidence.get(k, 0.0))

        components_for_decision = {
            k: float(v) * float(component_confidence.get(k, 0.0)) if k in component_confidence else float(v)
            for k, v in components.items()
        }

        composite = composite_instability_score_normalized(components, weights=weights_for_composite)
        self.score_history.append(float(composite))

        persistence = self._persistence_features()

        forecast = {
            "method": "regression+ar1",
            "trend": float(instability_trend(self.score_history)),
            "time_to_instability": time_to_instability(self.score_history),
            "ar1_next": forecast_next(self.score_history),
            "ar1_time_to_instability": time_to_threshold_ar1(self.score_history),
            "persistence": persistence,
        }

        decision = decision_output(
            composite_score=float(composite),
            components=components_for_decision,
            forecast=forecast,
            confidence_score=stabilized_confidence,
            classification_stability=classification_stability,
        )
        result.update(decision)
        stage_interpreted = DecisionStage.interpreted_state(
            structural=float(components.get("drift", 0.0)),
            relational=float(components.get("relational_drift", 0.0)),
            regime_distance=float(components.get("regime_drift", 0.0)),
            temporal_distortion=float(components.get("temporal_distortion", 0.0)),
            localization=1.0,
            trend=float(forecast.get("trend", 0.0)),
        )
        if (
            str(result.get("interpreted_state", "NOMINAL_STRUCTURE")) == "NOMINAL_STRUCTURE"
            and stage_interpreted != "NOMINAL_STRUCTURE"
        ):
            result["interpreted_state"] = stage_interpreted
        elif str(result.get("interpreted_state", "NOMINAL_STRUCTURE")) == "NOMINAL_STRUCTURE":
            # Single-node runtime fallback: preserve legacy structural/coupling detection
            # semantics when multi-node localization context is unavailable.
            rel = float(components.get("relational_drift", 0.0))
            drf = float(components.get("drift", 0.0))
            if rel > 0.9:
                result["interpreted_state"] = "COUPLING_INSTABILITY_OBSERVED"
            elif drf > 1.1:
                result["interpreted_state"] = "STRUCTURAL_INSTABILITY_OBSERVED"
        result["confidence_score"] = round(stabilized_confidence, 4)
        result["latest_instability"] = round(float(composite), 4)
        result["relational_instability_score"] = round(float(components.get("relational_drift", 0.0)), 4)
        result["temporal_distortion_score"] = round(float(components.get("temporal_distortion", data_quality_report.timestamp_irregularity)), 4)
        result["localization_score"] = 0.0

        self._state_history.append(decision.get("interpreted_state", "NOMINAL_STRUCTURE"))

        # Rolling baseline: update only when nominal and composite low (avoid absorbing instability).
        if (
            valid_signal_count >= 2
            and decision.get("interpreted_state") == "NOMINAL_STRUCTURE"
            and float(composite) < BASELINE_UPDATE_MAX_COMPOSITE
        ):
            if self._rolling_baseline_corr is None or self._rolling_baseline_corr.shape != corr_recent.shape:
                self._rolling_baseline_corr = np.array(corr_recent, dtype=float, copy=True)
            else:
                alpha = self.baseline_adaptation_alpha
                self._rolling_baseline_corr = alpha * self._rolling_baseline_corr + (1.0 - alpha) * corr_recent

        analytics["composite_instability"] = round(float(composite), 4)
        analytics["forecasting"] = forecast
        analytics["components"] = components
        explain_components = {
            "structural_drift_score": float(result.get("structural_drift_score", 0.0)),
            "relational_instability_score": float(result.get("relational_instability_score", 0.0)),
            "regime_distance": float(result.get("regime_distance", 0.0) or 0.0),
            "temporal_distortion_score": float(result.get("temporal_distortion_score", 0.0)),
        }
        msg, contrib = AttributionStage.explain(explain_components, str(result.get("state", "STABLE")))
        result["explanation"] = msg
        analytics["component_contributions"] = contrib
        result["dominant_driver"] = (
            max(contrib.items(), key=lambda item: item[1])[0]
            if contrib
            else result.get("dominant_driver")
        )
        result["component_confidence"] = component_confidence

        result["experimental_analytics"] = analytics
        self.latest_result = result

        return result