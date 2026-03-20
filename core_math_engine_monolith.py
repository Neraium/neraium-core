"""
Neraium SII core math engine — single file for copy/paste.
Dependencies: numpy only.

Contains: window normalization, correlation geometry, structural drift norms,
weighted winsorized composite instability, spectral/graph/directional/entropy
metrics, Granger-style causal proxy, subsystem spectral measures, regime
signatures, early-warning and instability forecasting helpers, observational
causal attribution scores.

Does NOT include: StructuralEngine, data-quality gate, decision_layer, or
RegimeStore JSON persistence. Use intelligence_layer_monolith.py for the full
runtime pipeline.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import numpy as np


# =============================================================================
# --- scoring (composite instability) ---
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

# --- geometry (windows, correlation, drift) ---
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


# --- forecasting (instability trend / TTI) ---
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

# --- forecast_models (AR1) ---
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


# --- causal_proxy (Granger-style, casual.py) ---
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

# --- regime (signatures, library distances) ---
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


# --- staged benchmark/runtime stages (bounded z, stages) ---
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

    @staticmethod
    def adjusted_instability(instability: float, confidence: float, localization: float) -> float:
        """Same inner product as state_from_score (instability × loc_gate × conf_gate), exposed for calibration."""
        loc_gate = 0.40 + 0.60 * float(localization)
        conf_gate = 0.55 + 0.45 * float(confidence)
        return float(max(0.0, float(instability) * loc_gate * conf_gate))


# Minimum baseline samples before switching from global DecisionStage to per-node quantile triage.
MIN_BASELINE_SAMPLES_FOR_CALIBRATION = 28


def decision_adjusted_score(instability: float, confidence: float, localization: float) -> float:
    """Alias for DecisionStage.adjusted_instability — benchmark / diagnostics naming."""
    return DecisionStage.adjusted_instability(instability, confidence, localization)


def state_from_node_quantiles(dec_adj: float, watch_thr: float, alert_thr: float) -> str:
    """Data-driven triage from a node's own baseline score distribution (no shared global cut)."""
    if dec_adj < watch_thr:
        return "STABLE"
    if dec_adj < alert_thr:
        return "WATCH"
    return "ALERT"


def decide_state_with_calibration(
    *,
    phase: str,
    adj: float,
    confidence: float,
    localization: float,
    dec_adj: float,
    baseline_dec_adj_prior: list[float],
    frozen_watch_alert: tuple[float, float] | None,
) -> tuple[str, str]:
    """
    Returns (state, decision_mode).

    After burn-in, baseline uses quantiles of *this node's prior* adjusted scores; after baseline,
    perturbation/recovery use frozen quantiles from the full baseline window for that node so
    variant-specific score paths produce variant-specific stability statistics.
    """
    if phase == "baseline":
        if len(baseline_dec_adj_prior) < MIN_BASELINE_SAMPLES_FOR_CALIBRATION:
            return DecisionStage.state_from_score(adj, confidence, localization), "global_fallback"
        arr = np.asarray(baseline_dec_adj_prior, dtype=float)
        w_thr = float(np.percentile(arr, 82.0))
        a_thr = float(np.percentile(arr, 93.5))
        return state_from_node_quantiles(dec_adj, w_thr, a_thr), "online_baseline_quantile"
    if frozen_watch_alert is not None:
        w_thr, a_thr = frozen_watch_alert
        return state_from_node_quantiles(dec_adj, w_thr, a_thr), "frozen_post_baseline_quantile"
    return DecisionStage.state_from_score(adj, confidence, localization), "global_fallback"


class AttributionStage:
    @staticmethod
    def explain(components: dict[str, float], state: str) -> tuple[str, dict[str, float]]:
        total = sum(max(0.0, v) for v in components.values()) + 1e-9
        contrib = {k: max(0.0, v) / total for k, v in components.items()}
        ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        top = [k for k, _ in ranked[:3]]
        msg = f"{state}: dominated by {', '.join(top)}." if top else f"{state}: no dominant structural drivers."
        return msg, contrib



