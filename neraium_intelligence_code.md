# Neraium Intelligence Code (SII)

This file contains only the **intelligence** (structural analysis) Python code: engine, decision layer, scoring, geometry, regime, forecasting, data quality, and related analytics. No UI, API, or test code.

---

## 1. `neraium_core/geometry.py` — Correlation & structural drift

```python
from __future__ import annotations

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
```

---

## 2. `neraium_core/regime.py` — Regime signatures & assignment

```python
from __future__ import annotations

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
    """Update the regime library. If no regime exists, bootstrap one."""
    if not regimes:
        regimes.append({"name": "regime_0", "signature": signature.tolist()})
        return regimes

    assigned = assign_regime(signature, regimes)
    if assigned is None or float(assigned["distance"]) > threshold:
        regimes.append(
            {"name": f"regime_{len(regimes)}", "signature": signature.tolist()}
        )

    return regimes
```

---

## 3. `neraium_core/scoring.py` — Composite instability score

```python
from __future__ import annotations

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
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def normalize_keys(values: Mapping[str, object]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in values.items():
        if key in DEFAULT_COMPONENTS or key in DEFAULT_WEIGHTS:
            normalized[key] = _coerce_float(value)
    for key, value in values.items():
        mapped = LEGACY_KEYS.get(key, key)
        if mapped in normalized:
            continue
        normalized[mapped] = _coerce_float(value)
    return normalized


def canonicalize_components(components: Mapping[str, object] | None = None) -> dict[str, float]:
    result = dict(DEFAULT_COMPONENTS)
    if not components:
        return result
    result.update(normalize_keys(components))
    return result


def canonicalize_weights(weights: Mapping[str, object] | None = None) -> dict[str, float]:
    result = dict(DEFAULT_WEIGHTS)
    if not weights:
        return result
    result.update(normalize_keys(weights))
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


DEFAULT_WINSORIZE_CAP = 3.0


def _winsorize(value: float, low: float = 0.0, high: float = DEFAULT_WINSORIZE_CAP) -> float:
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
    """Composite instability with per-component winsorization."""
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
```

---

## 4. `neraium_core/decision_layer.py` — Interpreted state & operator output

```python
from __future__ import annotations

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
) -> str:
    if relational_drift > 1.2 and regime_drift < 0.8:
        return "REGIME_SHIFT_OBSERVED"
    if relational_drift > 1.2 and regime_drift >= 0.8:
        return "STRUCTURAL_INSTABILITY_OBSERVED"
    if directional > 1.0 or spectral > 1.2:
        return "COUPLING_INSTABILITY_OBSERVED"
    return "NOMINAL_STRUCTURE"


def _operator_message(
    state: str,
    trend: float,
    time_to_instability: float | None,
) -> str:
    """Strictly observational language. No control, no directives."""
    if state == "STRUCTURAL_INSTABILITY_OBSERVED":
        if time_to_instability is not None:
            return (
                "Observed structural relationships are diverging from previously seen "
                "system patterns. Current configuration exhibits elevated instability "
                f"characteristics under current analysis, with continued progression "
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
    components: dict[str, float],
    forecast: dict[str, Any],
) -> dict[str, Any]:
    """Convert structural analytics into operator-safe decision output. Observational only."""
    relational_drift = float(components.get("relational_drift", 0.0))
    regime_drift = float(components.get("regime_drift", 0.0))
    directional = float(components.get("directional_divergence", 0.0))
    spectral = float(components.get("spectral", 0.0))

    trend = float(forecast.get("trend", 0.0))
    time_to_instability = forecast.get("ar1_time_to_instability") or forecast.get("time_to_instability")

    state = _interpret_state(
        relational_drift=relational_drift,
        regime_drift=regime_drift,
        directional=directional,
        spectral=spectral,
    )

    risk_level = _risk_level(composite_score)
    signal_strength = _signal_strength(composite_score, trend)
    confidence = _confidence(components)
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

    return {
        "phase": phase,
        "risk_level": risk_level,
        "signal_emitted": signal_emitted,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "operator_message": operator_message,
        "interpreted_state": state,
    }
```

---

## 5. `neraium_core/alignment.py` — StructuralEngine (main intelligence pipeline)

This is the main SII engine: it ingests frames, runs data quality, baseline vs recent correlation geometry, regime assignment, spectral/directional/causal/subsystem metrics, confidence-weighted composite score, and decision_output. **Location in repo:** `neraium_core/alignment.py` (full file ~382 lines). It imports and uses:

- `geometry`: normalize_window, correlation_matrix, structural_drift, signal_structural_importance  
- `regime`: build_regime_signature, assign_regime, update_regime_library  
- `regime_store`: RegimeStore  
- `data_quality`: compute_data_quality  
- `early_warning`: early_warning_metrics  
- `scoring`: canonicalize_components, canonicalize_weights, composite_instability_score_normalized  
- `decision_layer`: decision_output  
- `forecasting`: instability_trend, time_to_instability  
- `forecast_models`: forecast_next, time_to_threshold_ar1  
- `directional`: directional_metrics, lagged_correlation_matrix  
- `causal`: causal_metrics, granger_causality_matrix  
- `causal_graph`: causal_graph_metrics  
- `graph`: graph_metrics, thresholded_adjacency  
- `spectral`: spectral_radius, spectral_gap, dominant_mode_loading  
- `entropy`: interaction_entropy  
- `subsystems`: subsystem_spectral_measures  

The single entry point you care about is:

```python
def process_frame(self, frame: Dict) -> Dict:
    # frame: { "timestamp", "site_id", "asset_id", "sensor_values": { name: number } }
    # Returns: state, structural_drift_score, relational_stability_score, regime_*, 
    #          latest_instability, interpreted_state, phase, risk_level, operator_message, ...
```

Full `alignment.py` lives in your repo at **`neraium_core/alignment.py`** — open that file for the complete engine.

---

## 6. Supporting intelligence modules (short)

- **`neraium_core/forecasting.py`** — `instability_trend(series)`, `time_to_instability(series, threshold=1.5)` (linear regression trend and time-to-threshold).
- **`neraium_core/forecast_models.py`** — `forecast_next(series)`, `time_to_threshold_ar1(series, threshold=1.5)` (AR(1) one-step forecast and time-to-threshold).
- **`neraium_core/early_warning.py`** — `early_warning_metrics(observations)` → variance and lag-1 autocorrelation.
- **`neraium_core/data_quality.py`** — `compute_data_quality(baseline_matrix, recent_matrix, ...)` → DataQualityReport (missingness, flatlined/stale sensors, gate_passed).
- **`neraium_core/spectral.py`** — `spectral_radius`, `spectral_gap`, `dominant_mode_loading` from correlation matrix.
- **`neraium_core/entropy.py`** — `interaction_entropy(matrix)` from correlation magnitudes.
- **`neraium_core/graph.py`** — `thresholded_adjacency(corr, threshold=0.6)`, `graph_metrics(adjacency, corr)`.
- **`neraium_core/directional.py`** — `lagged_correlation_matrix(observations, lag=1)`, `directional_metrics(matrix)`.
- **`neraium_core/casual.py`** (used as `causal`) — `granger_causality_matrix(X, lag=1)`, `causal_metrics(C)`.
- **`neraium_core/causal_graph.py`** — `causal_graph_metrics(C, threshold=0.1)`.
- **`neraium_core/subsystems.py`** — `subsystem_spectral_measures(corr, k=3)` (subsystem instability from spectral clustering).
- **`regime_store.py`** (project root) — `RegimeStore(path)` load/save regime library JSON.

---

## Summary

- **Intelligence entry point:** `neraium_core.alignment.StructuralEngine.process_frame(frame)`.
- **Decision layer:** `neraium_core.decision_layer.decision_output(composite_score, components, forecast)` → interpreted_state, risk_level, operator_message, etc.
- **Scoring:** `neraium_core.scoring.composite_instability_score_normalized(components, weights=...)`.
- **Geometry:** `neraium_core.geometry` (normalize_window, correlation_matrix, structural_drift, signal_structural_importance).
- **Regime:** `neraium_core.regime` (build_regime_signature, assign_regime, update_regime_library).

All of the above live under **`neraium_core/`** and **`regime_store.py`** in your repo; this document is a single reference for “the intelligence code” only.
