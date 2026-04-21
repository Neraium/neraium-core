from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neraium_core.features.frequency_features import (
    band_energy,
    dominant_frequency_ratio,
    dominant_frequency_magnitude,
    low_high_frequency_energy_ratio,
    spectral_energy_concentration,
    spectral_energy_dispersion,
    spectral_centroid,
    spectral_entropy,
    spectral_spread,
)
from neraium_core.features.signal_features import (
    crest_factor,
    envelope_mean,
    envelope_std,
    kurtosis,
    lag1_autocorrelation,
    local_jitter_score,
    lower_tail_ratio,
    mean_abs_first_difference,
    mean_abs_second_difference,
    mean_absolute_deviation,
    peak_to_peak,
    percentile_spread,
    rolling_local_volatility,
    rms,
    rolling_energy,
    roughness,
    std_first_difference,
    shannon_entropy,
    skewness,
    trend_slope,
    upper_tail_ratio,
    variance,
    worsening_trend_persistence,
    zero_crossing_rate,
)


@dataclass(frozen=True)
class WindowFeaturePayload:
    per_channel: dict[str, dict[str, float]]
    cross_channel: dict[str, float]
    flattened: dict[str, float]


def _safe_matrix(window: np.ndarray) -> np.ndarray:
    arr = np.asarray(window, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("window must be 1D or 2D")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _channel_feature_vector(x: np.ndarray) -> dict[str, float]:
    feats: dict[str, float] = {
        "rms": rms(x),
        "peak_to_peak": peak_to_peak(x),
        "crest_factor": crest_factor(x),
        "kurtosis": kurtosis(x),
        "skewness": skewness(x),
        "variance": variance(x),
        "rolling_energy": rolling_energy(x),
        "entropy": shannon_entropy(x),
        "zero_crossing_rate": zero_crossing_rate(x),
        "mean_absolute_deviation": mean_absolute_deviation(x),
        "percentile_spread_p95_p5": percentile_spread(x),
        "upper_tail_ratio_p95": upper_tail_ratio(x),
        "lower_tail_ratio_p05": lower_tail_ratio(x),
        "mean_abs_first_diff": mean_abs_first_difference(x),
        "std_first_diff": std_first_difference(x),
        "mean_abs_second_diff": mean_abs_second_difference(x),
        "rolling_local_volatility": rolling_local_volatility(x),
        "local_jitter_score": local_jitter_score(x),
        "autocorr_lag1": lag1_autocorrelation(x),
        "trend_slope": trend_slope(x),
        "worsening_trend_persistence": worsening_trend_persistence(x),
        "roughness": roughness(x),
        "envelope_mean": envelope_mean(x),
        "envelope_std": envelope_std(x),
        "dominant_frequency_magnitude": dominant_frequency_magnitude(x),
        "dominant_frequency_ratio": dominant_frequency_ratio(x),
        "spectral_centroid": spectral_centroid(x),
        "spectral_spread": spectral_spread(x),
        "spectral_entropy": spectral_entropy(x),
        "low_high_frequency_energy_ratio": low_high_frequency_energy_ratio(x),
        "spectral_energy_dispersion": spectral_energy_dispersion(x),
        "spectral_energy_concentration_top3": spectral_energy_concentration(x, top_k=3),
    }
    feats.update(band_energy(x))
    return {k: float(v) for k, v in feats.items()}


def _cross_channel_features(matrix: np.ndarray) -> dict[str, float]:
    n_channels = int(matrix.shape[1])
    n_obs = int(matrix.shape[0])
    if n_channels < 2 or n_obs < 2:
        return {
            "channel_corr_mean": 1.0,
            "channel_corr_std": 0.0,
            "pairwise_energy_imbalance": 0.0,
            "sync_loss": 0.0,
            "relative_drift": 0.0,
            "decoupling_index": 0.0,
            "coherence_like": 1.0,
        }
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.triu_indices(n_channels, k=1)
    pairs = corr[idx] if idx[0].size else np.array([1.0], dtype=float)
    energies = np.mean(np.square(matrix), axis=0)
    energy_den = float(np.mean(energies) + 1e-12)
    energy_imbalance = float(np.mean(np.abs(energies[:, None] - energies[None, :])) / energy_den)

    slopes = []
    t = np.arange(matrix.shape[0], dtype=float)
    tc = t - float(np.mean(t))
    den = float(np.dot(tc, tc)) + 1e-12
    for i in range(n_channels):
        x = matrix[:, i]
        xc = x - float(np.mean(x))
        slopes.append(float(np.dot(tc, xc) / den))
    slopes_arr = np.asarray(slopes, dtype=float)

    sync_loss = float(max(0.0, 1.0 - np.mean(np.abs(pairs))))
    rel_drift = float(np.std(slopes_arr) / (np.mean(np.abs(slopes_arr)) + 1e-12))
    decoupling = float(np.std(pairs))
    coherence_like = float(max(0.0, min(1.0, np.mean(np.abs(pairs)))))

    return {
        "channel_corr_mean": float(np.mean(pairs)),
        "channel_corr_std": float(np.std(pairs)),
        "pairwise_energy_imbalance": energy_imbalance,
        "sync_loss": sync_loss,
        "relative_drift": rel_drift,
        "decoupling_index": decoupling,
        "coherence_like": coherence_like,
    }


def extract_window_features(window: np.ndarray, *, channel_names: list[str] | None = None) -> WindowFeaturePayload:
    matrix = _safe_matrix(window)
    n_channels = int(matrix.shape[1])
    names = channel_names if channel_names and len(channel_names) == n_channels else [f"ch_{i}" for i in range(n_channels)]

    per_channel: dict[str, dict[str, float]] = {}
    flattened: dict[str, float] = {}
    for i, name in enumerate(names):
        feats = _channel_feature_vector(matrix[:, i])
        per_channel[name] = feats
        for key, value in feats.items():
            flattened[f"{name}__{key}"] = float(value)

    cross = _cross_channel_features(matrix)
    for key, value in cross.items():
        flattened[f"cross__{key}"] = float(value)

    return WindowFeaturePayload(per_channel=per_channel, cross_channel=cross, flattened=flattened)


def summarize_feature_delta(
    baseline_window: np.ndarray,
    recent_window: np.ndarray,
    *,
    channel_names: list[str] | None = None,
) -> dict[str, Any]:
    base = extract_window_features(baseline_window, channel_names=channel_names)
    recent = extract_window_features(recent_window, channel_names=channel_names)
    delta: dict[str, float] = {}
    keys = sorted(set(base.flattened.keys()) | set(recent.flattened.keys()))
    for k in keys:
        delta[k] = float(recent.flattened.get(k, 0.0) - base.flattened.get(k, 0.0))
    drift_magnitude = float(np.mean(np.abs(list(delta.values())))) if delta else 0.0
    signed_drift = float(np.mean(list(delta.values()))) if delta else 0.0
    volatility_base = float(np.std(list(base.flattened.values()))) if base.flattened else 0.0
    volatility_recent = float(np.std(list(recent.flattened.values()))) if recent.flattened else 0.0
    volatility_drift = float(max(0.0, volatility_recent - volatility_base))
    coherence_related = [
        abs(delta.get(k, 0.0))
        for k in delta
        if "coherence_like" in k or "channel_corr" in k or "sync_loss" in k
    ]
    consistency_breakdown = float(np.mean(coherence_related)) if coherence_related else 0.0
    return {
        "baseline": base,
        "recent": recent,
        "delta": delta,
        "change_summary": {
            "feature_drift_magnitude": drift_magnitude,
            "feature_signed_drift": signed_drift,
            "feature_volatility_drift": volatility_drift,
            "feature_consistency_breakdown": consistency_breakdown,
        },
    }

