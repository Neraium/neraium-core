from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_array(timestamps: Iterable[float]) -> np.ndarray:
    return np.asarray(list(timestamps), dtype=float)


def perturb_timestamps(timestamps: Iterable[float], mode: str, strength: float = 0.15) -> np.ndarray:
    ts = _as_array(timestamps)
    if ts.size == 0:
        return ts
    m = (mode or "jitter").lower()
    idx = np.arange(ts.size, dtype=float)
    if m == "timestamp_jitter":
        jitter = strength * np.sin(0.7 * idx)
        return ts + jitter
    if m == "timestamp_compression":
        return ts[0] + (ts - ts[0]) * max(0.2, 1.0 - strength)
    if m == "timestamp_expansion":
        return ts[0] + (ts - ts[0]) * (1.0 + strength)
    if m == "bursty_intervals":
        delta = np.diff(ts, prepend=ts[0])
        delta[1::3] *= (1.0 + 2.5 * strength)
        return np.cumsum(delta) + ts[0]
    if m == "gap_insertion":
        out = ts.copy()
        if ts.size > 6:
            out[ts.size // 2 :] += strength * ts.size
        return out
    if m == "local_reordering":
        out = ts.copy()
        if ts.size > 4:
            i = ts.size // 3
            out[i : i + 3] = out[i : i + 3][::-1]
        return out
    if m == "sampling_irregularity":
        delta = np.diff(ts, prepend=ts[0])
        modifier = 1.0 + strength * np.sin(1.3 * idx)
        return np.cumsum(delta * modifier) + ts[0]
    return ts


def temporal_shift_metrics(base: dict, perturbed: dict) -> dict[str, float]:
    def g(payload: dict, key: str, default: float = 0.0) -> float:
        return float(payload.get(key, default) or default)

    b_state = str(base.get("state", "STABLE"))
    p_state = str(perturbed.get("state", "STABLE"))
    b_branch = float((base.get("experimental_analytics", {}).get("branching_analysis", {}) or {}).get("branch_count_estimate", 1.0) or 1.0)
    p_branch = float((perturbed.get("experimental_analytics", {}).get("branching_analysis", {}) or {}).get("branch_count_estimate", 1.0) or 1.0)

    b_h = g((base.get("experimental_analytics", {}).get("horizon_analysis", {}) or {}), "horizon_steps")
    p_h = g((perturbed.get("experimental_analytics", {}).get("horizon_analysis", {}) or {}), "horizon_steps")
    b_l = g((base.get("experimental_analytics", {}).get("constraint_analysis", {}) or {}), "lock_in_score")
    p_l = g((perturbed.get("experimental_analytics", {}).get("constraint_analysis", {}) or {}), "lock_in_score")

    base_temporal = g(base, "temporal_distortion_score")
    pert_temporal = g(perturbed, "temporal_distortion_score")
    drift_delta = abs(g(base, "latest_drift") - g(perturbed, "latest_drift"))
    alignment_effect = 0.6 * abs(base_temporal - pert_temporal) + 0.4 * drift_delta
    return {
        "alignment_effect_score": round(alignment_effect, 4),
        "structural_decision_change_rate": round(1.0 if (b_state != p_state or abs(base_temporal - pert_temporal) > 0.02) else 0.0, 4),
        "branch_flip_rate": round(1.0 if abs(b_branch - p_branch) > 0.5 else 0.0, 4),
        "horizon_shift_rate": round(abs(b_h - p_h) / (abs(b_h) + 1e-6) if b_h else 0.0, 4),
        "lock_in_shift_rate": round(abs(b_l - p_l) / (abs(b_l) + 1e-6) if b_l else 0.0, 4),
    }
