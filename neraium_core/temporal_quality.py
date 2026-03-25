from __future__ import annotations

from typing import Sequence

import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def derive_temporal_quality_signals(timestamps: Sequence[float] | None) -> dict[str, float]:
    ts = np.asarray(list(timestamps) if timestamps is not None else [], dtype=float)
    if ts.size < 3:
        return {
            "temporal_consistency_score": 0.0,
            "ordering_stability_score": 0.0,
            "timestamp_gap_irregularity": 1.0,
            "alignment_confidence": 0.0,
            "effective_sampling_density": 0.0,
        }

    dt = np.diff(ts)
    positive_ratio = float(np.mean(dt > 0.0))
    non_positive_count = int(np.sum(dt <= 0.0))
    monotonic_break_penalty = min(1.0, non_positive_count / max(1.0, dt.size))

    safe_dt = np.maximum(1e-6, np.abs(dt))
    gap_irregularity = float(np.std(safe_dt) / (np.mean(safe_dt) + 1e-9))
    ordering_stability = _clamp01(positive_ratio * (1.0 - monotonic_break_penalty))
    temporal_consistency = _clamp01((1.0 - gap_irregularity) * ordering_stability)

    span = max(1e-6, float(np.max(ts) - np.min(ts)))
    effective_sampling_density = _clamp01(float(ts.size / span))

    alignment_confidence = _clamp01(
        0.45 * temporal_consistency
        + 0.30 * ordering_stability
        + 0.25 * (1.0 - _clamp01(gap_irregularity))
    )

    return {
        "temporal_consistency_score": round(float(temporal_consistency), 6),
        "ordering_stability_score": round(float(ordering_stability), 6),
        "timestamp_gap_irregularity": round(float(_clamp01(gap_irregularity)), 6),
        "alignment_confidence": round(float(alignment_confidence), 6),
        "effective_sampling_density": round(float(effective_sampling_density), 6),
    }
