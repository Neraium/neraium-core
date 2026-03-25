from __future__ import annotations

from typing import Dict

import numpy as np


def classify_drift_noise(drift_history: list[float]) -> Dict[str, object]:
    arr = np.asarray(drift_history[-30:], dtype=float)
    if arr.size < 3:
        return {
            "persistent_change_score": 0.0,
            "transient_excursion_score": 0.0,
            "noise_to_drift_ratio": 1.0,
            "interpreted_change_type": "noise",
        }

    diffs = np.diff(arr)
    rolling = np.convolve(arr, np.ones(3) / 3.0, mode="valid") if arr.size >= 3 else arr
    persistence = float(np.mean(np.abs(rolling[-min(6, rolling.size):])))
    transient = float(np.max(np.abs(diffs - np.mean(diffs))))
    noise_ratio = float(transient / (persistence + 1e-6))

    if persistence > 1.1 and np.mean(diffs[-min(5, diffs.size):]) > 0.02:
        interpreted = "accelerating_drift"
    elif persistence > 0.7:
        interpreted = "persistent_drift"
    elif transient > 0.2:
        interpreted = "transient_shift"
    else:
        interpreted = "noise"

    return {
        "persistent_change_score": round(persistence, 4),
        "transient_excursion_score": round(transient, 4),
        "noise_to_drift_ratio": round(noise_ratio, 4),
        "interpreted_change_type": interpreted,
    }
