from __future__ import annotations

from typing import Dict

import numpy as np


def compute_multi_scale_states(matrix: np.ndarray) -> Dict[str, object]:
    safe = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    n = safe.shape[0] if safe.ndim == 2 else 0
    if n < 3:
        return {
            "short_term_state": "insufficient_data",
            "mid_term_state": "insufficient_data",
            "long_term_state": "insufficient_data",
            "scale_conflict": 0.0,
            "scale_alignment": 1.0,
            "scale_conflict_reason": "insufficient_data",
        }

    def state_for(window: np.ndarray) -> tuple[str, float]:
        slope = float(np.mean(window[-1] - window[0]))
        mag = abs(slope)
        if mag < 0.01:
            return "stable", mag
        if slope > 0:
            return "rising_drift", mag
        return "recovering", mag

    s_state, s_mag = state_for(safe[-max(3, n // 6):])
    m_state, m_mag = state_for(safe[-max(4, n // 3):])
    l_state, l_mag = state_for(safe)
    states = [s_state, m_state, l_state]
    align = float(max(states.count("stable"), states.count("rising_drift"), states.count("recovering")) / 3.0)
    conflict = 1.0 - align

    reason = "aligned" if conflict < 0.34 else "short_mid_long_disagreement"

    return {
        "short_term_state": s_state,
        "mid_term_state": m_state,
        "long_term_state": l_state,
        "scale_conflict": round(conflict, 4),
        "scale_alignment": round(align, 4),
        "scale_conflict_reason": reason,
        "scale_magnitudes": {
            "short": round(s_mag, 4),
            "mid": round(m_mag, 4),
            "long": round(l_mag, 4),
        },
    }
