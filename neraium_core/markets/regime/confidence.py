"""Confidence scoring functions."""

from __future__ import annotations

import numpy as np


def score_confidence(
    coherence: float,
    persistence: float,
    data_quality: float,
    contradiction_penalty: float,
) -> float:
    score = 0.35 * coherence + 0.3 * persistence + 0.25 * data_quality - 0.3 * contradiction_penalty
    return float(np.clip(score, 0.0, 1.0))
