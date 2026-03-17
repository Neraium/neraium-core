from __future__ import annotations

import numpy as np
import pandas as pd


def window_signature(window: pd.DataFrame) -> np.ndarray:
    return np.hstack([window.mean(axis=0).values, window.std(axis=0, ddof=0).values])


def nearest_regime(signature: np.ndarray, regimes: dict[str, np.ndarray]) -> tuple[str, float]:
    if not regimes:
        return "baseline", 0.0
    best = min(regimes.items(), key=lambda item: np.linalg.norm(signature - item[1]))
    return best[0], float(np.linalg.norm(signature - best[1]))
