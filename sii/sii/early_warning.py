from __future__ import annotations

import numpy as np
import pandas as pd


def early_warning_metrics(window: pd.DataFrame) -> dict[str, float]:
    variances = window.var(axis=0, ddof=0)
    ac1 = window.apply(lambda c: c.autocorr(lag=1), axis=0).replace([np.inf, -np.inf], np.nan)
    return {
        "variance_avg": float(np.nanmean(variances.values)),
        "lag1_autocorr_avg": float(np.nanmean(ac1.values)),
    }
