"""Day 6: signal stability diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.diagnostics import compute_signal_stability


def _df() -> pd.DataFrame:
    n = 12
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            DATE_COLUMN: idx,
            "regime_label": ["x"] * 6 + ["y"] * 6,
            "action_posture": ["watch"] * 5 + ["wait"] * 7,
            "confidence_score": np.linspace(0.5, 0.9, n),
        }
    )


def test_signal_stability_columns_exist():
    out = compute_signal_stability(_df())
    for c in ("regime_flip_flag", "action_flip_flag", "confidence_jump_flag", "stability_score"):
        assert c in out.columns
