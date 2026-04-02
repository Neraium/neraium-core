"""Day 6: signal stability diagnostics (deterministic flip / jump flags)."""

from __future__ import annotations

import pandas as pd

from config import DATE_COLUMN

# Confidence step larger than this counts as a "jump" (not smooth).
DEFAULT_CONFIDENCE_JUMP_THRESHOLD = 0.25

# Rolling window (rows) for smoothing stability_score.
STABILITY_ROLLING_WINDOW = 5


def compute_signal_stability(
    df: pd.DataFrame,
    confidence_jump_threshold: float = DEFAULT_CONFIDENCE_JUMP_THRESHOLD,
) -> pd.DataFrame:
    """
    Flag frequent regime/action flips and large confidence steps.

    Columns added:
    - ``regime_flip_flag``: 1 if ``regime_label`` differs from previous row, else 0 (first row 0).
    - ``action_flip_flag``: 1 if ``action_posture`` differs from previous row, else 0 (first row 0).
    - ``confidence_jump_flag``: 1 if |Δ confidence_score| > threshold, else 0 (first row 0).
    - ``stability_score``: in [0, 1], higher when recent rows have fewer flips/jumps.

    ``stability_score`` uses a rolling mean of a per-row calm score:
    ``1 - min(1, 0.35*regime_flip + 0.35*action_flip + 0.3*confidence_jump_flag)`` over
    ``STABILITY_ROLLING_WINDOW`` rows.
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    for col in ("regime_label", "action_posture", "confidence_score"):
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")

    regime = out["regime_label"].astype(str)
    action = out["action_posture"].astype(str)
    conf = out["confidence_score"].astype(float)

    out["regime_flip_flag"] = regime.ne(regime.shift(1)).astype(int).fillna(0).astype(int)
    out["action_flip_flag"] = action.ne(action.shift(1)).astype(int).fillna(0).astype(int)
    dconf = conf.diff().abs()
    out["confidence_jump_flag"] = (dconf > confidence_jump_threshold).astype(int)
    out.loc[0, "confidence_jump_flag"] = 0

    calm = 1.0 - (
        0.35 * out["regime_flip_flag"]
        + 0.35 * out["action_flip_flag"]
        + 0.30 * out["confidence_jump_flag"]
    ).clip(0.0, 1.0)
    out["stability_score"] = (
        calm.rolling(STABILITY_ROLLING_WINDOW, min_periods=1).mean().clip(0.0, 1.0)
    )

    return out.loc[:, ~out.columns.duplicated()]
