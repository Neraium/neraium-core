"""Day 5 naive baseline signal builders and comparison helpers."""

from __future__ import annotations

import pandas as pd

from config import DATE_COLUMN
from neraium.evaluation import _score_single_action


BASELINE_ACTION_COLUMNS = [
    "baseline_trend_action",
    "baseline_vol_action",
    "baseline_breadth_action",
]


def baseline_trend_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Simple trend baseline: ``spy_ret_5d > 0`` -> ``lean_long`` else ``lean_short``."""
    out = df.copy()
    if "spy_ret_5d" not in out.columns:
        raise KeyError("Missing required column: spy_ret_5d")
    out["baseline_trend_action"] = out["spy_ret_5d"].apply(
        lambda x: "lean_long" if pd.notna(x) and x > 0 else "lean_short"
    )
    return out


def baseline_vol_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility baseline using ``spy_vol_10d`` percentile thresholding."""
    out = df.copy()
    if "spy_vol_10d" not in out.columns:
        raise KeyError("Missing required column: spy_vol_10d")

    vol = out["spy_vol_10d"].astype(float)
    high_thr = vol.quantile(0.8)
    med_thr = vol.quantile(0.6)

    def _map(v: float) -> str:
        if pd.isna(v):
            return "watch"
        if v >= high_thr:
            return "avoid_risk"
        if v >= med_thr:
            return "reduce_exposure"
        return "watch"

    out["baseline_vol_action"] = vol.apply(_map)
    return out


def baseline_breadth_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Breadth baseline using only ``breadth_pct_above_20dma``."""
    out = df.copy()
    if "breadth_pct_above_20dma" not in out.columns:
        raise KeyError("Missing required column: breadth_pct_above_20dma")

    def _map(b: float) -> str:
        if pd.isna(b):
            return "watch"
        if b < 40:
            return "reduce_exposure"
        if b > 60:
            return "lean_long"
        return "watch"

    out["baseline_breadth_action"] = out["breadth_pct_above_20dma"].apply(_map)
    return out


def _summarize_action_set(df: pd.DataFrame, action_col: str, label: str) -> list[dict]:
    """Compute horizon-level usefulness summary for one action column."""
    rows: list[dict] = []
    fwd_cols = [c for c in df.columns if c.startswith("fwd_ret_") and c.endswith("d")]

    for col in fwd_cols:
        horizon = int(col.replace("fwd_ret_", "").replace("d", ""))
        useful = [
            _score_single_action(a, r, v)
            for a, r, v in zip(
                df[action_col],
                df[col],
                df.get("spy_vol_10d", pd.Series([None] * len(df))),
            )
        ]
        s = pd.Series(useful, dtype=float)
        rows.append(
            {
                "model": label,
                "horizon": horizon,
                "avg_usefulness": float(s.mean()),
                "hit_rate": float((s > 0).mean()),
                "non_neutral_count": int((s != 0).sum()),
                "abstention_rate": float((s == 0).mean()),
            }
        )

    return rows


def compare_to_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Neraium posture usefulness versus naive baselines.

    Returns one row per (model, horizon), with:
    ``avg_usefulness``, ``hit_rate``, ``non_neutral_count``, ``abstention_rate``.
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    required = ["action_posture"]
    for col in required:
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")

    out = baseline_trend_signal(out)
    out = baseline_vol_signal(out)
    out = baseline_breadth_signal(out)

    rows: list[dict] = []
    rows.extend(_summarize_action_set(out, "action_posture", "neraium"))
    rows.extend(_summarize_action_set(out, "baseline_trend_action", "baseline_trend"))
    rows.extend(_summarize_action_set(out, "baseline_vol_action", "baseline_vol"))
    rows.extend(_summarize_action_set(out, "baseline_breadth_action", "baseline_breadth"))

    return pd.DataFrame(rows).sort_values(["horizon", "model"]).reset_index(drop=True)
