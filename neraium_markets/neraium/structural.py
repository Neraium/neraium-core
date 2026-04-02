"""Day 3 structural snapshot: deterministic aggregates from the feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN, SECTOR_ASSETS


def _safe_norm01(s: pd.Series) -> pd.Series:
    """Min-max to [0, 1] with epsilon; NaN preserved until filled."""
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window, min_periods=max(3, window // 2)).corr(b)


def _sector_row_entropy(row: pd.Series) -> float:
    """Shannon entropy of normalized positive weights; scaled to [0, 1]."""
    x = row.to_numpy(dtype=float)
    x = np.abs(np.nan_to_num(x, nan=0.0))
    s = x.sum()
    if s <= 0:
        return 0.5
    p = x / s
    p = p[p > 0]
    h = float(-np.sum(p * np.log(p)))
    return float(h / np.log(len(x))) if len(x) > 1 else 0.5


def build_structural_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add structural columns expected by Day 4 regime logic.

    Produces (all aligned to input rows):
    - corr_drift_score: instability in rolling SPY/QQQ return correlation (0–1).
    - lag_drift_score: instability in lagged correlation (lead/lag structure) (0–1).
    - sector_entropy: normalized dispersion of sector participation (0–1).
    - sector_concentration_score: concentration of sector moves (0–1).
    - instability_score: composite stress (0–1).
    - coherence_score: cross-sectional agreement of core equity returns (0–1).

    Assumes Day 2 feature columns exist (returns, breadth, dispersion, risk_off_proxy).
    """
    out = df.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    spy_r = out.get("spy_ret_1d")
    qqq_r = out.get("qqq_ret_1d")
    iwm_r = out.get("iwm_ret_1d")

    if spy_r is not None and qqq_r is not None:
        rc = _rolling_corr(spy_r, qqq_r, 20)
        out["corr_drift_score"] = _safe_norm01((rc - rc.shift(5)).abs())
    else:
        out["corr_drift_score"] = 0.5

    if spy_r is not None and qqq_r is not None:
        lag_b = _rolling_corr(spy_r, qqq_r.shift(1), 20)
        out["lag_drift_score"] = _safe_norm01((lag_b - lag_b.shift(5)).abs())
    else:
        out["lag_drift_score"] = 0.5

    ret_cols = [f"{a}_ret_1d" for a in SECTOR_ASSETS if f"{a}_ret_1d" in out.columns]
    if ret_cols:
        ent = out[ret_cols].apply(lambda row: _sector_row_entropy(row), axis=1)
        out["sector_entropy"] = _safe_norm01(ent)
    else:
        out["sector_entropy"] = 0.5

    if "sector_concentration_top2" in out.columns:
        out["sector_concentration_score"] = out["sector_concentration_top2"].clip(0.0, 1.0)
    else:
        out["sector_concentration_score"] = 0.5

    # Coherence: low std of core equity same-day returns => high coherence
    core_cols = [c for c in ["spy_ret_1d", "qqq_ret_1d", "iwm_ret_1d"] if c in out.columns]
    if len(core_cols) >= 2:
        stacked = out[core_cols].astype(float)
        cs = stacked.std(axis=1).rolling(10, min_periods=3).mean()
        out["coherence_score"] = (1.0 - _safe_norm01(cs.fillna(cs.median()))).clip(0.0, 1.0)
    else:
        out["coherence_score"] = 0.5

    # Instability: vol + dispersion + drift + breadth churn
    parts: list[pd.Series] = []
    if "spy_vol_20d" in out.columns:
        parts.append(_safe_norm01(out["spy_vol_20d"].ffill().fillna(0.0)))
    if "sector_dispersion_1d" in out.columns:
        parts.append(_safe_norm01(out["sector_dispersion_1d"].fillna(0.0)))
    parts.append(out["corr_drift_score"])
    parts.append(out["lag_drift_score"])
    if "breadth_pct_above_20dma" in out.columns:
        b = out["breadth_pct_above_20dma"].fillna(50.0)
        parts.append(_safe_norm01(b.diff().abs().fillna(0.0)))

    if parts:
        inst = sum(parts) / len(parts)
        out["instability_score"] = inst.clip(0.0, 1.0)
    else:
        out["instability_score"] = 0.5

    return out.loc[:, ~out.columns.duplicated()]
