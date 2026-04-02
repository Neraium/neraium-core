from __future__ import annotations

"""Day 3 structural scoring utilities for Neraium Markets."""

import math
import numpy as np
import pandas as pd

from config import (
    BASELINE_WINDOW,
    DATE_COLUMN,
    LEAD_LAG_PAIRS,
    MAX_LAG,
    RECENT_WINDOW,
    SECTOR_ASSETS,
    STRUCTURE_ASSETS,
)


def _resolve_return_col(df: pd.DataFrame, asset: str) -> str | None:
    """Return the 1-day return column name for an asset if present."""
    candidates = [f"{asset.lower()}_ret_1d", f"{asset.upper()}_ret_1d"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _zscore_bound(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """Bounded z-score used to keep blends deterministic and stable."""
    numeric = pd.to_numeric(series, errors="coerce")
    mu = numeric.expanding(min_periods=10).mean()
    sigma = numeric.expanding(min_periods=10).std(ddof=0).replace(0.0, np.nan)
    z = (numeric - mu) / sigma
    z = z.clip(-clip, clip)
    return (z + clip) / (2 * clip)


def compute_correlation_matrices(
    df: pd.DataFrame,
    assets: list[str],
    baseline_window: int = 60,
    recent_window: int = 20,
) -> pd.DataFrame:
    """Compute a rolling correlation drift score from baseline vs recent windows."""
    out = pd.DataFrame({DATE_COLUMN: df[DATE_COLUMN]})
    ret_cols = [c for a in assets if (c := _resolve_return_col(df, a))]
    out["corr_drift_score"] = np.nan

    if len(ret_cols) < 2:
        return out

    returns = df[ret_cols].astype(float)
    dim = len(ret_cols)
    denom = math.sqrt(float(dim * dim))

    start_idx = max(baseline_window, recent_window) - 1
    for idx in range(start_idx, len(df)):
        baseline = returns.iloc[idx - baseline_window + 1 : idx + 1]
        recent = returns.iloc[idx - recent_window + 1 : idx + 1]
        if baseline.isna().all().all() or recent.isna().all().all():
            continue

        b_corr = baseline.corr().fillna(0.0).to_numpy(dtype=float)
        r_corr = recent.corr().fillna(0.0).to_numpy(dtype=float)
        score = np.linalg.norm(r_corr - b_corr, ord="fro") / denom
        out.at[idx, "corr_drift_score"] = float(score)

    return out


def _best_lag_relationship(x: pd.Series, y: pd.Series, max_lag: int) -> tuple[int, float]:
    """Find lag in [-max_lag, max_lag] with highest absolute correlation."""
    best_lag = 0
    best_corr = 0.0
    for lag in range(-max_lag, max_lag + 1):
        corr = x.corr(y.shift(lag))
        if pd.isna(corr):
            continue
        if abs(float(corr)) > abs(best_corr):
            best_corr = float(corr)
            best_lag = lag
    return best_lag, best_corr


def compute_lead_lag_drift(
    df: pd.DataFrame,
    asset_pairs: list[tuple[str, str]],
    baseline_window: int = 60,
    recent_window: int = 20,
    max_lag: int = 3,
) -> pd.DataFrame:
    """Compute rolling lead-lag drift via changes in best lagged correlation structure."""
    out = pd.DataFrame({DATE_COLUMN: df[DATE_COLUMN]})
    out["lag_drift_score"] = np.nan

    start_idx = max(baseline_window, recent_window) - 1
    for idx in range(start_idx, len(df)):
        baseline_penalties: list[float] = []
        for left, right in asset_pairs:
            left_col = _resolve_return_col(df, left)
            right_col = _resolve_return_col(df, right)
            if not left_col or not right_col:
                continue

            b_left = df[left_col].iloc[idx - baseline_window + 1 : idx + 1]
            b_right = df[right_col].iloc[idx - baseline_window + 1 : idx + 1]
            r_left = df[left_col].iloc[idx - recent_window + 1 : idx + 1]
            r_right = df[right_col].iloc[idx - recent_window + 1 : idx + 1]

            b_lag, b_corr = _best_lag_relationship(b_left, b_right, max_lag)
            r_lag, r_corr = _best_lag_relationship(r_left, r_right, max_lag)
            lag_change = abs(r_lag - b_lag) / max(1, max_lag)
            corr_change = abs(abs(r_corr) - abs(b_corr))
            baseline_penalties.append(0.5 * lag_change + 0.5 * corr_change)

        if baseline_penalties:
            out.at[idx, "lag_drift_score"] = float(np.mean(baseline_penalties))

    return out


def compute_concentration_entropy(df: pd.DataFrame, sector_assets: list[str]) -> pd.DataFrame:
    """Compute sector entropy and concentration from absolute 1-day returns."""
    out = pd.DataFrame({DATE_COLUMN: df[DATE_COLUMN]})
    ret_cols = [c for a in sector_assets if (c := _resolve_return_col(df, a))]

    if not ret_cols:
        out["sector_entropy"] = np.nan
        out["sector_concentration_score"] = np.nan
        return out

    abs_ret = df[ret_cols].abs().astype(float)
    row_sum = abs_ret.sum(axis=1)
    shares = abs_ret.div(row_sum.replace(0.0, np.nan), axis=0)

    n = len(ret_cols)
    entropy = -(shares * np.log(shares.clip(lower=1e-12))).sum(axis=1)
    out["sector_entropy"] = entropy / math.log(n) if n > 1 else np.nan
    out["sector_concentration_score"] = (shares.pow(2)).sum(axis=1)
    return out


def compute_instability_score(df: pd.DataFrame) -> pd.DataFrame:
    """Blend structural sub-scores into one deterministic instability score."""
    out = pd.DataFrame({DATE_COLUMN: df[DATE_COLUMN]})

    inverse_breadth = 1.0 - (pd.to_numeric(df.get("breadth_pct_above_20dma"), errors="coerce") / 100.0)
    components = {
        "corr": _zscore_bound(df.get("corr_drift_score", pd.Series(np.nan, index=df.index))),
        "lag": _zscore_bound(df.get("lag_drift_score", pd.Series(np.nan, index=df.index))),
        "conc": _zscore_bound(df.get("sector_concentration_score", pd.Series(np.nan, index=df.index))),
        "disp": _zscore_bound(df.get("sector_dispersion_1d", pd.Series(np.nan, index=df.index))),
        "breadth": _zscore_bound(inverse_breadth),
        "risk": _zscore_bound(df.get("risk_off_proxy", pd.Series(np.nan, index=df.index))),
    }

    weights = {
        "corr": 0.25,
        "lag": 0.15,
        "conc": 0.15,
        "disp": 0.15,
        "breadth": 0.15,
        "risk": 0.15,
    }
    total = sum(weights.values())
    score = sum(weights[k] * components[k] for k in components) / total
    out["instability_score"] = score.clip(0.0, 1.0)
    return out


def compute_coherence_score(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate directional agreement across structural stress components."""
    out = pd.DataFrame({DATE_COLUMN: df[DATE_COLUMN]})

    hi_corr = _zscore_bound(df.get("corr_drift_score", pd.Series(np.nan, index=df.index)))
    hi_lag = _zscore_bound(df.get("lag_drift_score", pd.Series(np.nan, index=df.index)))
    hi_conc = _zscore_bound(df.get("sector_concentration_score", pd.Series(np.nan, index=df.index)))
    lo_breadth = _zscore_bound(1.0 - pd.to_numeric(df.get("breadth_pct_above_20dma"), errors="coerce") / 100.0)
    hi_risk = _zscore_bound(df.get("risk_off_proxy", pd.Series(np.nan, index=df.index)))

    stress_matrix = pd.concat([hi_corr, hi_lag, hi_conc, lo_breadth, hi_risk], axis=1)
    avg = stress_matrix.mean(axis=1)
    disagreement = stress_matrix.sub(avg, axis=0).abs().mean(axis=1)
    out["coherence_score"] = (1.0 - disagreement * 2.0).clip(0.0, 1.0)
    return out


def _safe_merge_on_timestamp(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    right_cols = [c for c in right.columns if c == DATE_COLUMN or c not in left.columns]
    return left.merge(right[right_cols], on=DATE_COLUMN, how="left")


def build_structural_snapshot(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Build full Day 3 structural snapshot from Day 2 feature table."""
    snapshot = feature_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    corr = compute_correlation_matrices(
        snapshot,
        assets=STRUCTURE_ASSETS,
        baseline_window=BASELINE_WINDOW,
        recent_window=RECENT_WINDOW,
    )
    lag = compute_lead_lag_drift(
        snapshot,
        asset_pairs=LEAD_LAG_PAIRS,
        baseline_window=BASELINE_WINDOW,
        recent_window=RECENT_WINDOW,
        max_lag=MAX_LAG,
    )
    concentration = compute_concentration_entropy(snapshot, sector_assets=SECTOR_ASSETS)

    snapshot = _safe_merge_on_timestamp(snapshot, corr)
    snapshot = _safe_merge_on_timestamp(snapshot, lag)
    snapshot = _safe_merge_on_timestamp(snapshot, concentration)

    instability = compute_instability_score(snapshot)
    coherence = compute_coherence_score(snapshot)

    snapshot = _safe_merge_on_timestamp(snapshot, instability)
    snapshot = _safe_merge_on_timestamp(snapshot, coherence)

    snapshot = snapshot.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    snapshot = snapshot.loc[:, ~snapshot.columns.duplicated()]
    return snapshot
