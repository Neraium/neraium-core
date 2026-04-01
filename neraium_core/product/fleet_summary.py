"""Fleet-level summaries and rankings from replay or batch tabular outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _score_row(row: pd.Series) -> float:
    """Deterministic priority score: higher means more attention (not a probability)."""
    state = str(row.get("state", "STABLE")).upper()
    pressure = float(row.get("transition_pressure", 0.0) or 0.0)
    drift = float(row.get("structural_drift_score", 0.0) or 0.0)
    pre = float(
        row.get(
            "early_warning_pre_instability_score",
            row.get("pre_instability_score", 0.0),
        )
        or 0.0
    )
    if state == "ALERT":
        sev = 3.0
    elif state == "WATCH":
        sev = 2.0
    else:
        sev = 1.0
    return sev * 10.0 + 4.0 * pressure + 0.4 * drift + 2.0 * pre


def rank_assets_by_priority(df: pd.DataFrame, *, asset_col: str = "unit") -> pd.DataFrame:
    """
    Per-asset aggregate priority from slim or diagnostics replay frames.

    Uses the worst (max) score across rows per asset, then ranks.
    """
    if asset_col not in df.columns:
        raise ValueError(f"Column {asset_col!r} not found in dataframe")
    work = df.copy()
    work["_priority_score"] = work.apply(_score_row, axis=1)
    agg = (
        work.groupby(asset_col, as_index=False)
        .agg(
            max_priority_score=("_priority_score", "max"),
            mean_transition_pressure=("transition_pressure", "mean"),
            max_structural_drift=("structural_drift_score", "max"),
        )
    )
    counts = work.groupby(asset_col).size().reset_index(name="row_count")
    out = agg.merge(counts, on=asset_col)
    out = out.sort_values("max_priority_score", ascending=False).reset_index(drop=True)
    out["fleet_rank"] = range(1, len(out) + 1)
    return out


def build_fleet_summary(df: pd.DataFrame, *, asset_col: str = "unit") -> dict[str, Any]:
    """
    Structured fleet summary for JSON export: ranking table plus counts.

    ``df`` is typically a slim or diagnostics replay DataFrame.
    """
    ranked = rank_assets_by_priority(df, asset_col=asset_col)
    return {
        "asset_count": int(len(ranked)),
        "ranked_assets": ranked.to_dict(orient="records"),
        "top_asset": ranked.iloc[0].to_dict() if len(ranked) else None,
    }
