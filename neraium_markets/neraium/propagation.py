"""Day 8 regime propagation and influence scoring utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN


def build_regime_propagation_table(df: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Build source->target propagation counts from regime change events.

    For each source regime change at time t, we look for the first target regime
    change in (t, t + window] and record delay steps. Deterministic window=3 rows.
    """
    req = {DATE_COLUMN, "asset", "regime_label"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    window = 3
    data = df.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], utc=False)
    data = data[data["asset"].isin(assets)].sort_values(["asset", DATE_COLUMN], ascending=True)

    by_asset: dict[str, pd.DataFrame] = {}
    for asset, grp in data.groupby("asset", observed=False):
        g = grp.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
        g["prev_regime"] = g["regime_label"].shift(1)
        g["regime_change_flag"] = (g["regime_label"].astype(str) != g["prev_regime"].astype(str)).astype(int)
        g.loc[g.index[0], "regime_change_flag"] = 0
        by_asset[str(asset)] = g

    rows: list[dict[str, object]] = []
    for source in assets:
        src = by_asset.get(source)
        if src is None or src.empty:
            continue
        src_events = src[src["regime_change_flag"] == 1]

        for target in assets:
            if source == target:
                continue
            tgt = by_asset.get(target)
            if tgt is None or tgt.empty:
                continue
            tgt_events = tgt[tgt["regime_change_flag"] == 1]
            if tgt_events.empty:
                continue

            for _, ev in src_events.iterrows():
                t = ev[DATE_COLUMN]
                source_regime = str(ev["regime_label"])
                future = tgt_events[(tgt_events[DATE_COLUMN] > t) & (tgt_events[DATE_COLUMN] <= t + pd.Timedelta(days=window * 2))]
                if future.empty:
                    continue
                first = future.iloc[0]
                delay = int(max((first[DATE_COLUMN] - t).days, 1))
                rows.append(
                    {
                        "source_asset": str(source),
                        "target_asset": str(target),
                        "source_regime": source_regime,
                        "target_regime": str(first["regime_label"]),
                        "delay_steps": delay,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_asset",
                "target_asset",
                "source_regime",
                "target_regime",
                "propagation_count",
                "avg_delay_steps",
            ]
        )

    events = pd.DataFrame(rows)
    out = (
        events.groupby(["source_asset", "target_asset", "source_regime", "target_regime"], observed=False)
        .agg(propagation_count=("delay_steps", "size"), avg_delay_steps=("delay_steps", "mean"))
        .reset_index()
        .sort_values(["propagation_count", "avg_delay_steps", "source_asset", "target_asset"], ascending=[False, True, True, True])
        .reset_index(drop=True)
    )
    out["avg_delay_steps"] = out["avg_delay_steps"].astype(float)
    return out


def compute_asset_influence_scores(propagation_df: pd.DataFrame) -> pd.DataFrame:
    """Compute source-asset influence from outgoing propagation intensity and speed."""
    if propagation_df.empty:
        return pd.DataFrame(columns=["asset", "influence_score", "outgoing_propagation_count", "avg_lead_time"])

    req = {"source_asset", "propagation_count", "avg_delay_steps"}
    missing = req - set(propagation_df.columns)
    if missing:
        raise KeyError(f"Missing required propagation_df columns: {sorted(missing)}")

    g = (
        propagation_df.groupby("source_asset", observed=False)
        .agg(
            outgoing_propagation_count=("propagation_count", "sum"),
            avg_lead_time=("avg_delay_steps", "mean"),
        )
        .reset_index()
        .rename(columns={"source_asset": "asset"})
    )

    g["outgoing_propagation_count"] = g["outgoing_propagation_count"].astype(float)
    g["avg_lead_time"] = g["avg_lead_time"].astype(float).clip(lower=1e-6)

    count_norm = g["outgoing_propagation_count"] / max(g["outgoing_propagation_count"].max(), 1.0)
    speed_norm = 1.0 / g["avg_lead_time"]
    speed_norm = speed_norm / max(speed_norm.max(), 1.0)

    g["influence_score"] = (0.7 * count_norm + 0.3 * speed_norm).clip(0.0, 1.0)
    return g.sort_values(["influence_score", "outgoing_propagation_count", "asset"], ascending=[False, False, True]).reset_index(drop=True)


def compute_sector_influence_scores(
    propagation_df: pd.DataFrame,
    asset_to_sector_map: dict[str, str],
) -> pd.DataFrame:
    """Aggregate propagation from source assets into sector-level influence."""
    if propagation_df.empty:
        return pd.DataFrame(columns=["sector", "influence_score", "propagation_count"])

    req = {"source_asset", "propagation_count", "avg_delay_steps"}
    missing = req - set(propagation_df.columns)
    if missing:
        raise KeyError(f"Missing required propagation_df columns: {sorted(missing)}")

    df = propagation_df.copy()
    df["sector"] = df["source_asset"].map(asset_to_sector_map).fillna("other")

    out = (
        df.groupby("sector", observed=False)
        .agg(
            propagation_count=("propagation_count", "sum"),
            avg_delay=("avg_delay_steps", "mean"),
        )
        .reset_index()
    )

    cnt_norm = out["propagation_count"] / max(out["propagation_count"].max(), 1.0)
    spd_norm = (1.0 / out["avg_delay"].clip(lower=1e-6))
    spd_norm = spd_norm / max(spd_norm.max(), 1.0)
    out["influence_score"] = (0.75 * cnt_norm + 0.25 * spd_norm).clip(0.0, 1.0)

    return out[["sector", "influence_score", "propagation_count"]].sort_values(
        ["influence_score", "propagation_count", "sector"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
