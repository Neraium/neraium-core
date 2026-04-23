"""Day 8: regime change propagation and influence scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN


def build_regime_propagation_table(
    df: pd.DataFrame,
    assets: list[str],
    forward_window: int = 3,
) -> pd.DataFrame:
    """
    Count how often a regime change in ``source_asset`` is followed within ``forward_window``
    rows by a regime change in ``target_asset``.

    Long panel ``df`` must contain ``timestamp``, ``asset``, ``regime_label``.

    Output rows aggregate:
    ``source_asset``, ``target_asset``, ``source_regime`` (post-change label),
    ``target_regime`` (first new label observed for target), ``propagation_count``,
    ``avg_delay_steps``.
    """
    if DATE_COLUMN not in df.columns or "asset" not in df.columns or "regime_label" not in df.columns:
        raise KeyError("Panel must include timestamp, asset, regime_label")

    ts_list = sorted(df[DATE_COLUMN].unique())
    n = len(ts_list)
    if n < 2:
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

    wide: dict[str, np.ndarray] = {}
    for a in assets:
        g = df[df["asset"] == a].drop_duplicates(DATE_COLUMN).set_index(DATE_COLUMN)["regime_label"].astype(str)
        aligned = g.reindex(ts_list)
        wide[a] = aligned.to_numpy()

    events: list[tuple[int, str, str, str]] = []
    for ii in range(1, n):
        for a in assets:
            prev = wide[a][ii - 1]
            cur = wide[a][ii]
            if pd.isna(prev) or pd.isna(cur):
                continue
            if prev != cur:
                events.append((ii, a, str(prev), str(cur)))

    records: list[dict[str, object]] = []
    for ii, src, _prev_s, new_s in events:
        for tgt in assets:
            if tgt == src:
                continue
            r0 = wide[tgt][ii - 1]
            if pd.isna(r0):
                continue
            r0 = str(r0)
            for k in range(ii + 1, min(ii + forward_window + 1, n)):
                rk = wide[tgt][k]
                if pd.isna(rk):
                    continue
                rk = str(rk)
                if rk != r0:
                    delay = k - ii
                    records.append(
                        {
                            "source_asset": src,
                            "target_asset": tgt,
                            "source_regime": new_s,
                            "target_regime": rk,
                            "delay_steps": delay,
                        }
                    )
                    break

    if not records:
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

    rdf = pd.DataFrame.from_records(records)
    g = rdf.groupby(
        ["source_asset", "target_asset", "source_regime", "target_regime"],
        observed=False,
    ).agg(propagation_count=("delay_steps", "size"), avg_delay_steps=("delay_steps", "mean"))
    return g.reset_index().sort_values("propagation_count", ascending=False).reset_index(drop=True)


def compute_asset_influence_scores(propagation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Outgoing propagation counts per source asset and a simple influence score.

    Higher score when many propagations fire and delays are short (leading behavior).
    """
    if propagation_df.empty:
        return pd.DataFrame(
            columns=["asset", "influence_score", "outgoing_propagation_count", "avg_lead_time"]
        )

    g = propagation_df.groupby("source_asset", observed=False).agg(
        outgoing_propagation_count=("propagation_count", "sum"),
        avg_lead_time=("avg_delay_steps", "mean"),
    )
    g = g.reset_index().rename(columns={"source_asset": "asset"})

    oc = g["outgoing_propagation_count"].astype(float)
    mx = float(oc.max()) if len(oc) else 1.0
    norm_out = oc / mx if mx > 0 else oc * 0.0
    alt = g["avg_lead_time"].astype(float)
    # shorter delay -> larger weight
    speed = 1.0 / (1.0 + alt)
    smax = float(speed.max()) if len(speed) else 1.0
    norm_speed = speed / smax if smax > 0 else speed * 0.0
    g["influence_score"] = (0.65 * norm_out + 0.35 * norm_speed).clip(0.0, 1.0)
    return g.sort_values("influence_score", ascending=False).reset_index(drop=True)


def compute_sector_influence_scores(
    propagation_df: pd.DataFrame,
    asset_to_sector_map: dict[str, str],
) -> pd.DataFrame:
    """Aggregate propagation counts by source sector."""
    if propagation_df.empty:
        return pd.DataFrame(columns=["sector", "influence_score", "propagation_count"])

    p = propagation_df.copy()
    p["source_sector"] = p["source_asset"].map(asset_to_sector_map).fillna("unknown")
    rows: list[dict[str, object]] = []
    for sec, g in p.groupby("source_sector", sort=False):
        w = g["propagation_count"].astype(float)
        cnt = float(w.sum())
        if cnt <= 0:
            continue
        avg_d = float(np.average(g["avg_delay_steps"].astype(float), weights=w))
        rows.append({"sector": str(sec), "propagation_count": cnt, "avg_delay_steps": avg_d})
    g = pd.DataFrame(rows)
    if g.empty:
        return pd.DataFrame(columns=["sector", "influence_score", "propagation_count"])
    pc = g["propagation_count"].astype(float)
    mx = float(pc.max()) if len(pc) else 1.0
    g["influence_score"] = (pc / mx if mx > 0 else pc * 0.0).clip(0.0, 1.0)
    return g.sort_values("influence_score", ascending=False).reset_index(drop=True)
