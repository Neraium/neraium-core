"""Day 8 cross-asset similarity, clustering, and cluster summarization."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from config import DATE_COLUMN


def _mode(values: Iterable[str], default: str = "unknown") -> str:
    vals = [str(v) for v in values if pd.notna(v)]
    if not vals:
        return default
    return Counter(vals).most_common(1)[0][0]


def compute_asset_similarity_matrix(df: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Compute deterministic structural similarity across assets.

    Expected long-form columns (best effort):
    - timestamp, asset
    - ret_1d (or return_1d)
    - regime_label
    - action_posture
    - structural_score (optional)

    Similarity is the mean of available components in [0, 1]:
    1) return-correlation mapped from [-1, 1] -> [0, 1]
    2) regime sequence exact-match ratio
    3) action sequence exact-match ratio
    4) structural-score distance similarity (1 - normalized mean abs diff)
    """
    required = {DATE_COLUMN, "asset"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], utc=False)
    data = data[data["asset"].isin(assets)].sort_values([DATE_COLUMN, "asset"], ascending=True)

    ret_col = "ret_1d" if "ret_1d" in data.columns else "return_1d" if "return_1d" in data.columns else None
    reg_col = "regime_label" if "regime_label" in data.columns else None
    act_col = "action_posture" if "action_posture" in data.columns else None
    score_col = "structural_score" if "structural_score" in data.columns else None

    by_asset = {a: data[data["asset"] == a].set_index(DATE_COLUMN).sort_index() for a in assets}

    out = pd.DataFrame(index=assets, columns=assets, dtype=float)
    for a in assets:
        for b in assets:
            if a == b:
                out.loc[a, b] = 1.0
                continue

            comp: list[float] = []
            left = by_asset.get(a, pd.DataFrame())
            right = by_asset.get(b, pd.DataFrame())
            if left.empty or right.empty:
                out.loc[a, b] = 0.0
                continue

            if ret_col:
                merged = left[[ret_col]].join(right[[ret_col]], how="inner", lsuffix="_a", rsuffix="_b")
                if len(merged) >= 3:
                    corr = merged[f"{ret_col}_a"].corr(merged[f"{ret_col}_b"])
                    if pd.notna(corr):
                        comp.append(float(np.clip((float(corr) + 1.0) / 2.0, 0.0, 1.0)))

            if reg_col:
                merged = left[[reg_col]].join(right[[reg_col]], how="inner", lsuffix="_a", rsuffix="_b")
                if not merged.empty:
                    reg_sim = (merged[f"{reg_col}_a"].astype(str) == merged[f"{reg_col}_b"].astype(str)).mean()
                    comp.append(float(np.clip(reg_sim, 0.0, 1.0)))

            if act_col:
                merged = left[[act_col]].join(right[[act_col]], how="inner", lsuffix="_a", rsuffix="_b")
                if not merged.empty:
                    act_sim = (merged[f"{act_col}_a"].astype(str) == merged[f"{act_col}_b"].astype(str)).mean()
                    comp.append(float(np.clip(act_sim, 0.0, 1.0)))

            if score_col:
                merged = left[[score_col]].join(right[[score_col]], how="inner", lsuffix="_a", rsuffix="_b")
                if not merged.empty:
                    diffs = (merged[f"{score_col}_a"].astype(float) - merged[f"{score_col}_b"].astype(float)).abs()
                    denom = float(max(diffs.max(skipna=True), 1e-9))
                    score_sim = 1.0 - float(np.nanmean(diffs / denom))
                    comp.append(float(np.clip(score_sim, 0.0, 1.0)))

            out.loc[a, b] = float(np.mean(comp)) if comp else 0.0

    out = out.fillna(0.0)
    out = (out + out.T) / 2.0
    for i, a in enumerate(out.index):
        out.iat[i, i] = 1.0
    return out.loc[assets, assets]


def cluster_assets(similarity_df: pd.DataFrame) -> pd.DataFrame:
    """Cluster assets via deterministic thresholded connectivity (graph components)."""
    if similarity_df.empty:
        return pd.DataFrame(columns=["asset", "cluster_id"])

    sim = similarity_df.copy()
    if not sim.index.equals(sim.columns):
        raise ValueError("similarity_df must be square with matching index/columns")

    assets = list(sim.index.astype(str))
    vals = sim.values.astype(float)
    off_diag = vals[~np.eye(vals.shape[0], dtype=bool)]
    threshold = float(np.clip(np.nanmedian(off_diag) if off_diag.size else 0.5, 0.35, 0.85))

    adjacency: dict[str, set[str]] = {a: set() for a in assets}
    for i, a in enumerate(assets):
        for j, b in enumerate(assets):
            if i == j:
                continue
            if float(sim.iloc[i, j]) >= threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

    cluster_ids: dict[str, int] = {}
    cur = 0
    for asset in sorted(assets):
        if asset in cluster_ids:
            continue
        cur += 1
        stack = [asset]
        while stack:
            node = stack.pop()
            if node in cluster_ids:
                continue
            cluster_ids[node] = cur
            stack.extend(sorted(adjacency[node] - set(cluster_ids.keys())))

    out = pd.DataFrame({"asset": sorted(assets)})
    out["cluster_id"] = out["asset"].map(cluster_ids).astype(int)
    return out.sort_values(["cluster_id", "asset"], ascending=True).reset_index(drop=True)


def summarize_clusters(clustered_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize cluster composition and dominant state characteristics."""
    req = {"asset", "cluster_id"}
    missing = req - set(clustered_df.columns)
    if missing:
        raise KeyError(f"Missing required clustered_df columns: {sorted(missing)}")

    if signal_df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "member_assets",
                "dominant_regime",
                "average_confidence",
                "average_usefulness",
                "dominant_action_posture",
                "asset_count",
            ]
        )

    data = signal_df.copy()
    if DATE_COLUMN in data.columns:
        data = data.sort_values(DATE_COLUMN, ascending=True)

    latest = data.groupby("asset", observed=False).tail(1)
    latest = latest.merge(clustered_df[["asset", "cluster_id"]], on="asset", how="inner")

    usefulness_col = (
        "action_useful_5d" if "action_useful_5d" in latest.columns else "usefulness_score" if "usefulness_score" in latest.columns else None
    )
    conf_col = (
        "adjusted_confidence_score"
        if "adjusted_confidence_score" in latest.columns
        else "confidence_score"
        if "confidence_score" in latest.columns
        else None
    )

    rows: list[dict[str, object]] = []
    for cid, grp in latest.groupby("cluster_id", observed=False):
        rows.append(
            {
                "cluster_id": int(cid),
                "member_assets": ",".join(sorted(grp["asset"].astype(str).tolist())),
                "dominant_regime": _mode(grp.get("regime_label", pd.Series(dtype=str))),
                "average_confidence": float(grp[conf_col].astype(float).mean()) if conf_col else float("nan"),
                "average_usefulness": float(grp[usefulness_col].astype(float).mean()) if usefulness_col else float("nan"),
                "dominant_action_posture": _mode(grp.get("action_posture", pd.Series(dtype=str))),
                "asset_count": int(grp["asset"].nunique()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["cluster_id"], ascending=True).reset_index(drop=True)
