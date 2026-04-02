"""Day 8: asset similarity, structural clusters, and cluster summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.regime import (
    apply_interpretive_gate,
    classify_regime,
    compute_confidence_score,
    generate_action_posture,
)


def _safe_norm01(s: pd.Series) -> pd.Series:
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def _per_asset_structural(base: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Build a per-asset view of the structural snapshot for regime rules.

    Blends market-wide instability with asset-specific realized volatility and
    uses the asset's 1d return for interpretive gate conflict checks (via ``spy_ret_1d`` slot).
    """
    out = base.copy()
    ret_col = f"{asset}_ret_1d"
    if ret_col not in out.columns:
        ret_col = "spy_ret_1d"
    out["spy_ret_1d"] = out[ret_col].astype(float)

    vol_col = f"{asset}_vol_20d"
    if vol_col not in out.columns:
        vol_col = "spy_vol_20d" if "spy_vol_20d" in out.columns else None
    if vol_col is not None:
        mix = _safe_norm01(out[vol_col].astype(float).fillna(0.0))
        out["instability_score"] = 0.5 * out["instability_score"].astype(float) + 0.5 * mix.clip(0.0, 1.0)
    return out


def _fix_concat_columns(pieces: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(pieces, ignore_index=True)
    return out.loc[:, ~out.columns.duplicated()]


def build_asset_panel(structural_df: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """
    Run regime → confidence → gate → posture per asset on the shared timeline.

    Returns long-form rows: ``timestamp``, ``asset``, ``regime_label``, ``action_posture``,
    ``confidence_score``, ``instability_score``, ``coherence_score``.
    """
    pieces: list[pd.DataFrame] = []
    for a in assets:
        sub = _per_asset_structural(structural_df, a)
        sub = classify_regime(sub)
        sub = compute_confidence_score(sub)
        sub = apply_interpretive_gate(sub)
        sub = generate_action_posture(sub)
        sub["asset"] = a
        pieces.append(
            sub[
                [
                    DATE_COLUMN,
                    "asset",
                    "regime_label",
                    "action_posture",
                    "confidence_score",
                    "instability_score",
                    "coherence_score",
                ]
            ]
        )
    return _fix_concat_columns(pieces).sort_values([DATE_COLUMN, "asset"], ascending=True).reset_index(drop=True)


def compute_asset_similarity_matrix(
    df: pd.DataFrame,
    assets: list[str],
    structural_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Square similarity matrix in ``[0, 1]`` combining:

    - rolling return correlation proxy (Pearson on ``{a}_ret_1d`` aligned series)
    - regime agreement rate (same ``regime_label`` on common timestamps)
    - inverse mean absolute confidence gap (smoothed to [0,1])

    ``df`` is the long asset panel from ``build_asset_panel``.
    ``structural_df`` supplies per-asset returns for correlation.
    """
    n = len(assets)
    mat = pd.DataFrame(np.eye(n), index=assets, columns=assets, dtype=float)

    ret_cols = {a: f"{a}_ret_1d" for a in assets}
    for a in assets:
        if ret_cols[a] not in structural_df.columns:
            ret_cols[a] = "spy_ret_1d"

    for i, a in enumerate(assets):
        for j, b in enumerate(assets):
            if i >= j:
                continue
            ra = structural_df[ret_cols[a]].astype(float)
            rb = structural_df[ret_cols[b]].astype(float)
            mask = ra.notna() & rb.notna()
            if mask.sum() >= 5:
                corr = float(ra[mask].corr(rb[mask]))
                corr_sim = (corr + 1.0) / 2.0
            else:
                corr_sim = 0.5

            pa = df[df["asset"] == a].set_index(DATE_COLUMN)["regime_label"].astype(str)
            pb = df[df["asset"] == b].set_index(DATE_COLUMN)["regime_label"].astype(str)
            al, bl = pa.align(pb, join="inner")
            regime_sim = float((al == bl).mean()) if len(al) else 0.0

            ca = df[df["asset"] == a].set_index(DATE_COLUMN)["confidence_score"].astype(float)
            cb = df[df["asset"] == b].set_index(DATE_COLUMN)["confidence_score"].astype(float)
            cl, dl = ca.align(cb, join="inner")
            if len(cl):
                mad = float(np.mean(np.abs(cl - dl)))
                conf_sim = float(max(0.0, 1.0 - mad))
            else:
                conf_sim = 0.5

            s = 0.40 * corr_sim + 0.35 * regime_sim + 0.25 * conf_sim
            s = float(np.clip(s, 0.0, 1.0))
            mat.loc[a, b] = mat.loc[b, a] = s

    return mat


def cluster_assets(similarity_df: pd.DataFrame, threshold: float = 0.52) -> pd.DataFrame:
    """
    Deterministic threshold graph: connect pairs with similarity >= ``threshold``,
    label connected components as ``cluster_id`` (0-based integer).
    """
    assets = list(similarity_df.index)
    adj: dict[str, set[str]] = {a: set() for a in assets}
    for i, a in enumerate(assets):
        for b in assets[i + 1 :]:
            if float(similarity_df.loc[a, b]) >= threshold:
                adj[a].add(b)
                adj[b].add(a)

    visited: set[str] = set()
    cluster_id = 0
    rows: list[dict[str, object]] = []
    for a in sorted(assets):
        if a in visited:
            continue
        stack = [a]
        comp: list[str] = []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            for v in sorted(adj[u]):
                if v not in visited:
                    stack.append(v)
        for x in comp:
            rows.append({"asset": x, "cluster_id": cluster_id})
        cluster_id += 1

    return pd.DataFrame(rows).sort_values(["cluster_id", "asset"]).reset_index(drop=True)


def summarize_clusters(clustered_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ``cluster_id``: member list, dominant regime, average confidence,
    average usefulness (if ``action_useful_*`` present), dominant action posture.
    """
    merged = signal_df.merge(clustered_df, on="asset", how="left")
    merged["cluster_id"] = merged["cluster_id"].fillna(-1)
    useful_cols = [c for c in merged.columns if c.startswith("action_useful_") and c.endswith("d")]

    def _dominant(s: pd.Series) -> str:
        s = s.dropna()
        if s.empty:
            return ""
        m = s.mode()
        return str(m.iloc[0]) if len(m) else ""

    rows: list[dict[str, object]] = []
    for cid, g in merged.groupby("cluster_id", sort=True):
        members = sorted(g["asset"].unique().tolist())
        dom_reg = _dominant(g["regime_label"].astype(str))
        dom_act = _dominant(g["action_posture"].astype(str))
        avg_conf = float(g["confidence_score"].astype(float).mean())
        u_avg = float(g[useful_cols].mean(axis=1).mean()) if useful_cols else float("nan")
        rows.append(
            {
                "cluster_id": int(cid) if not pd.isna(cid) else -1,
                "member_assets": ",".join(members),
                "n_members": len(members),
                "dominant_regime": dom_reg,
                "avg_confidence": avg_conf,
                "avg_usefulness": u_avg,
                "dominant_action_posture": dom_act,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
