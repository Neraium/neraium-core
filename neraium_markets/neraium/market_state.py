"""Day 8: market-wide state synthesis, market action, and usefulness comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.evaluation import score_action_usefulness
from neraium.transitions import compute_regime_runs, summarize_regime_persistence


def attach_asset_forward_returns(
    panel: pd.DataFrame,
    merged_prices: pd.DataFrame,
    assets: list[str],
    horizons: tuple[int, ...] = (1, 5, 10),
) -> pd.DataFrame:
    """
    Add ``fwd_ret_{h}d`` per row using each asset's price column in ``merged_prices``.
    """
    out_parts: list[pd.DataFrame] = []
    merged_prices = merged_prices.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    for a in assets:
        sub = panel[panel["asset"] == a].copy().sort_values(DATE_COLUMN, ascending=True)
        sym = a.upper() if a.upper() in merged_prices.columns else a
        if sym not in merged_prices.columns:
            continue
        m = sub.merge(merged_prices[[DATE_COLUMN, sym]], on=DATE_COLUMN, how="left")
        m = m.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
        px = m[sym].astype(float)
        for h in horizons:
            if h <= 0:
                raise ValueError("horizons must be positive")
            m[f"fwd_ret_{h}d"] = (px.shift(-h) / px) - 1.0
        m = m.drop(columns=[sym], errors="ignore")
        out_parts.append(m)

    if not out_parts:
        return panel.copy()
    combined = pd.concat(out_parts, ignore_index=True)
    return combined.sort_values([DATE_COLUMN, "asset"], ascending=True).reset_index(drop=True).loc[
        :, ~combined.columns.duplicated()
    ]


def score_panel_action_usefulness(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply Day 5 usefulness scoring per asset group."""
    parts: list[pd.DataFrame] = []
    for a, g in panel.groupby("asset", sort=False):
        g2 = g.sort_values(DATE_COLUMN, ascending=True)
        g2 = score_action_usefulness(g2)
        parts.append(g2)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values([DATE_COLUMN, "asset"], ascending=True).reset_index(drop=True).loc[
        :, ~out.columns.duplicated()
    ]


def aggregate_panel_per_timestamp(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp with cross-sectional regime fractions and means."""
    rows: list[dict[str, object]] = []
    for ts, g in panel.groupby(DATE_COLUMN, sort=True):
        rl = g["regime_label"].astype(str)
        rows.append(
            {
                DATE_COLUMN: ts,
                "regime_risk_off_frac": float((rl == "risk_off_transition").mean()),
                "regime_high_vol_frac": float((rl == "high_volatility").mean()),
                "regime_stable_frac": float((rl == "stable_trend").mean()),
                "regime_unstable_frac": float((rl == "unstable").mean()),
                "regime_fragile_frac": float((rl == "fragile_rally").mean()),
                "avg_panel_confidence": float(g["confidence_score"].astype(float).mean()),
                "avg_panel_instability": float(g["instability_score"].astype(float).mean()),
                "avg_panel_coherence": float(g["coherence_score"].astype(float).mean()),
                "n_assets": int(g["asset"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def synthesize_market_state(
    df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    influence_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    ``df`` must be the structural timeline **left-merged** with
    ``aggregate_panel_per_timestamp`` columns on ``timestamp``.

    Assigns ``market_regime_label`` plus market-level confidence / instability / coherence.

    Labels (deterministic priority):
    - **market_risk_off** — elevated risk-off + unstable share or high average instability
    - **market_stable** — broad stable_trend share + calm average instability + coherent panel
    - **market_fragmented** — low stable share and low coherence
    - **market_transition** — many fragile / unstable / vol names without full risk-off
    - **market_fragile** — fragile_rally share elevated with moderate stress
    - **market_mixed** — default
    """
    merged = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    # Optional: weight by top influencers in risk-off (simple proxy)
    top_inf = 0.0
    if not influence_df.empty and "asset" in influence_df.columns and "influence_score" in influence_df.columns:
        top_inf = float(influence_df["influence_score"].head(3).mean()) if len(influence_df) else 0.0

    rr = merged.get("regime_risk_off_frac", pd.Series(0.0, index=merged.index)).astype(float).fillna(0.0)
    hv = merged.get("regime_high_vol_frac", pd.Series(0.0, index=merged.index)).astype(float).fillna(0.0)
    st = merged.get("regime_stable_frac", pd.Series(0.0, index=merged.index)).astype(float).fillna(0.0)
    ru = merged.get("regime_unstable_frac", pd.Series(0.0, index=merged.index)).astype(float).fillna(0.0)
    rf = merged.get("regime_fragile_frac", pd.Series(0.0, index=merged.index)).astype(float).fillna(0.0)
    ainst = merged.get("avg_panel_instability", merged["instability_score"]).astype(float).fillna(0.5)
    acoh = merged.get("avg_panel_coherence", merged["coherence_score"]).astype(float).fillna(0.5)
    aconf = merged.get("avg_panel_confidence", merged.get("confidence_score", pd.Series(0.5, index=merged.index))).astype(float).fillna(0.5)

    labels = np.full(len(merged), "market_mixed", dtype=object)

    m_risk = (rr >= 0.18) & ((ru + hv) >= 0.20 + 0.1 * top_inf)
    m_stable = (st >= 0.22) & (ainst <= 0.48) & (acoh >= 0.48)
    m_frag = (rf >= 0.15) & (ainst >= 0.42) & (rr < 0.15)
    m_trans = (ru + rf + hv) >= 0.35
    m_fragm = (st <= 0.12) & (acoh <= 0.46) & (rr < 0.12)

    labels[m_risk] = "market_risk_off"
    labels[m_stable & ~m_risk] = "market_stable"
    m_fragile_lab = m_frag & ~m_risk & ~m_stable
    labels[m_fragile_lab] = "market_fragile"
    labels[m_fragm & ~m_risk & ~m_stable & ~m_fragile_lab] = "market_fragmented"
    labels[m_trans & ~m_risk & ~m_stable & ~m_fragile_lab & ~m_fragm] = "market_transition"

    merged["market_regime_label"] = labels
    merged["market_confidence_score"] = aconf.clip(0.0, 1.0)
    merged["market_instability_score"] = ainst.clip(0.0, 1.0)
    merged["market_coherence_score"] = acoh.clip(0.0, 1.0)

    # Dominant cluster regime hint (optional soft bias)
    if not cluster_summary_df.empty and "dominant_regime" in cluster_summary_df.columns:
        dom = cluster_summary_df["dominant_regime"].astype(str).mode()
        hint = str(dom.iloc[0]) if len(dom) else ""
        if hint in ("risk_off_transition", "high_volatility") and (merged["market_regime_label"] == "market_mixed").any():
            merged.loc[merged["market_regime_label"] == "market_mixed", "market_regime_label"] = "market_transition"

    return merged.loc[:, ~merged.columns.duplicated()]


def generate_market_action_posture(market_state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ``market_regime_label`` and market scores into ``market_action_posture``.

    Values: ``risk_on``, ``cautious_risk_on``, ``neutral``, ``reduce_risk``, ``defensive``, ``wait``.
    """
    out = market_state_df.copy()
    mr = out["market_regime_label"].astype(str).to_numpy()
    conf = out["market_confidence_score"].astype(float).to_numpy()
    inst = out["market_instability_score"].astype(float).to_numpy()

    posture = np.full(len(out), "neutral", dtype=object)
    high_c = conf >= 0.58

    m_st = mr == "market_stable"
    posture[m_st & high_c] = "risk_on"
    posture[m_st & ~high_c] = "cautious_risk_on"
    m_fr = mr == "market_fragile"
    posture[m_fr] = np.where(inst[m_fr] >= 0.55, "reduce_risk", "cautious_risk_on")
    posture[mr == "market_risk_off"] = "defensive"
    posture[mr == "market_transition"] = "wait"
    posture[mr == "market_fragmented"] = "wait"
    m_mx = mr == "market_mixed"
    posture[m_mx] = np.where(inst[m_mx] >= 0.55, "reduce_risk", "neutral")

    out["market_action_posture"] = posture.astype(str)
    return out


def generate_market_explanation(market_state_df: pd.DataFrame) -> pd.DataFrame:
    """Short deterministic narrative from market regime and scores."""
    out = market_state_df.copy()
    lines: list[str] = []
    for _, row in out.iterrows():
        lab = str(row.get("market_regime_label", ""))
        conf = float(row.get("market_confidence_score", float("nan")))
        inst = float(row.get("market_instability_score", float("nan")))
        coh = float(row.get("market_coherence_score", float("nan")))
        parts = [
            f"Market state={lab}",
            f"market_confidence={conf:.2f}",
            f"market_instability={inst:.2f}",
            f"market_coherence={coh:.2f}",
        ]
        if lab == "market_risk_off":
            parts.append(
                "Many tracked names show risk-off or stress; cross-asset coherence is weak; defensive posture."
            )
        elif lab == "market_stable":
            parts.append("Clusters align with stable trend; coherence is supportive; risk-on bias when confidence is high.")
        elif lab == "market_fragmented":
            parts.append("Clusters disagree; structural coherence is low; prefer to wait for alignment.")
        elif lab == "market_transition":
            parts.append("Fragile and unstable regimes are common; transitions dominate; wait for clarity.")
        elif lab == "market_fragile":
            parts.append("Fragile rally signals are elevated; reduce risk or lean cautious.")
        else:
            parts.append("Mixed cross-section without a single dominant narrative; neutral stance.")
        lines.append("; ".join(parts))
    out["market_explanation"] = lines
    return out


def attach_spy_forward_returns(market_state_df: pd.DataFrame, spy_forwards_df: pd.DataFrame) -> pd.DataFrame:
    """Merge ``fwd_ret_*`` and ``spy_vol_10d`` (if present) from the SPY evaluation frame."""
    cols = [DATE_COLUMN] + [c for c in spy_forwards_df.columns if c.startswith("fwd_ret_") and c.endswith("d")]
    if "spy_vol_10d" in spy_forwards_df.columns:
        cols.append("spy_vol_10d")
    cols = [c for c in cols if c in spy_forwards_df.columns]
    if len(cols) <= 1:
        return market_state_df.copy()
    sub = spy_forwards_df[cols].drop_duplicates(DATE_COLUMN)
    return market_state_df.merge(sub, on=DATE_COLUMN, how="left")


def compare_market_vs_asset_usefulness(
    asset_df: pd.DataFrame,
    market_state_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Two-row summary: ``asset_level`` vs ``market_level`` on usefulness, confidence,
    abstention, persistence proxy, false-positive proxy.
    """
    useful_cols = [c for c in asset_df.columns if c.startswith("action_useful_") and c.endswith("d")]
    if useful_cols:
        asset_u = asset_df[useful_cols].mean(axis=1)
        avg_use_a = float(asset_u.mean())
        abstain_a = float((asset_u == 0).mean())
    else:
        avg_use_a = float("nan")
        abstain_a = float("nan")

    avg_conf_a = float(asset_df["confidence_score"].astype(float).mean()) if "confidence_score" in asset_df.columns else float("nan")

    # Persistence: mean run length per asset then mean
    prs = []
    for a, g in asset_df.groupby("asset", sort=False):
        runs = compute_regime_runs(g.assign(regime_label=g["regime_label"]))
        summ = summarize_regime_persistence(runs)
        if not summ.empty:
            prs.append(float(summ["avg_run_length"].mean()))
    persist_a = float(np.mean(prs)) if prs else float("nan")

    fp_a = float("nan")
    if "likely_false_positive_flag" in asset_df.columns:
        fp_a = float(asset_df["likely_false_positive_flag"].astype(float).mean())

    def _map_market_posture_to_action(p: str) -> str:
        m = {
            "risk_on": "lean_long",
            "cautious_risk_on": "watch",
            "neutral": "watch",
            "reduce_risk": "reduce_exposure",
            "defensive": "avoid_risk",
            "wait": "wait",
        }
        return m.get(str(p), "watch")

    mdf = market_state_df.copy()
    mdf = mdf.sort_values(DATE_COLUMN, ascending=True)
    if "market_action_posture" not in mdf.columns:
        raise KeyError("market_state_df must include market_action_posture")
    mdf = mdf.assign(action_posture=mdf["market_action_posture"].map(_map_market_posture_to_action))
    mdf = score_action_usefulness(mdf)
    mu_cols = [c for c in mdf.columns if c.startswith("action_useful_") and c.endswith("d")]
    if mu_cols:
        mu = mdf[mu_cols].mean(axis=1)
        avg_use_m = float(mu.mean())
        abstain_m = float((mu == 0).mean())
    else:
        avg_use_m = float("nan")
        abstain_m = float("nan")

    avg_conf_m = (
        float(mdf["market_confidence_score"].astype(float).mean())
        if "market_confidence_score" in mdf.columns
        else float("nan")
    )

    pr_m = float("nan")
    if "market_regime_label" in mdf.columns:
        tmp = compute_regime_runs(mdf.assign(regime_label=mdf["market_regime_label"]))
        sm = summarize_regime_persistence(tmp)
        if not sm.empty:
            pr_m = float(sm["avg_run_length"].mean())

    fp_m = 0.0

    return pd.DataFrame(
        [
            {
                "level": "asset_level",
                "avg_usefulness": avg_use_a,
                "avg_confidence": avg_conf_a,
                "abstention_rate": abstain_a,
                "regime_persistence_proxy": persist_a,
                "false_positive_rate_proxy": fp_a,
            },
            {
                "level": "market_level",
                "avg_usefulness": avg_use_m,
                "avg_confidence": avg_conf_m,
                "abstention_rate": abstain_m,
                "regime_persistence_proxy": pr_m,
                "false_positive_rate_proxy": fp_m,
            },
        ]
    )
