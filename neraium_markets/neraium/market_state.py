"""Day 8 market-wide state synthesis and usefulness comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthesize_market_state(
    df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    influence_df: pd.DataFrame,
) -> pd.DataFrame:
    """Synthesize market-level regime/confidence/instability/coherence from local states.

    Deterministic high-level logic:
    - risk_off_pressure: share of influential assets in risk_off/high_volatility/unstable
    - coherence: cluster dominance and confidence
    - instability: risk_off pressure + cluster dispersion

    Labels:
    - market_risk_off: high risk_off pressure
    - market_stable: high coherence, low instability
    - market_fragmented: low coherence and high dispersion
    - market_transition: elevated instability
    - market_fragile: medium instability + sub-high coherence
    - otherwise market_mixed
    """
    if df.empty:
        return pd.DataFrame(columns=["market_regime_label", "market_confidence_score", "market_instability_score", "market_coherence_score"])

    latest = df.groupby("asset", observed=False).tail(1).copy()

    inf = influence_df.copy()
    if inf.empty or "asset" not in inf.columns:
        inf = pd.DataFrame({"asset": latest["asset"].astype(str).unique(), "influence_score": 0.5})

    latest = latest.merge(inf[["asset", "influence_score"]], on="asset", how="left")
    latest["influence_score"] = latest["influence_score"].fillna(float(inf.get("influence_score", pd.Series([0.5])).mean()))

    riskish = latest["regime_label"].astype(str).isin({"risk_off_transition", "high_volatility", "unstable"}).astype(float)
    weight = latest["influence_score"].astype(float).clip(lower=1e-6)
    risk_off_pressure = float((riskish * weight).sum() / weight.sum()) if len(latest) else 0.0

    cluster_dispersion = 0.0
    if not cluster_summary_df.empty and "dominant_regime" in cluster_summary_df.columns:
        cluster_dispersion = float(cluster_summary_df["dominant_regime"].astype(str).nunique() / max(len(cluster_summary_df), 1))

    base_conf = (
        latest["adjusted_confidence_score"].astype(float).mean()
        if "adjusted_confidence_score" in latest.columns
        else latest.get("confidence_score", pd.Series([0.5])).astype(float).mean()
    )
    cluster_conf = (
        cluster_summary_df["average_confidence"].astype(float).mean()
        if not cluster_summary_df.empty and "average_confidence" in cluster_summary_df.columns
        else base_conf
    )

    coherence = float(np.clip(0.6 * cluster_conf + 0.4 * (1.0 - cluster_dispersion), 0.0, 1.0))
    instability = float(np.clip(0.65 * risk_off_pressure + 0.35 * cluster_dispersion, 0.0, 1.0))
    confidence = float(np.clip(0.55 * base_conf + 0.45 * coherence, 0.0, 1.0))

    if risk_off_pressure >= 0.55:
        label = "market_risk_off"
    elif coherence >= 0.70 and instability <= 0.35:
        label = "market_stable"
    elif coherence <= 0.45 and cluster_dispersion >= 0.55:
        label = "market_fragmented"
    elif instability >= 0.60:
        label = "market_transition"
    elif instability >= 0.45:
        label = "market_fragile"
    else:
        label = "market_mixed"

    return pd.DataFrame(
        {
            "market_regime_label": [label],
            "market_confidence_score": [confidence],
            "market_instability_score": [instability],
            "market_coherence_score": [coherence],
        }
    )


def generate_market_action_posture(market_state_df: pd.DataFrame) -> pd.DataFrame:
    """Map market regime + confidence into market action posture."""
    if market_state_df.empty:
        return pd.DataFrame(columns=["market_action_posture"])

    df = market_state_df.copy()

    def _map(row: pd.Series) -> str:
        label = str(row.get("market_regime_label", "market_mixed"))
        conf = float(row.get("market_confidence_score", 0.0))

        if label == "market_risk_off":
            return "defensive"
        if label == "market_stable" and conf >= 0.65:
            return "risk_on"
        if label == "market_fragile":
            return "cautious_risk_on" if conf >= 0.55 else "reduce_risk"
        if label in {"market_transition", "market_fragmented"}:
            return "wait"
        if label == "market_mixed":
            return "neutral"
        return "neutral"

    df["market_action_posture"] = df.apply(_map, axis=1)
    return df


def generate_market_explanation(market_state_df: pd.DataFrame) -> pd.DataFrame:
    """Generate a short deterministic market explanation string."""
    if market_state_df.empty:
        return pd.DataFrame(columns=["market_explanation"])

    df = market_state_df.copy()

    def _describe(row: pd.Series) -> str:
        label = str(row.get("market_regime_label", "market_mixed"))
        conf = float(row.get("market_confidence_score", 0.0))
        inst = float(row.get("market_instability_score", 0.0))
        coh = float(row.get("market_coherence_score", 0.0))
        action = str(row.get("market_action_posture", "neutral"))

        return (
            f"Market state is {label} (confidence={conf:.2f}, instability={inst:.2f}, coherence={coh:.2f}); "
            f"the synthesized market-wide posture is {action}."
        )

    df["market_explanation"] = df.apply(_describe, axis=1)
    return df


def compare_market_vs_asset_usefulness(
    asset_df: pd.DataFrame,
    market_state_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare asset-level and market-level usefulness diagnostics."""
    if asset_df.empty or market_state_df.empty:
        return pd.DataFrame(columns=["level", "average_usefulness", "average_confidence", "abstention_rate", "regime_persistence", "false_positive_rate"])

    a = asset_df.copy()
    usefulness_col = "action_useful_5d" if "action_useful_5d" in a.columns else "usefulness_score"
    conf_col = "adjusted_confidence_score" if "adjusted_confidence_score" in a.columns else "confidence_score"

    if "action_posture" in a.columns:
        abst = a["action_posture"].astype(str).isin({"wait", "watch"}).mean()
    else:
        abst = np.nan

    if "regime_label" in a.columns:
        ordered = a.sort_values(["asset", "timestamp"], ascending=True)
        changed = ordered.groupby("asset", observed=False)["regime_label"].apply(lambda s: (s.astype(str) != s.astype(str).shift(1)).astype(int))
        persistence = float(1.0 - changed.mean())
    else:
        persistence = np.nan

    if "likely_false_positive_flag" in a.columns:
        fp_rate = float(a["likely_false_positive_flag"].astype(float).mean())
    else:
        fp_rate = float(np.nan)

    asset_row = {
        "level": "asset_level",
        "average_usefulness": float(a[usefulness_col].astype(float).mean()) if usefulness_col in a.columns else np.nan,
        "average_confidence": float(a[conf_col].astype(float).mean()) if conf_col in a.columns else np.nan,
        "abstention_rate": float(abst),
        "regime_persistence": persistence,
        "false_positive_rate": fp_rate,
    }

    m = market_state_df.copy()
    market_row = {
        "level": "market_level",
        "average_usefulness": float(m.get("market_usefulness_proxy", pd.Series([asset_row["average_usefulness"]])).astype(float).mean()),
        "average_confidence": float(m.get("market_confidence_score", pd.Series([np.nan])).astype(float).mean()),
        "abstention_rate": float(m.get("market_action_posture", pd.Series(["neutral"])).astype(str).isin({"wait", "neutral"}).mean()),
        "regime_persistence": float(1.0),
        "false_positive_rate": float(m.get("market_false_positive_proxy", pd.Series([np.nan])).astype(float).mean()),
    }

    return pd.DataFrame([asset_row, market_row]).sort_values("level", ascending=True).reset_index(drop=True)
