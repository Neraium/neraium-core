"""Day 7 multi-timeframe alignment and agreement scoring utilities."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from config import DATE_COLUMN

# Deterministic compatibility map used by regime agreement scoring.
# Relationship is symmetric; pairs absent from the map are treated as incompatible.
REGIME_COMPATIBILITY: dict[frozenset[str], float] = {
    frozenset(("fragile_rally", "stable_trend")): 0.65,
    frozenset(("risk_off_transition", "high_volatility")): 0.75,
    frozenset(("false_calm", "unstable")): 0.60,
    frozenset(("mean_reversion", "stable_trend")): 0.55,
    frozenset(("mean_reversion", "high_volatility")): 0.50,
}

DEFENSIVE_ACTIONS = frozenset({"avoid_risk", "reduce_exposure", "wait", "watch"})
AGGRESSIVE_ACTIONS = frozenset({"lean_long", "lean_short"})


def _prep(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    req = {DATE_COLUMN, "regime_label", "action_posture", "confidence_score"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for {suffix}: {sorted(missing)}")

    keep = [
        DATE_COLUMN,
        "regime_label",
        "action_posture",
        "confidence_score",
        "filtered_action_posture",
        "action_useful_1d",
        "action_useful_5d",
        "action_useful_10d",
        "fwd_ret_1d",
        "fwd_ret_5d",
        "fwd_ret_10d",
        "spy_vol_10d",
    ]
    sub = df[[c for c in keep if c in df.columns]].copy()
    rename = {
        "regime_label": f"regime_{suffix}",
        "action_posture": f"action_{suffix}",
        "confidence_score": f"confidence_{suffix}",
        "filtered_action_posture": f"filtered_action_{suffix}",
    }
    sub = sub.rename(columns=rename)
    sub[DATE_COLUMN] = pd.to_datetime(sub[DATE_COLUMN], utc=False)
    return sub.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)


def build_timeframe_alignment_table(
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align daily/1h/15m signals on a common 15m timestamp grid.

    Deterministic alignment rule:
    - 15m rows define the output timeline.
    - For each 15m timestamp, attach the *most recent* daily and 1h row
      where higher-timeframe timestamp <= 15m timestamp (``merge_asof`` backward).
    """
    d = _prep(daily_df, "daily")
    h = _prep(hourly_df, "1h")
    i = _prep(intraday_df, "15m")

    out = i.copy()
    out = pd.merge_asof(out, h, on=DATE_COLUMN, direction="backward")
    out = pd.merge_asof(out, d, on=DATE_COLUMN, direction="backward")

    out["timeframe"] = "15m"
    out = out.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    return out.loc[:, ~out.columns.duplicated()]


def _pair_compatibility(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return REGIME_COMPATIBILITY.get(frozenset((str(a), str(b))), 0.0)


def _regime_score(regimes: Iterable[str]) -> tuple[float, str]:
    vals = [str(r) for r in regimes]
    if len(vals) != 3:
        return 0.0, "weak_alignment"

    pair_scores = [
        _pair_compatibility(vals[0], vals[1]),
        _pair_compatibility(vals[0], vals[2]),
        _pair_compatibility(vals[1], vals[2]),
    ]
    score = float(np.mean(pair_scores))

    counts = Counter(vals)
    max_count = max(counts.values())
    if min(pair_scores) >= 0.60:
        return max(score, 0.85), "strong_alignment"
    if max_count >= 2 or sum(ps > 0 for ps in pair_scores) >= 2:
        return max(score, 0.55), "medium_alignment"
    return min(score, 0.35), "weak_alignment"


def compute_regime_agreement(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """Add regime_agreement_score and regime_alignment_label."""
    out = alignment_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    req = ["regime_daily", "regime_1h", "regime_15m"]
    for col in req:
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")

    vals = out[req].fillna("unknown").astype(str).to_numpy()
    scores: list[float] = []
    labels: list[str] = []
    for r in vals:
        s, lbl = _regime_score(r.tolist())
        scores.append(float(np.clip(s, 0.0, 1.0)))
        labels.append(lbl)

    out["regime_agreement_score"] = scores
    out["regime_alignment_label"] = labels
    return out.loc[:, ~out.columns.duplicated()]


def _action_score(daily: str, hourly: str, intraday: str) -> tuple[float, str]:
    trio = [str(daily), str(hourly), str(intraday)]
    if trio[0] == trio[1] == trio[2]:
        return 1.0, "strong_alignment"

    higher = {trio[0], trio[1]}
    lower = trio[2]

    if higher.issubset(DEFENSIVE_ACTIONS) and lower in AGGRESSIVE_ACTIONS:
        return 0.20, "weak_alignment"

    if {"reduce_exposure", "avoid_risk"}.issubset(set(trio)):
        return 0.80, "strong_alignment"

    counts = Counter(trio)
    if max(counts.values()) >= 2:
        return 0.60, "medium_alignment"

    if all(a in DEFENSIVE_ACTIONS for a in trio):
        return 0.70, "medium_alignment"

    return 0.30, "weak_alignment"


def compute_action_agreement(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """Add action_agreement_score and action_alignment_label."""
    out = alignment_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    req = ["action_daily", "action_1h", "action_15m"]
    for col in req:
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")

    scores: list[float] = []
    labels: list[str] = []
    for d, h, i in out[req].fillna("wait").itertuples(index=False, name=None):
        s, lbl = _action_score(str(d), str(h), str(i))
        scores.append(float(np.clip(s, 0.0, 1.0)))
        labels.append(lbl)

    out["action_agreement_score"] = scores
    out["action_alignment_label"] = labels
    return out.loc[:, ~out.columns.duplicated()]


def apply_timeframe_confidence_adjustment(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add adjusted_confidence_score using 15m as base confidence.

    Formula:
    adjusted = base_15m + 0.12*(regime_agreement_score-0.5)
                         + 0.12*(action_agreement_score-0.5)
                         + 0.06*sign(conf_daily-base_15m)
                         - 0.08*(higher_defensive_and_15m_aggressive)
    Then clamp to [0, 1].
    """
    out = alignment_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    req = [
        "confidence_15m",
        "confidence_1h",
        "confidence_daily",
        "regime_agreement_score",
        "action_agreement_score",
        "action_daily",
        "action_1h",
        "action_15m",
    ]
    for col in req:
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")

    base = out["confidence_15m"].astype(float).fillna(0.0)
    reg_adj = 0.12 * (out["regime_agreement_score"].astype(float).fillna(0.0) - 0.5)
    act_adj = 0.12 * (out["action_agreement_score"].astype(float).fillna(0.0) - 0.5)

    daily_conf = out["confidence_daily"].astype(float).fillna(base)
    slope_adj = 0.06 * np.sign(daily_conf - base)

    higher_defensive = (
        out["action_daily"].astype(str).isin(DEFENSIVE_ACTIONS)
        | out["action_1h"].astype(str).isin(DEFENSIVE_ACTIONS)
    )
    lower_aggressive = out["action_15m"].astype(str).isin(AGGRESSIVE_ACTIONS)
    conflict_penalty = 0.08 * (higher_defensive & lower_aggressive).astype(float)

    out["adjusted_confidence_score"] = (base + reg_adj + act_adj + slope_adj - conflict_penalty).clip(0.0, 1.0)
    return out.loc[:, ~out.columns.duplicated()]
