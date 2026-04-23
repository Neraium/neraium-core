"""Day 10: human-readable review tables and top failure/success mining."""

from __future__ import annotations

import pandas as pd

from config import DATE_COLUMN

_REVIEW_COLS = [
    DATE_COLUMN,
    "regime_label",
    "scenario_path_label",
    "trajectory_direction",
    "warning_level",
    "action_posture",
    "path_adjusted_market_action",
    "confidence_score",
    "explanation",
    "fwd_ret_5d",
    "action_useful_5d",
    "decision_quality_flag",
]


def _quality_flag(u: float) -> str:
    if pd.isna(u):
        return "neutral"
    if float(u) > 0:
        return "good"
    if float(u) < 0:
        return "bad"
    return "neutral"


def build_decision_review_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact review table with ``decision_quality_flag`` in ``good | neutral | bad``
    from ``action_useful_5d``.
    """
    need = ["action_useful_5d"]
    for c in need:
        if c not in df.columns:
            raise KeyError(c)

    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    rl = out.get("regime_label", out.get("market_regime_label", pd.Series("", index=out.index))).astype(str)
    ap = out.get("action_posture", out.get("market_action_posture", pd.Series("", index=out.index))).astype(str)
    conf = out.get("confidence_score", out.get("market_confidence_score", pd.Series(float("nan"), index=out.index))).astype(
        float
    )
    wl = out.get("warning_level", out.get("market_warning_level", pd.Series("", index=out.index))).astype(str)
    expl = out.get("path_aware_market_explanation", out.get("market_explanation", pd.Series("", index=out.index))).astype(
        str
    )

    fwd5 = out["fwd_ret_5d"].astype(float) if "fwd_ret_5d" in out.columns else pd.Series(float("nan"), index=out.index)

    review = pd.DataFrame(
        {
            DATE_COLUMN: out[DATE_COLUMN],
            "regime_label": rl,
            "scenario_path_label": out.get("scenario_path_label", pd.Series("", index=out.index)).astype(str),
            "trajectory_direction": out.get("trajectory_direction", pd.Series("", index=out.index)).astype(str),
            "warning_level": wl,
            "action_posture": ap,
            "path_adjusted_market_action": out.get("path_adjusted_market_action", pd.Series("", index=out.index)).astype(
                str
            ),
            "confidence_score": conf,
            "explanation": expl,
            "fwd_ret_5d": fwd5,
            "action_useful_5d": out["action_useful_5d"].astype(float),
            "decision_quality_flag": out["action_useful_5d"].map(_quality_flag),
        }
    )
    return review.loc[:, ~review.columns.duplicated()][_REVIEW_COLS]


_COMPACT_TOP = [
    DATE_COLUMN,
    "regime_label",
    "market_regime_label",
    "confidence_score",
    "action_posture",
    "market_action_posture",
    "path_adjusted_market_action",
    "warning_level",
    "decision_type",
    "fwd_ret_5d",
    "action_useful_5d",
]


def _compact_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _COMPACT_TOP if c in df.columns]
    if not cols:
        return df.copy()
    return df.loc[:, cols]


def identify_top_failures(df: pd.DataFrame, n: int = 3, confidence_floor: float = 0.55) -> pd.DataFrame:
    """Rows with high confidence and harmful 5d outcome, sorted by confidence descending."""
    need = ["confidence_score", "action_useful_5d"]
    for c in need:
        if c not in df.columns:
            raise KeyError(c)

    sub = df.copy()
    if "confidence_score" not in sub.columns:
        sub["confidence_score"] = sub.get("market_confidence_score", float("nan"))

    mask = sub["confidence_score"].astype(float) >= confidence_floor
    mask &= sub["action_useful_5d"].astype(float) == -1
    out = sub.loc[mask].sort_values("confidence_score", ascending=False).head(n).reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()]
    return _compact_audit_columns(out)


def identify_top_successes(df: pd.DataFrame, n: int = 3, confidence_floor: float = 0.55) -> pd.DataFrame:
    """Rows with high confidence and clearly useful 5d outcome, sorted by confidence descending."""
    need = ["confidence_score", "action_useful_5d"]
    for c in need:
        if c not in df.columns:
            raise KeyError(c)

    sub = df.copy()
    if "confidence_score" not in sub.columns:
        sub["confidence_score"] = sub.get("market_confidence_score", float("nan"))

    mask = sub["confidence_score"].astype(float) >= confidence_floor
    mask &= sub["action_useful_5d"].astype(float) == 1
    out = sub.loc[mask].sort_values("confidence_score", ascending=False).head(n).reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()]
    return _compact_audit_columns(out)
