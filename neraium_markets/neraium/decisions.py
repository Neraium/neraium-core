"""Day 10: decision records, outcomes, quality metrics, and rolling memory."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from config import DATE_COLUMN
from neraium.evaluation import score_action_usefulness
from neraium.warnings import _map_market_posture_to_action

_POSTURE_TO_DECISION: dict[str, str] = {
    "risk_on": "act",
    "cautious_risk_on": "act",
    "neutral": "wait",
    "reduce_risk": "reduce",
    "defensive": "avoid",
    "wait": "wait",
}


def _decision_type_from_posture(posture: str) -> str:
    return _POSTURE_TO_DECISION.get(str(posture), "wait")


def generate_decision_record(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add audit columns: ``decision_type``, ``regime_label``, ``action_posture``,
    ``confidence_score``, ``warning_level`` (alongside existing ``market_*`` fields).
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    ap = out.get("market_action_posture", out.get("action_posture", pd.Series("", index=out.index))).astype(str)
    out["decision_type"] = ap.map(_decision_type_from_posture)

    if "regime_label" not in out.columns:
        out["regime_label"] = out.get("market_regime_label", pd.Series("", index=out.index)).astype(str)
    if "action_posture" not in out.columns:
        out["action_posture"] = ap
    if "confidence_score" not in out.columns:
        out["confidence_score"] = out.get("market_confidence_score", pd.Series(float("nan"), index=out.index)).astype(
            float
        )
    if "warning_level" not in out.columns:
        out["warning_level"] = out.get("market_warning_level", pd.Series("", index=out.index)).astype(str)

    return out.loc[:, ~out.columns.duplicated()]


def attach_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure ``fwd_ret_{1,5,10}d`` exist and compute ``action_useful_*`` from
    ``path_adjusted_market_action`` (fallback ``market_action_posture``).
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    req_fwd = ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_10d"]
    missing = [c for c in req_fwd if c not in out.columns]
    if missing:
        raise KeyError(f"attach_outcomes requires forward return columns: {missing}")

    drop_u = [c for c in out.columns if c.startswith("action_useful_") and c.endswith("d")]
    out = out.drop(columns=drop_u, errors="ignore")

    act_col = "path_adjusted_market_action"
    if act_col not in out.columns:
        act_col = "market_action_posture" if "market_action_posture" in out.columns else "action_posture"
    out["_decision_action_for_usefulness"] = out[act_col].astype(str).map(_map_market_posture_to_action)
    out = score_action_usefulness(out, action_col="_decision_action_for_usefulness", out_prefix="action_useful")
    out = out.drop(columns=["_decision_action_for_usefulness"], errors="ignore")
    return out.loc[:, ~out.columns.duplicated()]


def evaluate_decision_quality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Summary of decision quality using ``action_useful_*`` and ``decision_type``.
    """
    u1, u5, u10 = "action_useful_1d", "action_useful_5d", "action_useful_10d"
    for c in (u1, u5, u10):
        if c not in df.columns:
            raise KeyError(c)
    if "decision_type" not in df.columns:
        raise KeyError("decision_type")

    sub = df.copy()
    n = len(sub)
    counts = sub["decision_type"].value_counts().to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}

    avg1 = float(sub[u1].mean())
    avg5 = float(sub[u5].mean())
    avg10 = float(sub[u10].mean())

    by_dt: dict[str, dict[str, float]] = {}
    for dt, g in sub.groupby("decision_type", sort=False):
        by_dt[str(dt)] = {
            "avg_usefulness_1d": float(g[u1].mean()),
            "avg_usefulness_5d": float(g[u5].mean()),
            "avg_usefulness_10d": float(g[u10].mean()),
            "count": int(len(g)),
        }

    active = sub["decision_type"].isin(["act", "reduce"])
    fp_denom = int(active.sum())
    false_positive_rate = float((sub.loc[active, u5] == -1).mean()) if fp_denom else float("nan")

    u_mean = (sub[u1] + sub[u5] + sub[u10]) / 3.0
    abstention_rate = float((u_mean == 0).mean())

    return {
        "total_decisions": n,
        "counts_by_decision_type": counts,
        "avg_usefulness_1d": avg1,
        "avg_usefulness_5d": avg5,
        "avg_usefulness_10d": avg10,
        "usefulness_by_decision_type": by_dt,
        "false_positive_rate": false_positive_rate,
        "abstention_rate": abstention_rate,
    }


def decision_quality_summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten ``evaluate_decision_quality`` output to a two-column CSV-friendly table."""
    rows: list[dict[str, Any]] = [
        {"metric": "total_decisions", "value": summary["total_decisions"]},
        {"metric": "avg_usefulness_1d", "value": summary["avg_usefulness_1d"]},
        {"metric": "avg_usefulness_5d", "value": summary["avg_usefulness_5d"]},
        {"metric": "avg_usefulness_10d", "value": summary["avg_usefulness_10d"]},
        {"metric": "false_positive_rate", "value": summary["false_positive_rate"]},
        {"metric": "abstention_rate", "value": summary["abstention_rate"]},
        {"metric": "counts_by_decision_type_json", "value": json.dumps(summary["counts_by_decision_type"], sort_keys=True)},
        {
            "metric": "usefulness_by_decision_type_json",
            "value": json.dumps(summary["usefulness_by_decision_type"], sort_keys=True),
        },
    ]
    return pd.DataFrame(rows)


def build_decision_memory(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling metrics (``window`` rows, min_periods=1) on time-sorted decisions:

    - ``rolling_usefulness``: mean of mean(action_useful_1d,5d,10d) in window
    - ``rolling_success_rate``: mean(action_useful_5d == 1)
    - ``rolling_confidence_accuracy``: among rows with confidence >= 0.55 in the window,
      fraction where action_useful_5d == 1 (NaN if none high-confidence in window)
    """
    req = ["action_useful_1d", "action_useful_5d", "action_useful_10d", "confidence_score"]
    for c in req:
        if c not in df.columns:
            raise KeyError(c)

    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    u1, u5, u10 = out["action_useful_1d"].astype(float), out["action_useful_5d"].astype(float), out["action_useful_10d"].astype(float)
    composite = (u1 + u5 + u10) / 3.0
    out["rolling_usefulness"] = composite.rolling(window, min_periods=1).mean()

    out["rolling_success_rate"] = (u5 == 1).astype(float).rolling(window, min_periods=1).mean()

    conf = out["confidence_score"].astype(float)
    hi = conf >= 0.55
    succ = (u5 == 1).astype(float)

    acc: list[float] = []
    for i in range(len(out)):
        lo = max(0, i - window + 1)
        wdf = out.iloc[lo : i + 1]
        mask = wdf["confidence_score"].astype(float) >= 0.55
        if not mask.any():
            acc.append(float("nan"))
        else:
            acc.append(float((wdf.loc[mask, "action_useful_5d"].astype(float) == 1).mean()))
    out["rolling_confidence_accuracy"] = acc

    return out.loc[:, ~out.columns.duplicated()]
