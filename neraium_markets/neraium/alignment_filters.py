"""Day 7 alignment-aware filtering and aligned-vs-unaligned comparison."""

from __future__ import annotations

import pandas as pd

from config import DATE_COLUMN
from neraium.evaluation import score_action_usefulness

PASSIVE_POSTURES = frozenset({"wait", "watch"})
DEFENSIVE_ACTIONS = frozenset({"avoid_risk", "reduce_exposure", "wait", "watch"})
AGGRESSIVE_ACTIONS = frozenset({"lean_long", "lean_short"})


def apply_timeframe_alignment_filter(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic multi-timeframe gating to produce aligned_action_posture."""
    out = alignment_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    req = [
        "action_daily",
        "action_1h",
        "action_15m",
        "regime_alignment_label",
        "adjusted_confidence_score",
    ]
    for col in req:
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")

    higher_defensive = out["action_daily"].astype(str).isin(DEFENSIVE_ACTIONS) | out["action_1h"].astype(str).isin(DEFENSIVE_ACTIONS)
    lower_aggressive = out["action_15m"].astype(str).isin(AGGRESSIVE_ACTIONS)

    suppress = (
        ((out["action_daily"].astype(str).isin({"avoid_risk", "reduce_exposure"})) & out["action_15m"].astype(str).eq("lean_long"))
        | ((out["regime_alignment_label"].astype(str) == "weak_alignment") & (out["adjusted_confidence_score"].astype(float) < 0.50))
        | (higher_defensive & lower_aggressive)
    )

    preserve = (
        (out["regime_alignment_label"].astype(str) == "strong_alignment")
        & (out.get("action_alignment_label", pd.Series("", index=out.index)).astype(str) == "strong_alignment")
    )

    out["alignment_filter_flag"] = suppress.astype(int)
    out["aligned_action_posture"] = out["action_15m"].astype(str)
    out.loc[suppress & ~preserve, "aligned_action_posture"] = "wait"
    out["alignment_kept_flag"] = (out["aligned_action_posture"] == out["action_15m"].astype(str)).astype(int)

    return out.loc[:, ~out.columns.duplicated()]


def compare_alignment_filtered_vs_unfiltered(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare original 15m actions vs alignment-filtered actions.

    Uses existing forward-return columns and usefulness scoring.
    """
    out = alignment_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    req = ["action_15m", "aligned_action_posture"]
    for col in req:
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")

    out["action_posture"] = out["action_15m"].astype(str)
    scored = score_action_usefulness(out, action_col="action_posture", out_prefix="action_useful")
    scored = score_action_usefulness(scored, action_col="aligned_action_posture", out_prefix="aligned_useful")

    def _summary(version: str, action_col: str, useful_prefix: str, conf_col: str) -> dict[str, object]:
        ucols = [f"{useful_prefix}_1d", f"{useful_prefix}_5d", f"{useful_prefix}_10d"]
        active = ~scored[action_col].astype(str).isin(PASSIVE_POSTURES)
        return {
            "version": version,
            "avg_usefulness_1d": float(scored[ucols[0]].mean()) if ucols[0] in scored.columns else float("nan"),
            "avg_usefulness_5d": float(scored[ucols[1]].mean()) if ucols[1] in scored.columns else float("nan"),
            "avg_usefulness_10d": float(scored[ucols[2]].mean()) if ucols[2] in scored.columns else float("nan"),
            "active_signal_count": int(active.sum()),
            "abstention_rate": float((~active).mean()),
            "average_confidence": float(scored.loc[active, conf_col].astype(float).mean()) if active.any() and conf_col in scored.columns else float("nan"),
        }

    rows = [
        _summary("unaligned", "action_posture", "action_useful", "confidence_15m"),
        _summary("alignment_filtered", "aligned_action_posture", "aligned_useful", "adjusted_confidence_score"),
    ]

    labels = ["regime_alignment_label", "action_alignment_label"]
    for lbl in labels:
        if lbl in scored.columns:
            for version, prefix in (("unaligned", "action_useful"), ("alignment_filtered", "aligned_useful")):
                col = f"{prefix}_5d"
                if col in scored.columns:
                    by = scored.groupby(lbl, observed=False)[col].mean().to_dict()
                    row = next(r for r in rows if r["version"] == version)
                    row[f"usefulness_by_{lbl}"] = {str(k): float(v) for k, v in by.items()}

    return pd.DataFrame(rows)
