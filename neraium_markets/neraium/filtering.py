"""Day 6: false-positive pattern flags and filtered signal pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN

# Passive postures that do not count as "active" directional risk.
PASSIVE_POSTURES = frozenset({"wait", "watch"})

# Rule thresholds (deterministic, documented).
HIGH_CONFIDENCE = 0.65
LOW_COHERENCE = 0.35
LOW_BREADTH_PCT = 38.0
HIGH_INSTABILITY = 0.85
EXTREME_INSTABILITY = 0.90
LOW_STABILITY_SCORE = 0.35
STABILITY_WEAK = 0.45


def identify_false_positive_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic flags for noisy or harmful signal conditions.

    Requires prior ``compute_regime_runs`` and ``compute_signal_stability``.

    Flags:
    - ``low_persistence_flag``: run length 1 (single-step regime).
    - ``contradiction_flag``: high confidence with low structural coherence.
    - ``unstable_signal_flag``: low stability_score and/or extreme stress vs coherence.
    - ``likely_false_positive_flag``: composite of the above plus optional churn signals.

    Optional enrichments when columns exist:
    - ``breadth_churn_flag``: breadth_pct_above_20dma below LOW_BREADTH_PCT.
    - ``action_quick_reversal_flag``: posture changed vs prior and matches next row (whipsaw).
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    for col in ("regime_run_length", "confidence_score", "coherence_score", "stability_score"):
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col} (run regime runs + stability first)")

    rl = out["regime_run_length"].astype(float)
    out["low_persistence_flag"] = (rl <= 1.0).astype(int)

    conf = out["confidence_score"].astype(float)
    coh = out["coherence_score"].astype(float)
    out["contradiction_flag"] = ((conf >= HIGH_CONFIDENCE) & (coh < LOW_COHERENCE)).astype(int)

    inst = out["instability_score"].astype(float) if "instability_score" in out.columns else pd.Series(0.5, index=out.index)
    stab = out["stability_score"].astype(float)
    out["unstable_signal_flag"] = (
        (stab < STABILITY_WEAK) | ((inst > HIGH_INSTABILITY) & (coh < LOW_COHERENCE))
    ).astype(int)

    if "breadth_pct_above_20dma" in out.columns:
        b = out["breadth_pct_above_20dma"].astype(float)
        out["breadth_churn_flag"] = (b < LOW_BREADTH_PCT).astype(int)
    else:
        out["breadth_churn_flag"] = 0

    act = out["action_posture"].astype(str)
    prev_a = act.shift(1)
    next_a = act.shift(-1)
    out["action_quick_reversal_flag"] = (
        act.ne(prev_a) & act.eq(next_a) & prev_a.notna() & next_a.notna()
    ).astype(int)

    drift = out["corr_drift_score"].astype(float) if "corr_drift_score" in out.columns else pd.Series(0.0, index=out.index)
    useful_cols = [c for c in out.columns if c.startswith("action_useful_") and c.endswith("d")]
    if useful_cols:
        avg_u = out[useful_cols].mean(axis=1)
        out["high_drift_poor_outcome_flag"] = ((drift > 0.85) & (avg_u < 0)).astype(int)
    else:
        out["high_drift_poor_outcome_flag"] = 0

    likely = (
        ((out["low_persistence_flag"] == 1) & (out["unstable_signal_flag"] == 1))
        | ((out["contradiction_flag"] == 1) & (out["low_persistence_flag"] == 1))
        | ((out["contradiction_flag"] == 1) & (conf >= 0.72) & (coh < 0.30))
        | ((out["low_persistence_flag"] == 1) & (inst > HIGH_INSTABILITY))
        | (out["action_quick_reversal_flag"] == 1)
        | ((out["breadth_churn_flag"] == 1) & (out["contradiction_flag"] == 1))
    )
    out["likely_false_positive_flag"] = likely.astype(int)

    return out.loc[:, ~out.columns.duplicated()]


def apply_signal_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce ``filtered_action_posture`` and ``signal_kept_flag``.

    Suppression sets posture to ``wait`` (abstain) when rules fire; kept rows copy ``action_posture``.

    Rules (deterministic):
    - Suppress if ``likely_false_positive_flag`` == 1.
    - Suppress single-step regimes (``low_persistence_flag``) when ``instability_score`` > 0.75.
    - Suppress ``contradiction_flag`` and ``unstable_signal_flag`` together.
    - Suppress extreme stress: ``instability_score`` > 0.9 and ``stability_score`` < 0.35.
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    for col in (
        "action_posture",
        "likely_false_positive_flag",
        "low_persistence_flag",
        "contradiction_flag",
        "unstable_signal_flag",
        "stability_score",
    ):
        if col not in out.columns:
            raise KeyError(f"Missing required column: {col}")

    inst = out["instability_score"].astype(float) if "instability_score" in out.columns else pd.Series(0.0, index=out.index)
    stab = out["stability_score"].astype(float)

    suppress = (
        (out["likely_false_positive_flag"] == 1)
        | ((out["low_persistence_flag"] == 1) & (inst > 0.75))
        | ((out["contradiction_flag"] == 1) & (out["unstable_signal_flag"] == 1))
        | ((inst > EXTREME_INSTABILITY) & (stab < LOW_STABILITY_SCORE))
    )

    out["filtered_action_posture"] = np.where(suppress, "wait", out["action_posture"].astype(str))
    out["signal_kept_flag"] = (~suppress).astype(int)

    return out.loc[:, ~out.columns.duplicated()]


def compare_filtered_vs_unfiltered(df: pd.DataFrame) -> pd.DataFrame:
    """
    Side-by-side summary: unfiltered ``action_posture`` vs ``filtered_action_posture`` usefulness.

    Expects ``filtered_useful_*`` columns (from ``score_action_usefulness(..., action_col=filtered_action_posture, out_prefix='filtered_useful')``)
    and ``action_useful_*`` for unfiltered.

    Output columns:
    - version, avg_usefulness_1d/5d/10d, active_signal_count, abstention_rate, avg_confidence_kept
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    def _active_mask(action_series: pd.Series) -> pd.Series:
        return ~action_series.astype(str).isin(PASSIVE_POSTURES)

    def _abstention(useful_cols: list[str]) -> float:
        if not useful_cols:
            return float("nan")
        m = out[useful_cols].mean(axis=1)
        return float((m == 0).mean())

    rows: list[dict[str, object]] = []

    ucols = [c for c in ("action_useful_1d", "action_useful_5d", "action_useful_10d") if c in out.columns]
    fcols = [c for c in ("filtered_useful_1d", "filtered_useful_5d", "filtered_useful_10d") if c in out.columns]

    if ucols:
        am = _active_mask(out["action_posture"])
        avg_conf_unfiltered = (
            float(out.loc[am, "confidence_score"].astype(float).mean())
            if "confidence_score" in out.columns and am.any()
            else float("nan")
        )
        rows.append(
            {
                "version": "unfiltered",
                "avg_usefulness_1d": float(out["action_useful_1d"].mean()) if "action_useful_1d" in out.columns else float("nan"),
                "avg_usefulness_5d": float(out["action_useful_5d"].mean()) if "action_useful_5d" in out.columns else float("nan"),
                "avg_usefulness_10d": float(out["action_useful_10d"].mean()) if "action_useful_10d" in out.columns else float("nan"),
                "active_signal_count": int(_active_mask(out["action_posture"]).sum()),
                "abstention_rate": _abstention(ucols),
                "avg_confidence_kept": avg_conf_unfiltered,
            }
        )

    if fcols:
        am_f = _active_mask(out["filtered_action_posture"])
        if "signal_kept_flag" in out.columns and "confidence_score" in out.columns:
            sub_conf = out.loc[(out["signal_kept_flag"] == 1) & am_f]
            avg_conf_f = float(sub_conf["confidence_score"].astype(float).mean()) if len(sub_conf) else float("nan")
        elif "confidence_score" in out.columns:
            avg_conf_f = float(out.loc[am_f, "confidence_score"].astype(float).mean()) if am_f.any() else float("nan")
        else:
            avg_conf_f = float("nan")
        rows.append(
            {
                "version": "filtered",
                "avg_usefulness_1d": float(out["filtered_useful_1d"].mean()) if "filtered_useful_1d" in out.columns else float("nan"),
                "avg_usefulness_5d": float(out["filtered_useful_5d"].mean()) if "filtered_useful_5d" in out.columns else float("nan"),
                "avg_usefulness_10d": float(out["filtered_useful_10d"].mean()) if "filtered_useful_10d" in out.columns else float("nan"),
                "active_signal_count": int(_active_mask(out["filtered_action_posture"]).sum()),
                "abstention_rate": _abstention(fcols),
                "avg_confidence_kept": avg_conf_f,
            }
        )

    return pd.DataFrame(rows)
