"""Day 6: regime run persistence and transition analysis (read-only diagnostics)."""

from __future__ import annotations

import pandas as pd

from config import DATE_COLUMN


def compute_regime_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label consecutive runs of the same ``regime_label`` (sorted by ``timestamp``).

    Adds:
    - ``regime_run_id``: integer id per contiguous run (changes when regime changes)
    - ``regime_run_length``: total length of the run (same for all rows in the run)
    - ``regime_position_in_run``: 1-based index within the run
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    if "regime_label" not in out.columns:
        raise KeyError("Missing required column: regime_label")

    regime = out["regime_label"].astype(str)
    new_run = regime.ne(regime.shift(1))
    out["regime_run_id"] = new_run.cumsum().astype(int)
    out["regime_run_length"] = out.groupby("regime_run_id", sort=False)["regime_label"].transform("size")
    out["regime_position_in_run"] = out.groupby("regime_run_id", sort=False).cumcount() + 1

    return out.loc[:, ~out.columns.duplicated()]


def summarize_regime_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per ``regime_label`` with run-level statistics.

    Uses rows where ``regime_position_in_run == 1`` as one sample per run.
    Requires ``compute_regime_runs`` output.
    """
    if "regime_run_length" not in df.columns or "regime_position_in_run" not in df.columns:
        raise KeyError("Run compute_regime_runs before summarize_regime_persistence")

    starts = df[df["regime_position_in_run"] == 1].copy()

    summary = (
        starts.groupby("regime_label", observed=False)["regime_run_length"]
        .agg(
            count_runs="size",
            avg_run_length="mean",
            median_run_length="median",
            max_run_length="max",
            pct_single_step_runs=lambda s: float((s == 1).mean()) if len(s) else 0.0,
        )
        .reset_index()
    )
    return summary.sort_values("regime_label").reset_index(drop=True)


def build_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count regime_label[t] -> regime_label[t+1] transitions and conditional probabilities.

    Returns long-form table:
    - from_regime, to_regime, transition_count, p_to_given_from
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    if "regime_label" not in out.columns:
        raise KeyError("Missing required column: regime_label")

    from_reg = out["regime_label"].iloc[:-1].astype(str).reset_index(drop=True)
    to_reg = out["regime_label"].iloc[1:].astype(str).reset_index(drop=True)
    pairs = pd.DataFrame({"from_regime": from_reg, "to_regime": to_reg})

    counts = (
        pairs.groupby(["from_regime", "to_regime"], observed=False)
        .size()
        .reset_index(name="transition_count")
    )
    counts["p_to_given_from"] = counts.groupby("from_regime", observed=False)["transition_count"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0.0
    )
    return counts.sort_values(["from_regime", "transition_count"], ascending=[True, False]).reset_index(
        drop=True
    )


def summarize_transition_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average action usefulness by (from_regime, to_regime) transition.

    Associates row *t* (signal at t) with transition (regime[t], regime[t+1]).
    Last row is dropped (no successor regime).

    Requires ``action_useful_1d`` (and optionally 5d/10d) from Day 5 scoring.
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    if "regime_label" not in out.columns:
        raise KeyError("Missing required column: regime_label")

    useful_cols = [c for c in ("action_useful_1d", "action_useful_5d", "action_useful_10d") if c in out.columns]
    if not useful_cols:
        raise KeyError("Need action_useful_* columns; run score_action_usefulness first")

    sub = out.iloc[:-1].copy()
    sub["to_regime"] = out["regime_label"].iloc[1:].astype(str).values
    sub.rename(columns={"regime_label": "from_regime"}, inplace=True)

    rename_map = {
        "action_useful_1d": "avg_usefulness_1d",
        "action_useful_5d": "avg_usefulness_5d",
        "action_useful_10d": "avg_usefulness_10d",
    }
    g = sub.groupby(["from_regime", "to_regime"], observed=False)
    counts = g.size().reset_index(name="transition_count")
    means = g[useful_cols].mean().reset_index()
    merged = counts.merge(means, on=["from_regime", "to_regime"])
    for old, new in rename_map.items():
        if old in merged.columns:
            merged.rename(columns={old: new}, inplace=True)
    return merged.sort_values("transition_count", ascending=False).reset_index(drop=True)
