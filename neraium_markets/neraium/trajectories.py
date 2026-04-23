"""Day 9: market-wide state trajectory tracking and run segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN

_REQ = (
    "market_regime_label",
    "market_confidence_score",
    "market_instability_score",
    "market_coherence_score",
)


def _assign_runs(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Consecutive-run IDs and lengths for a categorical series (sorted timeline).

    Returns (run_id, run_length, position_in_run) as int Series aligned to ``series``.
    """
    s = series.astype(str)
    change = s.ne(s.shift(1)).astype(int)
    run_id = change.cumsum()
    grp = run_id.groupby(run_id)
    run_length = grp.transform("size").astype(int)
    position_in_run = grp.cumcount() + 1
    return run_id.astype(int), run_length.astype(int), position_in_run.astype(int)


def compute_market_state_trajectory(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Track how market-wide state evolves row-by-row (time-ordered).

    **Rules (deterministic)**

    - ``prior_market_regime_label``: previous row's ``market_regime_label``.
    - ``regime_change_flag``: 1 when current label differs from prior (0 on first row).
    - Deltas are first differences vs prior row (NaN on first row):
      ``instability_delta``, ``coherence_delta``, ``confidence_delta``.

    - **Net stress signal** (higher = structurally better):
      ``favor = -instability_delta + coherence_delta + confidence_delta``

    - ``trajectory_direction``:
      - ``flat`` — ``|favor| < 0.015`` (no meaningful move).
      - ``improving`` — ``favor >= 0.025`` (stress easing on net).
      - ``deteriorating`` — ``favor <= -0.025`` (stress building on net).
      - ``stabilizing`` — small ``|favor|`` but mixed signed deltas (not clearly improving
        or deteriorating): ``0.015 <= |favor| < 0.025`` and neither instability nor coherence
        dominates alone.
      - ``reversing`` — sign of ``favor`` opposes the sign of the prior row's ``favor``,
        and both current and prior ``|favor| >= 0.02`` (direction flip in net movement).

    - ``trajectory_speed_score``: rolling mean of Euclidean magnitude of the last three
      (instability_delta, coherence_delta, confidence_delta) vectors, clamped to [0, 1]
      after ``min(1.0, raw / 0.15)``. First two rows use available history only.
    """
    for c in _REQ:
        if c not in market_df.columns:
            raise KeyError(f"Missing required column: {c}")

    out = market_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    mr = out["market_regime_label"].astype(str)
    inst = out["market_instability_score"].astype(float)
    coh = out["market_coherence_score"].astype(float)
    conf = out["market_confidence_score"].astype(float)

    out["prior_market_regime_label"] = mr.shift(1)
    out["regime_change_flag"] = mr.ne(out["prior_market_regime_label"]).fillna(False).astype(int)

    out["instability_delta"] = inst.diff()
    out["coherence_delta"] = coh.diff()
    out["confidence_delta"] = conf.diff()

    inst_d = out["instability_delta"].astype(float)
    coh_d = out["coherence_delta"].astype(float)
    conf_d = out["confidence_delta"].astype(float)
    favor = -inst_d + coh_d + conf_d
    prior_favor = favor.shift(1)

    abs_f = favor.abs()
    rev = (
        prior_favor.notna()
        & favor.notna()
        & (np.sign(prior_favor.to_numpy()) * np.sign(favor.to_numpy()) < 0)
        & (prior_favor.abs() >= 0.02)
        & (favor.abs() >= 0.02)
    )

    direction = pd.Series(index=out.index, dtype=object)
    for i in range(len(out)):
        if bool(rev.iloc[i]):
            direction.iloc[i] = "reversing"
            continue
        f = favor.iloc[i]
        af = abs_f.iloc[i]
        if pd.isna(f) or pd.isna(af):
            direction.iloc[i] = "flat"
            continue
        if af < 0.015:
            direction.iloc[i] = "flat"
        elif f >= 0.025:
            direction.iloc[i] = "improving"
        elif f <= -0.025:
            direction.iloc[i] = "deteriorating"
        elif 0.015 <= af < 0.025:
            direction.iloc[i] = "stabilizing"
        else:
            direction.iloc[i] = "flat"

    out["trajectory_direction"] = direction.astype(str)

    mag = np.sqrt(inst_d**2 + coh_d**2 + conf_d**2)
    speed = mag.rolling(window=3, min_periods=1).mean()
    out["trajectory_speed_score"] = np.clip(speed / 0.15, 0.0, 1.0).fillna(0.0)

    return out.loc[:, ~out.columns.duplicated()]


def compute_market_state_runs(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment consecutive runs of ``market_regime_label`` and ``trajectory_direction``.

    Requires ``trajectory_direction`` (run ``compute_market_state_trajectory`` first).

    Adds:
    - ``market_run_id``, ``market_run_length``, ``market_position_in_run`` for regime label runs.
    - ``trajectory_run_id``, ``trajectory_run_length``, ``trajectory_position_in_run`` for
      trajectory direction runs.
    """
    need = list(_REQ) + ["trajectory_direction"]
    for c in need:
        if c not in market_df.columns:
            raise KeyError(f"Missing required column: {c}")

    out = market_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    mr_id, mr_len, mr_pos = _assign_runs(out["market_regime_label"])
    out["market_run_id"] = mr_id
    out["market_run_length"] = mr_len
    out["market_position_in_run"] = mr_pos

    tr_id, tr_len, tr_pos = _assign_runs(out["trajectory_direction"])
    out["trajectory_run_id"] = tr_id
    out["trajectory_run_length"] = tr_len
    out["trajectory_position_in_run"] = tr_pos

    return out.loc[:, ~out.columns.duplicated()]
