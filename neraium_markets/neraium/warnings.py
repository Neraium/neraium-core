"""Day 9: early warnings, path persistence / reversal, path-aware action, usefulness compare."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.evaluation import score_action_usefulness

_WARNING_COLS = (
    "instability_rising_warning",
    "coherence_break_warning",
    "false_calm_warning",
    "transition_acceleration_warning",
    "fragmentation_warning",
    "reversal_warning",
)


def _map_market_posture_to_action(p: str) -> str:
    return {
        "risk_on": "lean_long",
        "cautious_risk_on": "watch",
        "neutral": "watch",
        "reduce_risk": "reduce_exposure",
        "defensive": "avoid_risk",
        "wait": "wait",
    }.get(str(p), "watch")


def compute_early_warning_flags(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic early-warning flags (0/1) before full adverse regime confirmation.

    **Rules**

    - ``instability_rising_warning`` — last **3** ``instability_delta`` values all strictly > 0.
    - ``coherence_break_warning`` — ``coherence_delta < -0.03`` while
      ``market_confidence_score >= 0.50``.
    - ``false_calm_warning`` — ``market_regime_label == market_stable`` and
      ``instability_delta > 0`` and ``coherence_delta < 0`` (surface calm, internals worsening).
    - ``transition_acceleration_warning`` — ``trajectory_speed_score`` jumps by > 0.25 vs prior row
      or ``scenario_path_label`` in ``{false_calm_break, fragile_to_risk_off}``.
    - ``fragmentation_warning`` — ``market_regime_label == market_fragmented`` or
      ``regime_fragmented_frac >= 0.35`` when present.
    - ``reversal_warning`` — ``trajectory_direction == reversing``, or run exhaustion
      (``market_position_in_run`` >= 4 and next row is a regime change), or ``reversal_risk_score >= 0.65``
      after ``refine_early_warnings_with_reversal``.
    """
    req = [
        "market_regime_label",
        "market_confidence_score",
        "market_instability_score",
        "market_coherence_score",
        "instability_delta",
        "coherence_delta",
        "trajectory_speed_score",
        "trajectory_direction",
    ]
    for c in req:
        if c not in market_df.columns:
            raise KeyError(f"Missing required column: {c}")

    out = market_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    inst_d = out["instability_delta"].astype(float)
    coh_d = out["coherence_delta"].astype(float)
    conf = out["market_confidence_score"].astype(float)
    mr = out["market_regime_label"].astype(str)
    spd = out["trajectory_speed_score"].astype(float)
    scen = out.get("scenario_path_label", pd.Series("", index=out.index)).astype(str)
    traj = out["trajectory_direction"].astype(str)

    inst_rising = (inst_d > 0) & (inst_d.shift(1) > 0) & (inst_d.shift(2) > 0)
    out["instability_rising_warning"] = inst_rising.fillna(False).astype(int)

    out["coherence_break_warning"] = ((coh_d < -0.03) & (conf >= 0.50)).astype(int)

    out["false_calm_warning"] = (
        (mr == "market_stable") & (inst_d > 0) & (out["coherence_delta"].astype(float) < 0)
    ).astype(int)

    spd_jump = spd.diff().abs() > 0.25
    scen_adv = scen.isin({"false_calm_break", "fragile_to_risk_off", "stable_to_fragile"})
    out["transition_acceleration_warning"] = (spd_jump | scen_adv).astype(int)

    frag = mr == "market_fragmented"
    if "regime_fragmented_frac" in out.columns:
        frag = frag | (out["regime_fragmented_frac"].astype(float) >= 0.35)
    out["fragmentation_warning"] = frag.astype(int)

    run_exhaust = pd.Series(False, index=out.index)
    if "market_position_in_run" in out.columns and "regime_change_flag" in out.columns:
        run_exhaust = (out["market_position_in_run"].astype(int) >= 4) & (
            out["regime_change_flag"].shift(-1).fillna(0).astype(int) == 1
        )

    out["reversal_warning"] = ((traj == "reversing") | run_exhaust).astype(int)

    w = 0.18 * out["instability_rising_warning"]
    w = w + 0.18 * out["coherence_break_warning"]
    w = w + 0.16 * out["false_calm_warning"]
    w = w + 0.14 * out["transition_acceleration_warning"]
    w = w + 0.14 * out["fragmentation_warning"]
    w = w + 0.20 * out["reversal_warning"]
    out["warning_score"] = w.astype(float).clip(0.0, 1.0)

    return out.loc[:, ~out.columns.duplicated()]


def generate_market_warning_level(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ``warning_score`` to ``market_warning_level`` in
    ``none | low | moderate | elevated | high``.
    """
    if "warning_score" not in market_df.columns:
        raise KeyError("warning_score")

    out = market_df.copy()
    s = out["warning_score"].astype(float).fillna(0.0)
    lvl = np.full(len(out), "none", dtype=object)
    lvl[s >= 0.15] = "low"
    lvl[s >= 0.30] = "moderate"
    lvl[s >= 0.50] = "elevated"
    lvl[s >= 0.70] = "high"
    out["market_warning_level"] = lvl.astype(str)
    return out


def compute_path_persistence_score(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    ``path_persistence_score`` in [0,1]: higher when the current path likely continues.

    Uses ``trajectory_run_length``, ``scenario_path_length``, ``market_confidence_score``,
    ``market_coherence_score``, ``market_instability_score`` (lower instability helps).
    """
    need = [
        "market_confidence_score",
        "market_coherence_score",
        "market_instability_score",
    ]
    for c in need:
        if c not in market_df.columns:
            raise KeyError(c)

    out = market_df.copy()
    trl = out.get("trajectory_run_length", pd.Series(1, index=out.index)).astype(float)
    spl = out.get("scenario_path_length", pd.Series(1, index=out.index)).astype(float)
    mrl = out.get("market_run_length", pd.Series(1, index=out.index)).astype(float)

    conf = out["market_confidence_score"].astype(float).fillna(0.5)
    coh = out["market_coherence_score"].astype(float).fillna(0.5)
    inst = out["market_instability_score"].astype(float).fillna(0.5)

    run_component = (np.log1p(trl) + np.log1p(mrl) + 0.5 * np.log1p(spl)) / (np.log1p(12) + np.log1p(12) + 0.5 * np.log1p(4))
    run_component = np.clip(run_component, 0.0, 1.0)

    score = (
        0.30 * conf
        + 0.25 * coh
        + 0.20 * (1.0 - inst)
        + 0.25 * run_component
    )
    out["path_persistence_score"] = np.clip(score, 0.0, 1.0)
    return out


def compute_reversal_risk_score(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    ``reversal_risk_score`` in [0,1]: higher when a regime/path reversal is more likely.

    Combines weakening coherence trend, rising instability, falling confidence,
    trajectory conflict (``trajectory_direction == reversing``), and run exhaustion
    (long regime run with deteriorating trajectory). Does **not** depend on warning flags
    so it can run before ``compute_early_warning_flags``.
    """
    out = market_df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    coh = out["market_coherence_score"].astype(float)
    inst = out["market_instability_score"].astype(float)
    conf = out["market_confidence_score"].astype(float)

    coh_roll = coh.rolling(3, min_periods=1).mean()
    coh_weak = (coh_roll.diff() < -0.02).astype(float)
    inst_rise = (inst.diff() > 0.02).astype(float)
    conf_fall = (conf.diff() < -0.02).astype(float)

    traj_rev = (
        out["trajectory_direction"].astype(str) == "reversing"
        if "trajectory_direction" in out.columns
        else pd.Series(False, index=out.index)
    ).astype(float)

    exhaustion = pd.Series(0.0, index=out.index)
    if {"market_run_length", "trajectory_direction"}.issubset(out.columns):
        exhaustion = (
            (out["market_run_length"].astype(float) >= 5.0)
            & (out["trajectory_direction"].astype(str) == "deteriorating")
        ).astype(float)

    rev = (
        0.26 * coh_weak
        + 0.22 * inst_rise
        + 0.18 * conf_fall
        + 0.24 * traj_rev
        + 0.10 * exhaustion
    )
    out["reversal_risk_score"] = np.clip(rev, 0.0, 1.0)
    return out


def adjust_market_action_for_path(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce ``path_adjusted_market_action`` from ``market_action_posture`` using warnings and path context.

    **Rules (applied in order)**

    1. ``scenario_path_label == fragile_to_risk_off`` → map ``risk_on`` / ``cautious_risk_on`` to ``reduce_risk``.
    2. ``false_calm_warning`` and posture ``risk_on`` → ``cautious_risk_on``.
    3. ``reversal_risk_score >= 0.65`` → step down one level (``risk_on``→``cautious_risk_on``,
       ``cautious_risk_on``→``neutral``, ``neutral``→``reduce_risk``; defensive/wait unchanged).
    4. ``market_warning_level == high`` → cap ``risk_on`` at ``wait``, ``cautious_risk_on`` at ``neutral``.
    5. ``market_warning_level == elevated`` → cap ``risk_on`` at ``cautious_risk_on``.
    6. If ``path_persistence_score >= 0.62``, trajectory in ``{improving, stabilizing}``, and
       warning level in ``{none, low}`` → keep original posture (undo steps 3–5 only for that row
       by restoring base when strictly beneficial — here we **skip** de-escalation when 6 holds by
       applying step 6 last as: if 6 applies, set to ``market_action_posture``).
    """
    rank = {
        "risk_on": 5,
        "cautious_risk_on": 4,
        "neutral": 3,
        "reduce_risk": 2,
        "defensive": 1,
        "wait": 0,
    }

    def step_down(p: str) -> str:
        r = rank.get(p, 3)
        if r <= 1:
            return p
        return {5: "cautious_risk_on", 4: "neutral", 3: "reduce_risk", 2: "reduce_risk"}.get(r, p)

    need = ["market_action_posture", "market_warning_level"]
    for c in need:
        if c not in market_df.columns:
            raise KeyError(c)

    out = market_df.copy()
    base = out["market_action_posture"].astype(str)
    wl = out["market_warning_level"].astype(str)
    fcalm = out.get("false_calm_warning", pd.Series(0, index=out.index)).astype(int)
    scen = out.get("scenario_path_label", pd.Series("", index=out.index)).astype(str)
    rr = out.get("reversal_risk_score", pd.Series(0.0, index=out.index)).astype(float)
    pp = out.get("path_persistence_score", pd.Series(0.5, index=out.index)).astype(float)
    traj = out.get("trajectory_direction", pd.Series("flat", index=out.index)).astype(str)

    adjusted: list[str] = []
    for i in range(len(out)):
        p = str(base.iloc[i])
        if scen.iloc[i] == "fragile_to_risk_off" and p in ("risk_on", "cautious_risk_on"):
            p = "reduce_risk"
        if int(fcalm.iloc[i]) == 1 and str(base.iloc[i]) == "risk_on":
            p = "cautious_risk_on"
        if float(rr.iloc[i]) >= 0.65:
            p = step_down(p)
        if wl.iloc[i] == "high":
            if p == "risk_on":
                p = "wait"
            elif p == "cautious_risk_on":
                p = "neutral"
        elif wl.iloc[i] == "elevated" and p == "risk_on":
            p = "cautious_risk_on"
        if (
            float(pp.iloc[i]) >= 0.62
            and str(traj.iloc[i]) in {"improving", "stabilizing"}
            and wl.iloc[i] in {"none", "low"}
        ):
            p = str(base.iloc[i])
        adjusted.append(p)

    out["path_adjusted_market_action"] = pd.Series(adjusted, index=out.index, dtype=str)
    return out


def generate_path_aware_market_explanation(market_df: pd.DataFrame) -> pd.DataFrame:
    """Short deterministic narrative including scenario, trajectory, and warning level."""
    out = market_df.copy()
    lines: list[str] = []
    for _, row in out.iterrows():
        scen = str(row.get("scenario_path_label", "unclassified"))
        traj = str(row.get("trajectory_direction", ""))
        wl = str(row.get("market_warning_level", "none"))
        rr = float(row.get("reversal_risk_score", float("nan")))
        before = str(row.get("market_action_posture", ""))
        after = str(row.get("path_adjusted_market_action", before))
        parts = [
            f"scenario={scen}",
            f"trajectory={traj}",
            f"warnings={wl}",
            f"reversal_risk={rr:.2f}",
            f"action {before}->{after}",
        ]
        lines.append(
            "Market-wide structure path: " + "; ".join(parts) + "."
        )
    out["path_aware_market_explanation"] = lines
    return out


def path_aware_usefulness_breakdowns(market_df: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Nested dicts: usefulness by ``scenario_path_label`` and by ``market_warning_level`` (path-adjusted action)."""
    df = market_df.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    tmp = df.assign(
        action_posture=df["path_adjusted_market_action"].map(_map_market_posture_to_action)
    )
    tmp = score_action_usefulness(tmp)
    ucols = [c for c in tmp.columns if c.startswith("action_useful_") and c.endswith("d")]
    if not ucols:
        raise ValueError("usefulness columns missing")
    by_scen: dict[str, dict[str, float]] = {}
    if "scenario_path_label" in tmp.columns:
        for s, g in tmp.groupby("scenario_path_label", sort=False):
            gu = g[ucols].mean(axis=1)
            by_scen[str(s)] = {
                "avg_usefulness": float(gu.mean()),
                "harmful_rate": float((gu < 0).mean()),
            }
    by_wl: dict[str, dict[str, float]] = {}
    if "market_warning_level" in tmp.columns:
        for s, g in tmp.groupby("market_warning_level", sort=False):
            gu = g[ucols].mean(axis=1)
            by_wl[str(s)] = {
                "avg_usefulness": float(gu.mean()),
                "harmful_rate": float((gu < 0).mean()),
            }
    return by_scen, by_wl


def compare_path_aware_vs_static_market_usefulness(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-row summary comparing static ``market_action_posture`` vs ``path_adjusted_market_action``.

    Scalar columns only (CSV-friendly). Use ``path_aware_usefulness_breakdowns`` or Day 9 JSON
    report for scenario / warning-level tables.
    """
    req = ["market_action_posture", "path_adjusted_market_action"]
    for c in req:
        if c not in market_df.columns:
            raise KeyError(c)

    df = market_df.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    def _metrics(version: str, action_col: str) -> dict[str, float]:
        tmp = df.assign(action_posture=df[action_col].map(_map_market_posture_to_action))
        tmp = score_action_usefulness(tmp)
        ucols = [c for c in tmp.columns if c.startswith("action_useful_") and c.endswith("d")]
        if not ucols:
            raise ValueError("usefulness columns missing after scoring")
        u = tmp[ucols].mean(axis=1)
        harmful = float((u < 0).mean())
        abstain = float((u == 0).mean())
        avg_u = {h: float(tmp[c].mean()) for h, c in zip(["1d", "5d", "10d"], ucols)}

        warn_hit = float("nan")
        if "warning_score" in tmp.columns:
            warn_hit = float((tmp["warning_score"].astype(float) >= 0.30).mean())

        return {
            "version": version,
            "avg_usefulness_1d": avg_u["1d"],
            "avg_usefulness_5d": avg_u["5d"],
            "avg_usefulness_10d": avg_u["10d"],
            "harmful_rate": harmful,
            "abstention_rate": abstain,
            "warning_hit_rate": warn_hit,
        }

    rows = [
        _metrics("static_market_action", "market_action_posture"),
        _metrics("path_adjusted_market_action", "path_adjusted_market_action"),
    ]
    return pd.DataFrame(rows)


def refine_early_warnings_with_reversal(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional second pass: set ``reversal_warning`` using ``reversal_risk_score`` when available.
    Recomputes ``warning_score`` and should run after ``compute_reversal_risk_score``.
    """
    if "reversal_risk_score" not in market_df.columns:
        return market_df
    out = market_df.copy()
    rr = out["reversal_risk_score"].astype(float) >= 0.65
    if "reversal_warning" in out.columns:
        out["reversal_warning"] = (out["reversal_warning"].astype(bool) | rr).astype(int)
    w = (
        0.18 * out["instability_rising_warning"]
        + 0.18 * out["coherence_break_warning"]
        + 0.16 * out["false_calm_warning"]
        + 0.14 * out["transition_acceleration_warning"]
        + 0.14 * out["fragmentation_warning"]
        + 0.20 * out["reversal_warning"]
    )
    out["warning_score"] = w.astype(float).clip(0.0, 1.0)
    return out
