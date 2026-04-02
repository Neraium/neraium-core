"""Fleet-level intelligence, rankings, and summaries from replay or batch tabular outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

# Deterministic tie-break orders (first wins when counts tie).
_STATE_ORDER = ("ALERT", "WATCH", "STABLE", "BASELINE_FORMING", "UNKNOWN")
_TRANSITION_STATE_ORDER = (
    "SUSTAINED_TRANSITION",
    "EMERGING_TRANSITION",
    "METASTABLE",
    "NONE",
    "UNKNOWN",
)
_HORIZON_ORDER = ("IMMINENT", "NEAR_TERM", "MID_TERM", "LONGER_HORIZON", "UNKNOWN")
_TRAJECTORY_ORDER = ("DIVERGING", "METASTABLE", "STABILIZING", "UNAVAILABLE", "UNKNOWN")

# fleet_priority_score: fixed weights (sum 1.0 before trust/safe adjustments).
_WEIGHT_SEVERITY = 0.22
_WEIGHT_PERSISTENCE = 0.14
_WEIGHT_TRAJECTORY = 0.14
_WEIGHT_HORIZON = 0.14
_WEIGHT_URGENCY = 0.12
_WEIGHT_LOCK_IN = 0.10
_WEIGHT_PNR = 0.08
_WEIGHT_TRUST_PENALTY = 0.06
_WEIGHT_SAFE_TO_ACT = 0.05

FLEET_PRIORITY_SCORE_EXPLANATION = (
    "fleet_priority_score is a 0 to 100 index. It blends severity peak, WATCH or ALERT persistence, "
    "dominant trajectory divergence, dominant horizon risk, peak recommendation urgency, "
    "maximum lock-in and point-of-no-return risk, mean trust shortfall, and reduced weight when "
    "safe_to_act is often true. Higher means higher operator attention. Ties break on asset id then unit."
)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, float) and pd.isna(v):
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _safe_bool(v: Any) -> bool | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _state_severity_numeric(state: str) -> float:
    st = (state or "STABLE").upper()
    if st == "ALERT":
        return 1.0
    if st == "WATCH":
        return 0.62
    if st == "STABLE":
        return 0.12
    return 0.2


def _urgency_numeric(urgency: str) -> float:
    u = (urgency or "low").lower()
    if u == "critical":
        return 1.0
    if u == "high":
        return 0.78
    if u == "medium":
        return 0.52
    return 0.28


def _horizon_numeric(horizon: str) -> float:
    h = (horizon or "LONGER_HORIZON").upper()
    if h == "IMMINENT":
        return 1.0
    if h == "NEAR_TERM":
        return 0.78
    if h == "MID_TERM":
        return 0.52
    if h == "LONGER_HORIZON":
        return 0.22
    return 0.35


def _trajectory_divergence_numeric(trajectory: str) -> float:
    t = (trajectory or "METASTABLE").upper()
    if t == "DIVERGING":
        return 1.0
    if t == "METASTABLE":
        return 0.48
    if t == "STABILIZING":
        return 0.12
    if t == "UNAVAILABLE":
        return 0.25
    return 0.4


def _dominant_preference(series: pd.Series, preference: tuple[str, ...], default: str = "UNKNOWN") -> str:
    s = series.dropna()
    if s.empty:
        return default
    s = s.astype(str)
    counts = s.value_counts()
    m = int(counts.max())
    candidates = [str(k) for k, v in counts.items() if int(v) == m]
    for pref in preference:
        if pref in candidates:
            return pref
    return candidates[0] if candidates else default


def resolve_asset_column(df: pd.DataFrame, asset_col: str = "unit") -> str:
    """Prefer ``asset_id`` when present and non-empty if caller left default ``unit``."""
    if asset_col != "unit":
        if asset_col not in df.columns:
            raise ValueError(f"Column {asset_col!r} not found in dataframe")
        return asset_col
    if "asset_id" in df.columns and df["asset_id"].notna().any():
        return "asset_id"
    if "unit" not in df.columns:
        raise ValueError("Dataframe must contain 'unit' or 'asset_id'")
    return "unit"


def compute_fleet_priority_score_components(
    *,
    severity_peak: float,
    persistence_score: float,
    trajectory_div: float,
    horizon_risk: float,
    urgency_peak: float,
    lock_in_max: float,
    pnr_max: float,
    mean_trust: float,
    safe_to_act_rate: float,
) -> float:
    """
    Single scalar 0 to 100. All inputs are expected in 0 to 1 except mean_trust and safe_to_act_rate
    (also 0 to 1).
    """
    trust_penalty = max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, mean_trust))))
    safe = max(0.0, min(1.0, safe_to_act_rate))
    inner = (
        _WEIGHT_SEVERITY * max(0.0, min(1.0, severity_peak))
        + _WEIGHT_PERSISTENCE * max(0.0, min(1.0, persistence_score))
        + _WEIGHT_TRAJECTORY * max(0.0, min(1.0, trajectory_div))
        + _WEIGHT_HORIZON * max(0.0, min(1.0, horizon_risk))
        + _WEIGHT_URGENCY * max(0.0, min(1.0, urgency_peak))
        + _WEIGHT_LOCK_IN * max(0.0, min(1.0, lock_in_max))
        + _WEIGHT_PNR * max(0.0, min(1.0, pnr_max))
        + _WEIGHT_TRUST_PENALTY * trust_penalty
        - _WEIGHT_SAFE_TO_ACT * safe
    )
    return round(max(0.0, min(100.0, 100.0 * inner)), 4)


def compute_fleet_intelligence_table(df: pd.DataFrame, *, asset_col: str = "unit") -> pd.DataFrame:
    """
    One row per asset with fleet metrics and fleet_priority_score (higher is higher priority).

    Works from slim replay rows when columns are present; missing optional columns use safe defaults.
    """
    key = resolve_asset_column(df, asset_col)
    if key not in df.columns:
        raise ValueError(f"Column {key!r} not found in dataframe")

    work = df.copy()
    if "cycle" not in work.columns:
        work["cycle"] = work.groupby(key, sort=False).cumcount()

    rows: list[dict[str, Any]] = []

    for asset_val, g in work.groupby(key, sort=False):
        g = g.sort_values("cycle")
        cycles = g["cycle"]
        last_cycle = int(cycles.max())

        actionable = g["transition_outputs_actionable"] if "transition_outputs_actionable" in g.columns else None
        first_actionable_cycle: int | None = None
        if actionable is not None:
            mask = actionable.apply(lambda x: _safe_bool(x) is True)
            if mask.any():
                first_actionable_cycle = int(g.loc[mask, "cycle"].min())

        states = g["state"] if "state" in g.columns else pd.Series(["STABLE"] * len(g))
        severity_peak = max(_state_severity_numeric(str(x)) for x in states)

        watch_alert = states.astype(str).str.upper().isin({"WATCH", "ALERT"})
        persistence_score = float(watch_alert.mean()) if len(watch_alert) else 0.0

        traj_col = "decision_trajectory" if "decision_trajectory" in g.columns else None
        if traj_col is None and "trajectory_analysis_dominant_path" in g.columns:
            traj_col = "trajectory_analysis_dominant_path"
        traj_series = g[traj_col] if traj_col else pd.Series(["METASTABLE"] * len(g))
        dominant_trajectory = _dominant_preference(traj_series, _TRAJECTORY_ORDER, "METASTABLE")
        trajectory_div = _trajectory_divergence_numeric(dominant_trajectory)

        hz_col = "horizon_analysis_risk_horizon" if "horizon_analysis_risk_horizon" in g.columns else None
        hz_series = g[hz_col] if hz_col else pd.Series(["LONGER_HORIZON"] * len(g))
        dominant_horizon = _dominant_preference(hz_series, _HORIZON_ORDER, "LONGER_HORIZON")
        horizon_risk = _horizon_numeric(dominant_horizon)

        ts_col = "transition_state" if "transition_state" in g.columns else None
        ts_series = g[ts_col] if ts_col else pd.Series(["NONE"] * len(g))
        dominant_transition_state = _dominant_preference(ts_series, _TRANSITION_STATE_ORDER, "NONE")

        st_series = g["state"] if "state" in g.columns else pd.Series(["STABLE"] * len(g))
        dominant_state = _dominant_preference(st_series, _STATE_ORDER, "STABLE")

        urgency_col = "recommended_urgency" if "recommended_urgency" in g.columns else None
        if urgency_col:
            urgency_peak = max(_urgency_numeric(str(u)) for u in g[urgency_col])
        else:
            urgency_peak = 0.28

        lock_col = "lock_in_score" if "lock_in_score" in g.columns else "constraint_analysis_lock_in_score"
        if lock_col in g.columns:
            lock_in_max = max(_safe_float(x) for x in g[lock_col])
        else:
            lock_in_max = 0.0

        pnr_col = "point_of_no_return_risk" if "point_of_no_return_risk" in g.columns else "constraint_analysis_point_of_no_return_risk"
        if pnr_col in g.columns:
            pnr_max = max(_safe_float(x) for x in g[pnr_col])
        else:
            pnr_max = 0.0

        if "trust_score" in g.columns:
            mean_trust = float(g["trust_score"].apply(_safe_float).mean())
        else:
            mean_trust = 0.5

        if "safe_to_act" in g.columns:
            sb = g["safe_to_act"].apply(_safe_bool)
            valid = sb.notna()
            if valid.any():
                safe_to_act_rate = float(sb[valid].astype(bool).mean())
            else:
                safe_to_act_rate = 0.0
        else:
            safe_to_act_rate = 0.0

        max_drift = max(_safe_float(x) for x in g["structural_drift_score"]) if "structural_drift_score" in g.columns else 0.0
        max_inst = max(_safe_float(x) for x in g["latest_instability"]) if "latest_instability" in g.columns else 0.0
        max_pressure = max(_safe_float(x) for x in g["transition_pressure"]) if "transition_pressure" in g.columns else 0.0

        lead_min: float | None = None
        lead_last: float | None = None
        lead_cycles: int | None = None
        if "lead_time_hours" in g.columns:
            lt = g["lead_time_hours"].apply(lambda x: _safe_float(x, default=float("nan")))
            lt_valid = lt[lt > 0]
            if not lt_valid.empty:
                lead_min = float(lt_valid.min())
                lead_cycles = int(round(lead_min))
            last_row = g.iloc[-1]
            lv = _safe_float(last_row.get("lead_time_hours"), default=float("nan"))
            if lv == lv:  # not NaN
                lead_last = float(lv)

        fleet_priority_score = compute_fleet_priority_score_components(
            severity_peak=severity_peak,
            persistence_score=persistence_score,
            trajectory_div=trajectory_div,
            horizon_risk=horizon_risk,
            urgency_peak=urgency_peak,
            lock_in_max=lock_in_max,
            pnr_max=pnr_max,
            mean_trust=mean_trust,
            safe_to_act_rate=safe_to_act_rate,
        )

        rec = {
            key: asset_val,
            "last_cycle": last_cycle,
            "first_actionable_cycle": first_actionable_cycle,
            "lead_time_hours_min": lead_min,
            "lead_time_hours_at_last_cycle": lead_last,
            "lead_time_cycles": lead_cycles,
            "max_structural_drift": round(max_drift, 6),
            "max_latest_instability": round(max_inst, 6),
            "max_transition_pressure": round(max_pressure, 6),
            "dominant_transition_state": dominant_transition_state,
            "dominant_state": dominant_state,
            "dominant_trajectory": dominant_trajectory,
            "dominant_horizon": dominant_horizon,
            "highest_lock_in_score": round(lock_in_max, 6),
            "highest_point_of_no_return_risk": round(pnr_max, 6),
            "mean_trust_score": round(mean_trust, 6),
            "safe_to_act_rate": round(safe_to_act_rate, 6),
            "recommendation_urgency_peak": round(urgency_peak, 6),
            "fleet_priority_score": fleet_priority_score,
        }
        if "unit" in g.columns:
            try:
                rec["unit"] = int(g["unit"].iloc[0]) if pd.notna(g["unit"].iloc[0]) else g["unit"].iloc[0]
            except (TypeError, ValueError):
                rec["unit"] = g["unit"].iloc[0]
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        out["fleet_priority_rank"] = pd.Series(dtype=int)
        return out

    tie_asset = out[key].astype(str)
    out = out.assign(_tie=tie_asset)
    out = out.sort_values(
        ["fleet_priority_score", "_tie"],
        ascending=[False, True],
    ).reset_index(drop=True)
    out["fleet_priority_rank"] = range(1, len(out) + 1)
    out["fleet_rank"] = out["fleet_priority_rank"]
    out = out.drop(columns=["_tie"])
    return out


def rank_assets_by_priority(df: pd.DataFrame, *, asset_col: str = "unit") -> pd.DataFrame:
    """
    Backward compatible: same as :func:`compute_fleet_intelligence_table` plus ``max_priority_score`` alias.
    """
    t = compute_fleet_intelligence_table(df, asset_col=asset_col)
    if t.empty:
        return t
    t = t.copy()
    t["max_priority_score"] = t["fleet_priority_score"]
    return t


def build_fleet_summary(df: pd.DataFrame, *, asset_col: str = "unit") -> dict[str, Any]:
    """
    Structured fleet summary for JSON export: ranking table plus metadata.

    ``df`` is typically a slim or diagnostics replay DataFrame.
    """
    ranked = compute_fleet_intelligence_table(df, asset_col=asset_col)
    records = ranked.to_dict(orient="records")
    return {
        "schema_version": "2",
        "asset_count": int(len(ranked)),
        "ranked_assets": records,
        "top_asset": records[0] if records else None,
        "fleet_priority_score_explanation": FLEET_PRIORITY_SCORE_EXPLANATION,
        "ranking": "fleet_priority_rank 1 is highest priority (sort by fleet_priority_score descending)",
    }
