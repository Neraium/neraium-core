"""
Deterministic structural attribution from baseline vs recent covariance and current vector.

Read-only advisory outputs. No control or actuation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _corr_matrix(X: np.ndarray) -> np.ndarray:
    if X.shape[0] < 2:
        return np.eye(X.shape[1])
    c = np.corrcoef(X, rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    return c


def rank_signal_drivers(
    *,
    sensor_order: list[str],
    mean: np.ndarray,
    cov: np.ndarray,
    inv_cov: np.ndarray,
    current_vector: np.ndarray,
    baseline_matrix: np.ndarray | None,
    recent_matrix: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Per-signal drivers from current departure and recent variability."""
    n = len(sensor_order)
    if n == 0 or mean.shape[0] != n or current_vector.shape[0] != n:
        return []

    delta = np.nan_to_num(current_vector - mean, nan=0.0)
    diag_sig = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    try:
        partial = np.abs(inv_cov @ delta)
    except Exception:
        partial = np.abs(delta)

    total = float(np.sum(partial) + 1e-9)
    drivers: list[dict[str, Any]] = [
        {
            "signal": sensor_order[i],
            "contribution_score": round(float(partial[i] / total), 6),
            "baseline_deviation": round(float(abs(delta[i]) / (diag_sig[i] + 1e-9)), 6),
            "recent_instability": None,
        }
        for i in range(n)
    ]

    if recent_matrix is not None and recent_matrix.shape[0] >= 2 and recent_matrix.shape[1] == n:
        for i in range(n):
            col = recent_matrix[:, i]
            if np.isnan(col).any():
                drivers[i]["recent_instability"] = None
            else:
                rstd = float(np.std(col))
                bstd = float(np.std(baseline_matrix[:, i])) if baseline_matrix is not None and baseline_matrix.shape[0] >= 2 else 1.0
                drivers[i]["recent_instability"] = round(float(rstd / max(bstd, 1e-9)), 6)

    drivers.sort(key=lambda d: float(d["contribution_score"]), reverse=True)
    return drivers


def rank_signal_pairs(
    *,
    sensor_order: list[str],
    baseline_matrix: np.ndarray | None,
    recent_matrix: np.ndarray | None,
    max_pairs: int = 15,
) -> list[dict[str, Any]]:
    """Strongest pairwise correlation shifts between baseline and recent windows."""
    if baseline_matrix is None or recent_matrix is None:
        return []
    if baseline_matrix.shape[1] != recent_matrix.shape[1]:
        return []
    if baseline_matrix.shape[0] < 2 or recent_matrix.shape[0] < 2:
        return []

    cb = _corr_matrix(baseline_matrix)
    cr = _corr_matrix(recent_matrix)
    n = cb.shape[0]
    pairs: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            shift = float(abs(cr[i, j] - cb[i, j]))
            pairs.append(
                {
                    "signal_a": sensor_order[i],
                    "signal_b": sensor_order[j],
                    "relationship_shift_score": round(shift, 6),
                }
            )
    pairs.sort(key=lambda p: float(p["relationship_shift_score"]), reverse=True)
    return pairs[:max_pairs]


def infer_group_contributions(sensor_order: list[str], drivers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Simple prefix-based grouping when no explicit config exists.

    Returns empty list or list of {group, contribution_score}.
    """
    if not sensor_order or not drivers:
        return []

    groups: dict[str, float] = {}
    by_name = {d["signal"]: float(d["contribution_score"]) for d in drivers}
    for name, score in by_name.items():
        if len(name) < 2:
            g = "other"
        elif name.startswith("op"):
            g = "operating_condition"
        elif name.startswith("s"):
            g = "sensor"
        else:
            g = "other"
        groups[g] = groups.get(g, 0.0) + score

    total = sum(groups.values()) + 1e-9
    return [{"group": k, "contribution_score": round(v / total, 4)} for k, v in sorted(groups.items(), key=lambda x: -x[1])]


def localized_vs_systemic(experimental: dict[str, Any]) -> tuple[float, str]:
    """Map hierarchy origin_scope to score and label."""
    hierarchy = experimental.get("hierarchy_analysis") or {}
    if isinstance(hierarchy, dict) and hierarchy.get("available") is False:
        return 0.0, "LOCALIZED"
    scope = str((hierarchy or {}).get("origin_scope", "LOCAL")).upper()
    if scope == "SYSTEMIC":
        return 1.0, "SYSTEMIC"
    if scope == "MULTI_SUBSYSTEM":
        return 0.55, "MULTI_SUBSYSTEM"
    return 0.15, "LOCALIZED"


def infer_primary_structural_shift(
    *,
    drift_score: float,
    cov_drift: float,
    transition_pressure: float,
    transition_state: str,
    dominant_path: str,
    top_driver_share: float,
    pair_shift_max: float,
) -> str:
    """Choose a single human-readable label from engine signals."""
    if drift_score < 0.35 and transition_pressure < 0.25:
        return "stable baseline telemetry"

    if cov_drift > 1.2 and drift_score > 1.5:
        return "broad structural destabilization"

    if pair_shift_max > 0.45 and 0.4 < drift_score < 2.5:
        return "subsystem coupling loss"

    if top_driver_share > 0.45 and cov_drift < 0.9:
        return "localized divergence"

    dp = (dominant_path or "").upper()
    ts = (transition_state or "").upper()
    if transition_pressure >= 0.35 and (dp == "METASTABLE" or ts == "METASTABLE"):
        return "metastable transition pressure"

    if transition_pressure >= 0.5 or drift_score >= 1.2:
        return "broad structural destabilization"

    return "metastable transition pressure"


def attribution_confidence(
    *,
    readiness: dict[str, Any],
    frame_count: int,
    min_history_required: int,
    drivers: list[dict[str, Any]],
    geometry_available: bool,
    state_space_available: bool,
    state_graph_available: bool,
) -> float:
    """Deterministic 0..1 confidence in attribution."""
    structural = bool(readiness.get("structural_ready", False))
    transition_ready = bool(readiness.get("transition_classification_ready", False))
    hist_frac = min(1.0, float(frame_count) / float(max(1, min_history_required)))

    sep = 0.0
    if len(drivers) >= 2:
        c0 = float(drivers[0]["contribution_score"])
        c1 = float(drivers[1]["contribution_score"])
        sep = _clamp01((c0 - c1) / max(c0, 1e-6))

    sub = 0.0
    if geometry_available:
        sub += 0.12
    if state_space_available:
        sub += 0.12
    if state_graph_available:
        sub += 0.12

    score = (
        0.22 * (1.0 if structural else 0.4)
        + 0.18 * (1.0 if transition_ready else 0.5)
        + 0.25 * hist_frac
        + 0.18 * sep
        + sub
    )
    return round(_clamp01(score), 4)


def build_rationale(
    primary: str,
    loc_label: str,
    top_signal: str | None,
    top_pair: str | None,
) -> str:
    parts = [
        f"Primary shift: {primary}.",
        f"Scope: {loc_label}.",
    ]
    if top_signal:
        parts.append(f"Largest driver: {top_signal}.")
    if top_pair:
        parts.append(f"Strongest relationship change: {top_pair}.")
    return " ".join(parts)


def compute_structural_attribution(
    *,
    sensor_order: list[str],
    baseline: dict[str, Any] | None,
    current_vector: np.ndarray,
    baseline_matrix: np.ndarray | None,
    recent_matrix: np.ndarray | None,
    drift_score: float,
    covariance_drift: float,
    transition_pressure: float,
    transition_state: str,
    experimental_analytics: dict[str, Any],
    readiness: dict[str, Any],
    frame_count: int,
    min_history_required: int,
    geometry: dict[str, Any],
    state_space: dict[str, Any],
    state_graph: dict[str, Any],
) -> dict[str, Any]:
    """
    Full attribution payload. Returns ``available=False`` when history is insufficient.
    """
    if baseline is None or baseline_matrix is None or recent_matrix is None:
        return {
            "available": False,
            "reason": "insufficient history for baseline or recent covariance",
            "top_signal_drivers": [],
            "top_signal_pairs": [],
            "group_contributions": [],
            "primary_structural_shift": "stable baseline telemetry",
            "localized_vs_systemic_score": 0.0,
            "localized_vs_systemic_label": "LOCALIZED",
            "attribution_confidence": 0.0,
            "rationale": "Warmup: baseline or recent windows not yet available for structural attribution.",
        }

    mean = baseline["mean"]
    inv_cov = baseline["inv_cov"]

    cov = baseline["cov"]
    drivers = rank_signal_drivers(
        sensor_order=sensor_order,
        mean=mean,
        cov=cov,
        inv_cov=inv_cov,
        current_vector=current_vector,
        baseline_matrix=baseline_matrix,
        recent_matrix=recent_matrix,
    )
    pairs = rank_signal_pairs(sensor_order=sensor_order, baseline_matrix=baseline_matrix, recent_matrix=recent_matrix)
    groups = infer_group_contributions(sensor_order, drivers)

    loc_score, loc_label = localized_vs_systemic(experimental_analytics)

    traj = experimental_analytics.get("trajectory_analysis") or {}
    dominant_path = str((traj or {}).get("dominant_path", "METASTABLE"))
    if isinstance(traj, dict) and traj.get("available") is False:
        dominant_path = "UNAVAILABLE"

    top_share = float(drivers[0]["contribution_score"]) if drivers else 0.0
    pair_max = float(pairs[0]["relationship_shift_score"]) if pairs else 0.0

    primary = infer_primary_structural_shift(
        drift_score=drift_score,
        cov_drift=covariance_drift,
        transition_pressure=transition_pressure,
        transition_state=transition_state,
        dominant_path=dominant_path,
        top_driver_share=top_share,
        pair_shift_max=pair_max,
    )

    geo_ok = isinstance(geometry, dict) and geometry.get("available") is not False
    ss_ok = isinstance(state_space, dict) and state_space.get("available") is not False
    sg_ok = isinstance(state_graph, dict) and state_graph.get("available") is not False

    conf = attribution_confidence(
        readiness=readiness,
        frame_count=frame_count,
        min_history_required=min_history_required,
        drivers=drivers[:5],
        geometry_available=geo_ok,
        state_space_available=ss_ok,
        state_graph_available=sg_ok,
    )

    top_sig = drivers[0]["signal"] if drivers else None
    top_pair_str = None
    if pairs:
        top_pair_str = f"{pairs[0]['signal_a']} vs {pairs[0]['signal_b']}"

    rationale = build_rationale(primary, loc_label, top_sig, top_pair_str)

    return {
        "available": True,
        "reason": None,
        "top_signal_drivers": drivers[: min(12, len(drivers))],
        "top_signal_pairs": pairs,
        "group_contributions": groups,
        "primary_structural_shift": primary,
        "localized_vs_systemic_score": round(loc_score, 4),
        "localized_vs_systemic_label": loc_label,
        "attribution_confidence": conf,
        "rationale": rationale,
    }
