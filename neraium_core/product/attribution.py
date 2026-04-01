"""Attribution summaries from structural, geometry, and graph signals (explainable, deterministic)."""

from __future__ import annotations

from typing import Any

from neraium_core.product._util import clamp01, safe_float, safe_get


def _scope_to_score_label(origin_scope: str) -> tuple[float, str]:
    s = (origin_scope or "LOCAL").upper()
    if s == "SYSTEMIC":
        return 1.0, "systemic"
    if s == "MULTI_SUBSYSTEM":
        return 0.55, "multi_subsystem"
    return 0.15, "localized"


def build_attribution(result: dict[str, Any]) -> dict[str, Any]:
    experimental = safe_get(result, "experimental_analytics") or {}
    geometry = safe_get(result, "geometry") or {}
    state_space = safe_get(result, "state_space_statistics") or {}
    state_graph = safe_get(result, "state_graph") or {}
    early_warning = safe_get(result, "early_warning") or {}

    drift = safe_float(result.get("structural_drift_score"))
    pressure = safe_float(result.get("transition_pressure"))
    vel = safe_float(result.get("drift_velocity"))
    mahal = safe_float(result.get("mahalanobis_score"))
    cov_d = safe_float(result.get("covariance_drift_score"))

    pre_inst = clamp01(safe_float(early_warning.get("pre_instability_score", 0.0)))

    curvature = clamp01(safe_float(geometry.get("curvature", 0.0))) if geometry.get("available") is not False else 0.0
    dir_cons = clamp01(safe_float(geometry.get("directional_consistency", 0.0))) if geometry.get("available") is not False else 0.0

    contraction = clamp01(safe_float(state_space.get("state_contraction_score", 0.0))) if state_space.get("available") is not False else 0.0

    branching = clamp01(safe_float(state_graph.get("branching_factor", 0.0))) if state_graph.get("available") is not False else 0.0
    path_commit = clamp01(safe_float(state_graph.get("path_commitment_score", 0.0))) if state_graph.get("available") is not False else 0.0

    lock_in = clamp01(safe_float((experimental.get("constraint_analysis") or {}).get("lock_in_score", 0.0)))

    hierarchy = experimental.get("hierarchy_analysis") or {}
    origin_scope = str((hierarchy or {}).get("origin_scope", "LOCAL"))
    loc_score, loc_label = _scope_to_score_label(origin_scope)

    traj = experimental.get("trajectory_analysis") or {}
    dominant_path = str((traj or {}).get("dominant_path", "METASTABLE"))
    transition_state = str(result.get("transition_state", "NONE"))

    drivers: list[dict[str, Any]] = [
        {"name": "structural_drift", "contribution": clamp01(drift / 5.0), "direction": "higher_is_worse"},
        {"name": "transition_pressure", "contribution": clamp01(pressure), "direction": "higher_is_worse"},
        {"name": "pre_instability", "contribution": pre_inst, "direction": "higher_is_worse"},
        {"name": "relational_covariance_drift", "contribution": clamp01(cov_d / max(1.0, abs(mahal) + cov_d + 1e-6)), "direction": "higher_is_worse"},
        {"name": "manifold_departure", "contribution": clamp01(mahal / max(1.0, mahal + cov_d + 1e-6)), "direction": "higher_is_worse"},
        {"name": "geometry_curvature", "contribution": curvature, "direction": "higher_is_worse"},
        {"name": "lock_in_pressure", "contribution": lock_in, "direction": "higher_is_worse"},
    ]
    drivers.sort(key=lambda x: float(x["contribution"]), reverse=True)
    top_drivers = drivers[:6]

    pairs: list[dict[str, Any]] = [
        {"pair": "drift_vs_pressure", "score": clamp01(0.5 * (drift / 5.0) + 0.5 * pressure)},
        {"pair": "drift_vs_pre_instability", "score": clamp01(0.5 * (drift / 5.0) + 0.5 * pre_inst)},
        {"pair": "pressure_vs_lock_in", "score": clamp01(0.5 * pressure + 0.5 * lock_in)},
        {"pair": "curvature_vs_branching", "score": clamp01(0.5 * curvature + 0.5 * branching)},
    ]
    pairs.sort(key=lambda x: float(x["score"]), reverse=True)

    structure_score = clamp01(0.45 * (drift / 5.0) + 0.35 * (cov_d / max(0.01, cov_d + mahal)) + 0.20 * abs(vel) / 0.35)
    temporal_score = clamp01(min(1.0, abs(vel) / 0.25))
    geometry_score = clamp01(0.5 * curvature + 0.5 * (1.0 - dir_cons))
    graph_score = clamp01(0.5 * branching + 0.5 * path_commit)
    total = structure_score + temporal_score + geometry_score + graph_score + 1e-9
    group_contributions = {
        "structure": round(structure_score / total, 4),
        "temporal_velocity": round(temporal_score / total, 4),
        "geometry": round(geometry_score / total, 4),
        "state_graph": round(graph_score / total, 4),
    }

    primary_shift = f"{transition_state} trajectory:{dominant_path}"

    readiness = safe_get(result, "readiness") or {}
    structural_ready = bool(readiness.get("structural_ready", False))
    transition_ready = bool(result.get("transition_outputs_actionable", False))
    attr_conf = clamp01(
        0.35 * (1.0 if structural_ready else 0.35)
        + 0.35 * (1.0 if transition_ready else 0.45)
        + 0.15 * (1.0 if geometry.get("available") is not False else 0.4)
        + 0.15 * (1.0 if state_graph.get("available") is not False else 0.4)
    )

    return {
        "top_signal_drivers": top_drivers,
        "top_signal_pairs": pairs[:4],
        "group_contributions": group_contributions,
        "primary_structural_shift": primary_shift,
        "localized_vs_systemic_score": round(loc_score, 4),
        "localized_vs_systemic_label": loc_label,
        "attribution_confidence": round(attr_conf, 4),
    }
