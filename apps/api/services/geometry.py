from __future__ import annotations

import math
from typing import Any

import numpy as np

STRUCTURAL_FLOW_PLANE_MAX_N = 24
GEOMETRY_VISUAL_MIN_NODES = 6
_STABLE_STRESS_CEILING = 0.33


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(f):
        return float(default)
    return float(f)


def _graph_analytics_payload(
    analytics_dict: dict[str, Any],
    *,
    feature_names: list[str],
) -> dict[str, Any] | None:
    g = analytics_dict.get("graph")
    cg = analytics_dict.get("causal_graph")
    if not isinstance(g, dict) and not isinstance(cg, dict):
        return None
    out: dict[str, Any] = {}
    if isinstance(g, dict):
        out["correlation_graph"] = {
            "mean_degree": round(_safe_float(g.get("mean_degree")), 6),
            "density": round(_safe_float(g.get("density")), 6),
            "clustering": round(_safe_float(g.get("clustering")), 6),
            "connectivity": round(_safe_float(g.get("connectivity")), 6),
            "mean_absolute_connectivity": round(_safe_float(g.get("mean_absolute_connectivity")), 6),
        }
    if isinstance(cg, dict):
        raw_ds = cg.get("dominant_sources")
        idx_list: list[int] = []
        if isinstance(raw_ds, list):
            for x in raw_ds:
                try:
                    idx_list.append(int(x))
                except (TypeError, ValueError):
                    continue
        labels: list[str] = []
        for i in idx_list:
            if 0 <= i < len(feature_names):
                labels.append(str(feature_names[i]))
        out["causal_graph"] = {
            "density": round(_safe_float(cg.get("density")), 6),
            "asymmetry": round(_safe_float(cg.get("asymmetry")), 6),
            "dominant_source_indices": idx_list,
            "dominant_source_labels": labels,
        }
    return out or None


def _system_state_payload(result: dict[str, Any], *, analytics_dict: dict[str, Any]) -> dict[str, Any]:
    structural_flag = result.get("structural_analysis_available")
    if structural_flag is None:
        structural_available = not bool(analytics_dict.get("relational_metrics_skipped", True))
    else:
        structural_available = bool(structural_flag)

    regime_mem = result.get("regime_memory_state")
    regime_mem_dict = regime_mem if isinstance(regime_mem, dict) else {}

    rs = analytics_dict.get("regime_signature")
    rs_dict = rs if isinstance(rs, dict) else {}
    nearest = rs_dict.get("nearest")
    nearest_dict = nearest if isinstance(nearest, dict) else {}

    out: dict[str, Any] = {
        "structural_analysis_available": structural_available,
        "phase": result.get("phase") or result.get("interpreted_state") or result.get("state"),
        "interpreted_state": result.get("interpreted_state") or result.get("state"),
        "regime_name": result.get("regime_name"),
        "confidence": round(
            _safe_float(
                result.get("confidence"),
                _safe_float(result.get("confidence_score"), 0.0),
            ),
            4,
        ),
        "regime_memory": {
            "regime_name": regime_mem_dict.get("regime_name"),
            "library_size": regime_mem_dict.get("library_size"),
            "baseline_count": regime_mem_dict.get("baseline_count"),
        },
        "regime_signature": {
            "assigned_name": rs_dict.get("assigned_name"),
            "library_size": rs_dict.get("library_size"),
        },
    }
    rd = result.get("regime_distance")
    if rd is not None:
        out["regime_distance"] = round(_safe_float(rd), 4)
    ndist = nearest_dict.get("distance")
    if ndist is not None:
        out["nearest_regime"] = {
            "name": nearest_dict.get("name"),
            "distance": round(_safe_float(ndist), 6),
        }
    return out


def _fallback_correlation_from_relationships(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.ones((1, 1), dtype=float)
    rho = min(0.28, 1.0 / max(float(n - 1), 1.0))
    c = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(c, 1.0)
    return c


def _block_pad_correlation_matrix(corr: np.ndarray, m_add: int) -> np.ndarray:
    n0 = int(corr.shape[0])
    m = int(m_add)
    if m <= 0:
        return corr
    a = np.asarray(corr, dtype=float)
    b = _fallback_correlation_from_relationships(m)
    out = np.zeros((n0 + m, n0 + m), dtype=float)
    out[:n0, :n0] = a
    out[n0:, n0:] = b
    np.fill_diagonal(out, 1.0)
    return out


def _expand_geometry_sensor_names(
    core_names: list[str],
    result: dict[str, Any],
    *,
    min_nodes: int,
    max_nodes: int,
) -> list[str]:
    out = [str(x).strip() for x in core_names if str(x).strip()]
    seen = set(out)
    if len(out) >= min_nodes or len(out) >= max_nodes:
        return out[:max_nodes]
    rel = result.get("sensor_relationships")
    if isinstance(rel, list):
        for x in rel:
            if len(out) >= max_nodes:
                break
            sx = str(x).strip()
            if sx and sx not in seen:
                out.append(sx)
                seen.add(sx)
            if len(out) >= min_nodes:
                return out[:max_nodes]
    sv = result.get("sensor_values")
    if isinstance(sv, dict):
        for k in sv.keys():
            if len(out) >= max_nodes:
                break
            sx = str(k).strip()
            if sx and sx not in seen:
                out.append(sx)
                seen.add(sx)
            if len(out) >= min_nodes:
                break
    return out[:max_nodes]


def _to_square_matrix(value: Any) -> np.ndarray | None:
    try:
        mat = np.asarray(value, dtype=float)
    except Exception:
        return None
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        return None
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(mat, 1.0)
    return mat


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-9:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def _vertical_lift_y_from_stress(
    stress01: float,
    *,
    drift_global: float,
    inst_global: float,
    lift_max: float = 0.46,
) -> float:
    drift_g = float(np.clip(drift_global, 0.0, 1.0))
    inst_g = float(np.clip(inst_global, 0.0, 1.0))
    global_scale = 1.0 + 0.45 * drift_g + 0.4 * inst_g
    s = float(np.clip(stress01, 0.0, 1.0))
    sev = max(0.0, (s - 0.12) / 0.88) ** 1.12
    sev = min(1.0, sev)
    return float(lift_max * sev * global_scale)


def _lift_y_structural_flow(
    i: int,
    stress_norm: np.ndarray,
    *,
    drift_global: float,
    inst_global: float,
) -> float:
    s = float(stress_norm[i]) if i < len(stress_norm) else 0.0
    if s < _STABLE_STRESS_CEILING:
        return 0.0
    return _vertical_lift_y_from_stress(s, drift_global=drift_global, inst_global=inst_global)


def _triangle_center_plane_four(
    stress_norm: np.ndarray,
    *,
    drift_global: float,
    inst_global: float,
    corr_matrix: np.ndarray | None = None,
) -> np.ndarray:
    _ = corr_matrix
    r = 0.62
    out = np.zeros((4, 3), dtype=float)
    for k in range(3):
        theta = math.pi / 2.0 + float(k) * (2.0 * math.pi / 3.0)
        out[k, 0] = r * math.cos(theta)
        out[k, 2] = r * math.sin(theta)
        out[k, 1] = _lift_y_structural_flow(k, stress_norm, drift_global=drift_global, inst_global=inst_global)
    out[3, 0] = 0.0
    out[3, 2] = 0.0
    out[3, 1] = _lift_y_structural_flow(3, stress_norm, drift_global=drift_global, inst_global=inst_global)
    return out


def _diamond_plane_positions_four(
    stress_norm: np.ndarray,
    *,
    drift_global: float,
    inst_global: float,
    corr_matrix: np.ndarray | None = None,
) -> np.ndarray:
    return _triangle_center_plane_four(
        stress_norm,
        drift_global=drift_global,
        inst_global=inst_global,
        corr_matrix=corr_matrix,
    )


def _plane_ring_positions(
    n: int,
    stress_norm: np.ndarray,
    *,
    drift_global: float,
    inst_global: float,
    corr_matrix: np.ndarray | None = None,
) -> np.ndarray:
    _ = corr_matrix
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    out = np.zeros((n, 3), dtype=float)

    def _lift(i: int) -> float:
        return _lift_y_structural_flow(i, stress_norm, drift_global=drift_global, inst_global=inst_global)

    if n == 2:
        r = 0.56
        angles = (math.pi / 4.0, 3.0 * math.pi / 4.0)
        for i in range(2):
            theta = angles[i]
            out[i, 0] = r * math.cos(theta)
            out[i, 2] = r * math.sin(theta)
            out[i, 1] = _lift(i)
        return out

    if n <= 12:
        r = 0.5 + 0.02 * float(min(n, 12))
        for i in range(n):
            theta = 2.0 * math.pi * float(i) / float(n) - math.pi / 2.0
            out[i, 0] = r * math.cos(theta)
            out[i, 2] = r * math.sin(theta)
            out[i, 1] = _lift(i)
        return out

    n_inner = n // 2
    n_outer = n - n_inner
    r_in = 0.44
    r_out = 0.74 + 0.006 * float(max(0, n - 14))
    for i in range(n_inner):
        theta = 2.0 * math.pi * float(i) / float(max(n_inner, 1)) - math.pi / 2.0
        out[i, 0] = r_in * math.cos(theta)
        out[i, 2] = r_in * math.sin(theta)
        out[i, 1] = _lift(i)
    phase_off = math.pi / float(max(n_outer * 2, 1))
    for j in range(n_outer):
        idx = n_inner + j
        theta = 2.0 * math.pi * float(j) / float(max(n_outer, 1)) - math.pi / 2.0 + phase_off
        out[idx, 0] = r_out * math.cos(theta)
        out[idx, 2] = r_out * math.sin(theta)
        out[idx, 1] = _lift(idx)
    return out


def _inflate_spectral_embedding_if_degenerate(
    pos: np.ndarray,
    *,
    corr_matrix: np.ndarray,
    stress_norm: np.ndarray,
) -> np.ndarray:
    _ = corr_matrix
    _ = stress_norm
    n = int(pos.shape[0])
    if n < 3:
        return pos
    centered = pos - np.mean(pos, axis=0, keepdims=True)
    c = np.cov(centered.T)
    if not np.all(np.isfinite(c)):
        return pos
    ev, vecs = np.linalg.eigh(c)
    ev = np.maximum(ev, 0.0)
    if ev[-1] <= 1e-14:
        return pos
    if float(ev[-2] / ev[-1]) > 0.05:
        return pos
    e2 = vecs[:, -2]
    e3 = vecs[:, -3]
    out = pos.copy()
    scale = 0.048
    for i in range(n):
        ang = 2.0 * math.pi * float(i) / float(n)
        delta = e2 * (scale * math.cos(ang)) + e3 * (scale * math.sin(ang))
        out[i, :] += delta
    return out


def _project_geometry_positions(
    corr_current: np.ndarray,
    *,
    node_stress: np.ndarray,
    corr_baseline: np.ndarray | None,
    metrics: dict[str, Any] | None = None,
) -> np.ndarray:
    n = int(corr_current.shape[0])
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    if n == 1:
        return np.asarray([[0.0, 0.0, 0.0]], dtype=float)
    drift_g = _safe_float(metrics.get("structural_drift_score"), 0.0) if metrics else 0.0
    inst_g = _safe_float(metrics.get("composite_instability"), 0.0) if metrics else 0.0
    if n == 4 and len(node_stress) >= 4:
        return _diamond_plane_positions_four(
            node_stress,
            drift_global=drift_g,
            inst_global=inst_g,
            corr_matrix=corr_current,
        )
    if 2 <= n <= STRUCTURAL_FLOW_PLANE_MAX_N and len(node_stress) >= n:
        return _plane_ring_positions(
            n,
            node_stress,
            drift_global=drift_g,
            inst_global=inst_g,
            corr_matrix=corr_current,
        )

    vals, vecs = np.linalg.eigh(corr_current)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    def _axis(component_idx: int) -> np.ndarray:
        if component_idx >= vecs.shape[1]:
            return np.zeros((n,), dtype=float)
        scale = float(np.sqrt(max(vals[component_idx], 1e-6)))
        return np.asarray(vecs[:, component_idx], dtype=float) * scale

    axis_x = _axis(0)
    axis_y = _axis(1)
    if corr_baseline is not None and corr_baseline.shape == corr_current.shape:
        axis_z = np.mean(np.abs(corr_current - corr_baseline), axis=1)
    else:
        axis_z = _axis(2)

    for axis in (axis_x, axis_y, axis_z):
        max_abs = float(np.max(np.abs(axis))) if axis.size else 0.0
        if max_abs > 1e-9:
            axis /= max_abs

    spread = float(np.std(axis_x) + np.std(axis_y))
    if spread < 1e-5:
        theta = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
        axis_x = np.cos(theta)
        axis_y = np.sin(theta)

    axis_z = 0.55 * axis_z + 0.45 * (2.0 * _normalize_vector(node_stress) - 1.0)
    max_abs_z = float(np.max(np.abs(axis_z))) if axis_z.size else 0.0
    if max_abs_z > 1e-9:
        axis_z /= max_abs_z

    pos = np.stack([axis_x, axis_y, axis_z], axis=1)
    return _inflate_spectral_embedding_if_degenerate(pos, corr_matrix=corr_current, stress_norm=node_stress)


def _stress_state(value: float) -> str:
    if value >= 0.66:
        return "critical"
    if value >= 0.33:
        return "watch"
    return "stable"


def _build_geometry_nodes(
    *,
    feature_names: list[str],
    positions: np.ndarray,
    magnitude_norm: np.ndarray,
    stress_norm: np.ndarray,
    core_name_set: set[str] | None = None,
) -> list[dict[str, Any]]:
    n = min(len(feature_names), int(positions.shape[0]), int(magnitude_norm.shape[0]), int(stress_norm.shape[0]))
    out: list[dict[str, Any]] = []
    for idx in range(n):
        stress = float(stress_norm[idx])
        state = _stress_state(stress)
        nm = str(feature_names[idx]).strip()
        in_corr = True if core_name_set is None else nm in core_name_set
        out.append(
            {
                "id": nm,
                "label": nm,
                "position": {
                    "x": round(float(positions[idx, 0]), 6),
                    "y": round(float(positions[idx, 1]), 6),
                    "z": round(float(positions[idx, 2]), 6),
                },
                "magnitude": round(float(magnitude_norm[idx]), 6),
                "stress": round(stress, 6),
                "state": state,
                "unstable": state == "critical",
                "is_unstable": state == "critical",
                "in_range": state == "stable",
                "in_correlation_window": in_corr,
                "role": "signal",
            }
        )
    return out


def _build_geometry_edges(
    corr_matrix: np.ndarray,
    *,
    feature_names: list[str],
    baseline_ref: np.ndarray | None = None,
    limit: int = 240,
    full_connectivity_max_n: int = STRUCTURAL_FLOW_PLANE_MAX_N,
) -> list[dict[str, Any]]:
    n = int(corr_matrix.shape[0])
    out: list[dict[str, Any]] = []
    if n <= 1:
        return out
    use_full = n <= int(full_connectivity_max_n)
    threshold = 0.0
    if not use_full:
        upper_idx = np.triu_indices(n, k=1)
        upper_abs = np.abs(corr_matrix[upper_idx])
        threshold = float(np.clip(np.percentile(upper_abs, 72.0), 0.22, 0.78)) if upper_abs.size else 1.1

    for i in range(n):
        for j in range(i + 1, n):
            weight = float(corr_matrix[i, j])
            magnitude = abs(weight)
            if not use_full and magnitude < threshold:
                continue
            baseline_weight = float(baseline_ref[i, j]) if baseline_ref is not None else weight
            delta = weight - baseline_weight
            out.append(
                {
                    "source": str(feature_names[i]),
                    "target": str(feature_names[j]),
                    "weight": round(weight, 6),
                    "magnitude": round(magnitude, 6),
                    "delta": round(delta, 6),
                    "type": "positive" if weight >= 0.0 else "negative",
                }
            )
    out.sort(key=lambda e: float(e.get("magnitude", 0.0)), reverse=True)
    return out[: max(1, int(limit))]


def build_geometry_payload(result: dict[str, Any], *, run_id: str | None) -> dict[str, Any]:
    analytics = result.get("experimental_analytics")
    analytics_dict = analytics if isinstance(analytics, dict) else {}
    corr_geometry = analytics_dict.get("correlation_geometry")
    corr_geometry_dict = corr_geometry if isinstance(corr_geometry, dict) else {}
    corr_current = _to_square_matrix(corr_geometry_dict.get("current"))
    corr_baseline = _to_square_matrix(corr_geometry_dict.get("baseline"))
    if corr_current is not None and corr_baseline is not None and corr_baseline.shape != corr_current.shape:
        corr_baseline = None

    feature_names: list[str] = []
    names_from_analytics = analytics_dict.get("valid_sensor_names") or analytics_dict.get("feature_names")
    if isinstance(names_from_analytics, list):
        feature_names = [str(v) for v in names_from_analytics if str(v).strip()]
    if not feature_names:
        raw = result.get("sensor_relationships")
        if isinstance(raw, list):
            feature_names = [str(v) for v in raw if str(v).strip()]

    metrics = {
        "state": result.get("state") or result.get("interpreted_state"),
        "phase": result.get("phase") or result.get("interpreted_state") or result.get("state"),
        "risk_level": result.get("risk_level", "UNKNOWN"),
        "trend": result.get("trend", "UNKNOWN"),
        "structural_drift_score": _safe_float(result.get("structural_drift_score"), 0.0),
        "composite_instability": _safe_float(
            result.get("latest_instability"),
            _safe_float(analytics_dict.get("composite_instability"), 0.0),
        ),
        "confidence": _safe_float(result.get("confidence"), _safe_float(result.get("confidence_score"), 0.0)),
    }
    projection = {
        "method": "structural_flow_coherent_plane_or_spectral",
        "is_visualization_projection": True,
        "source": "engine correlation geometry + graph analytics",
        "layout": "coherent_plane_ring",
        "plane_axes": ["x", "z"],
        "vertical_axis": "y",
        "note": (
            f"2–{STRUCTURAL_FLOW_PLANE_MAX_N} sensors: deterministic layout on shared XZ (triangle+hub when "
            f"n=4, even ring when n≤12, dual ring when n>12); in-range sensors stay on the plane; "
            f"out-of-range lift on +Y. Larger counts use spectral projection. Visualization-only."
        ),
    }
    provenance = {
        "engine_fields": [
            "sensor_relationships",
            "experimental_analytics.correlation_geometry.current",
            "experimental_analytics.correlation_geometry.baseline",
            "experimental_analytics.signal_structural_importance",
            "experimental_analytics.graph",
            "experimental_analytics.causal_graph",
            "experimental_analytics.regime_signature",
            "regime_memory_state",
            "risk_level",
            "state",
            "interpreted_state",
            "trend",
            "structural_drift_score",
            "latest_instability",
            "confidence/confidence_score",
        ],
        "positions": "deterministic projection from engine outputs",
    }

    try:
        result_id = int(result.get("result_id")) if result.get("result_id") is not None else None
    except (TypeError, ValueError):
        result_id = None

    geometry_fallback = False
    if corr_current is None:
        if len(feature_names) < 1:
            graph_analytics = _graph_analytics_payload(analytics_dict, feature_names=feature_names)
            system_state = _system_state_payload(result, analytics_dict=analytics_dict)
            return {
                "run_id": run_id or result.get("run_id"),
                "result_id": result_id,
                "timestamp": result.get("timestamp") or result.get("persisted_at"),
                "available": False,
                "reason": "No sensor relationships or correlation geometry available for this result.",
                "metrics": metrics,
                "nodes": [],
                "edges": [],
                "views": {},
                "summary": {},
                "projection": projection,
                "provenance": provenance,
                "graph_analytics": graph_analytics,
                "system_state": system_state,
            }
        corr_current = _fallback_correlation_from_relationships(len(feature_names))
        corr_baseline = None
        geometry_fallback = True

    n = int(corr_current.shape[0])
    if len(feature_names) < n:
        feature_names = feature_names + [f"signal_{i + 1}" for i in range(len(feature_names), n)]
    if len(feature_names) > n:
        feature_names = feature_names[:n]

    core_names_before_expand = list(feature_names)
    n_core_corr = n
    expanded_names = _expand_geometry_sensor_names(
        feature_names,
        result,
        min_nodes=GEOMETRY_VISUAL_MIN_NODES,
        max_nodes=STRUCTURAL_FLOW_PLANE_MAX_N,
    )
    visual_expansion_count = 0
    if len(expanded_names) > n:
        visual_expansion_count = len(expanded_names) - n
        corr_current = _block_pad_correlation_matrix(corr_current, visual_expansion_count)
        if corr_baseline is not None and corr_baseline.shape == (n, n):
            corr_baseline = _block_pad_correlation_matrix(corr_baseline, visual_expansion_count)
        feature_names = expanded_names
        n = int(corr_current.shape[0])

    core_name_set = {str(x).strip() for x in core_names_before_expand if str(x).strip()}

    importance_raw = analytics_dict.get("signal_structural_importance")
    if isinstance(importance_raw, list) and len(importance_raw) >= n:
        importance = np.asarray([_safe_float(v, 0.0) for v in importance_raw[:n]], dtype=float)
    else:
        corr_abs = np.abs(corr_current - np.eye(n))
        importance = np.mean(corr_abs, axis=1)
    importance_norm = _normalize_vector(importance)

    if corr_baseline is not None and corr_baseline.shape == corr_current.shape:
        stress_raw = np.mean(np.abs(corr_current - corr_baseline), axis=1)
    else:
        stress_raw = importance.copy()
    stress_norm = _normalize_vector(stress_raw)

    if 2 <= n <= STRUCTURAL_FLOW_PLANE_MAX_N:
        perm = np.array(sorted(range(n), key=lambda i: str(feature_names[i]).lower()), dtype=int)
        feature_names = [feature_names[i] for i in perm]
        corr_current = corr_current[np.ix_(perm, perm)]
        if corr_baseline is not None:
            corr_baseline = corr_baseline[np.ix_(perm, perm)]
        stress_norm = stress_norm[perm]
        importance_norm = importance_norm[perm]

    positions = _project_geometry_positions(
        corr_current,
        node_stress=stress_norm,
        corr_baseline=corr_baseline,
        metrics=metrics,
    )

    nodes_current = _build_geometry_nodes(
        feature_names=feature_names,
        positions=positions,
        magnitude_norm=importance_norm,
        stress_norm=stress_norm,
        core_name_set=core_name_set,
    )
    edges_current = _build_geometry_edges(
        corr_current,
        feature_names=feature_names,
        baseline_ref=corr_baseline,
        limit=400,
    )

    corr_reference = corr_baseline if corr_baseline is not None else corr_current
    importance_reference = np.mean(np.abs(corr_reference - np.eye(n)), axis=1)
    importance_reference_norm = _normalize_vector(importance_reference)
    stress_reference_norm = importance_reference_norm.copy()
    positions_reference = _project_geometry_positions(
        corr_reference,
        node_stress=stress_reference_norm,
        corr_baseline=corr_reference,
        metrics=metrics,
    )
    nodes_baseline = _build_geometry_nodes(
        feature_names=feature_names,
        positions=positions_reference,
        magnitude_norm=importance_reference_norm,
        stress_norm=stress_reference_norm,
        core_name_set=core_name_set,
    )
    edges_baseline = _build_geometry_edges(
        corr_reference,
        feature_names=feature_names,
        baseline_ref=corr_reference,
        limit=240,
    )
    baseline_by_id = {str(n.get("id")): n for n in nodes_baseline}
    for node in nodes_current:
        node_id = str(node.get("id"))
        base_node = baseline_by_id.get(node_id) or {}
        node["position_current"] = dict(node.get("position") or {})
        node["position_baseline"] = dict(base_node.get("position") or node.get("position") or {})

    summary = {
        "critical_nodes_current": sum(1 for n in nodes_current if str(n.get("state")) == "critical"),
        "watch_nodes_current": sum(1 for n in nodes_current if str(n.get("state")) == "watch"),
        "unstable_nodes_current": sum(1 for n in nodes_current if bool(n.get("unstable"))),
        "changed_edges_current": sum(1 for e in edges_current if abs(_safe_float(e.get("delta"), 0.0)) >= 0.10),
        "in_range_nodes": sum(1 for n in nodes_current if bool(n.get("in_range"))),
        "out_of_range_nodes": sum(1 for n in nodes_current if not bool(n.get("in_range"))),
    }

    graph_analytics = _graph_analytics_payload(analytics_dict, feature_names=feature_names)
    system_state = _system_state_payload(result, analytics_dict=analytics_dict)

    projection_out = dict(projection)
    if n == 4:
        projection_out["layout"] = "triangle_center_plane_four_sensors"
        projection_out["plane_axes"] = ["x", "z"]
        projection_out["vertical_axis"] = "y"
        projection_out["method"] = "triangle_center_plane_four_with_vertical_lift"
        projection_out["ring_node_count"] = 4
    elif 2 <= n <= STRUCTURAL_FLOW_PLANE_MAX_N:
        projection_out["layout"] = "coherent_plane_ring"
        projection_out["plane_axes"] = ["x", "z"]
        projection_out["vertical_axis"] = "y"
        projection_out["method"] = (
            "regular_polygon_plane_ring_with_vertical_lift"
            if n <= 12
            else "dual_ring_plane_with_vertical_lift"
        )
        projection_out["ring_node_count"] = n
    else:
        projection_out["layout"] = "spectral_correlation_projection"
        projection_out["plane_axes"] = None
        projection_out["vertical_axis"] = None
        projection_out["method"] = "spectral_projection_from_engine_correlation_geometry"
        projection_out.pop("ring_node_count", None)
    if geometry_fallback:
        projection_out["geometry_fallback"] = True
        base_note = str(projection_out.get("note") or "")
        projection_out["note"] = (
            base_note
            + " Correlation matrix was synthesized from sensor names because stored "
            "correlation_geometry.current was unavailable; baseline correlation is not available."
        ).strip()

    projection_out["correlation_core_count"] = int(n_core_corr)
    if visual_expansion_count > 0:
        projection_out["visual_expansion"] = {
            "enabled": True,
            "added_sensor_count": int(visual_expansion_count),
            "note": (
                "Additional channels from this run catalog were included so the structural view "
                "shows broader asset coverage. They use a separate correlation cluster (no cross-correlation "
                "to the engine estimation window); see each node's in_correlation_window flag."
            ),
        }

    return {
        "run_id": run_id or result.get("run_id"),
        "result_id": result_id,
        "timestamp": result.get("timestamp") or result.get("persisted_at"),
        "available": True,
        "reason": None,
        "metrics": metrics,
        "nodes": nodes_current,
        "edges": edges_current,
        "views": {
            "current": {
                "label": "Current structure",
                "source": "experimental_analytics.correlation_geometry.current",
                "available": True,
                "nodes": nodes_current,
                "edges": edges_current,
            },
            "baseline": {
                "label": "Baseline structure",
                "source": (
                    "experimental_analytics.correlation_geometry.baseline"
                    if corr_baseline is not None
                    else "baseline_not_available_using_current_structure_as_reference"
                ),
                "available": corr_baseline is not None,
                "nodes": nodes_baseline,
                "edges": edges_baseline,
            },
        },
        "summary": summary,
        "projection": projection_out,
        "provenance": provenance,
        "graph_analytics": graph_analytics,
        "system_state": system_state,
    }
