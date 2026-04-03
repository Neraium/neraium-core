from __future__ import annotations

import json
from typing import Any, Literal


def csv_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    if any(ch in text for ch in [",", "\"", "\n", "\r"]):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def build_export(results: list[dict[str, Any]], *, format_name: Literal["json", "csv"]) -> tuple[str, str]:
    if format_name == "json":
        return ("application/json; charset=utf-8", json.dumps(results, indent=2, sort_keys=False))

    header = [
        "result_id", "run_id", "timestamp", "site_id", "asset_id", "state", "phase", "risk_level", "trend",
        "operator_message", "structural_drift_score", "composite_instability", "persisted_at", "geometry_path_length",
        "geometry_local_velocity_norm", "geometry_local_acceleration_norm", "geometry_curvature",
        "geometry_directional_consistency", "geometry_angular_change", "geometry_path_smoothness",
        "state_space_statistics_local_volume", "state_space_statistics_local_density", "state_space_statistics_covariance_trace",
        "state_space_statistics_principal_direction_strength", "state_space_statistics_anisotropy",
        "state_space_statistics_state_contraction_score", "state_space_statistics_state_expansion_score",
        "state_graph_node_count", "state_graph_edge_count", "state_graph_branching_factor", "state_graph_transition_entropy",
        "state_graph_revisit_rate", "state_graph_path_commitment_score", "state_graph_graph_divergence_score",
    ]
    lines = [",".join(header)]
    for row in results:
        composite = None
        analytics = row.get("experimental_analytics")
        if isinstance(analytics, dict):
            raw = analytics.get("composite_instability")
            if raw is not None:
                try:
                    composite = float(raw)
                except (TypeError, ValueError):
                    composite = None
        geom = row.get("geometry") if isinstance(row.get("geometry"), dict) else {}
        stats = row.get("state_space_statistics") if isinstance(row.get("state_space_statistics"), dict) else {}
        sgraph = row.get("state_graph") if isinstance(row.get("state_graph"), dict) else {}
        data = [
            row.get("result_id"), row.get("run_id"), row.get("timestamp"), row.get("site_id"), row.get("asset_id"),
            row.get("state"), row.get("phase"), row.get("risk_level"), row.get("trend"), row.get("operator_message"),
            row.get("structural_drift_score"), composite, row.get("persisted_at"), geom.get("path_length"),
            geom.get("local_velocity_norm"), geom.get("local_acceleration_norm"), geom.get("curvature"),
            geom.get("directional_consistency"), geom.get("angular_change"), geom.get("path_smoothness"),
            stats.get("local_volume"), stats.get("local_density"), stats.get("covariance_trace"),
            stats.get("principal_direction_strength"), stats.get("anisotropy"), stats.get("state_contraction_score"),
            stats.get("state_expansion_score"), sgraph.get("node_count"), sgraph.get("edge_count"),
            sgraph.get("branching_factor"), sgraph.get("transition_entropy"), sgraph.get("revisit_rate"),
            sgraph.get("path_commitment_score"), sgraph.get("graph_divergence_score"),
        ]
        lines.append(",".join(csv_escape(v) for v in data))
    return ("text/csv; charset=utf-8", "\n".join(lines))
