from __future__ import annotations

from typing import Mapping


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sensor_to_cluster_map(
    sensor_names: list[str], clusters: list[list[int]]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    n = len(sensor_names)
    sensor_cluster: dict[str, str] = {}
    cluster_members: dict[str, list[str]] = {}

    for cluster_index, members in enumerate(clusters):
        label = f"cluster_{cluster_index}"
        names: list[str] = []
        for idx in members:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= i < n:
                name = sensor_names[i]
                sensor_cluster[name] = label
                names.append(name)
        if names:
            cluster_members[label] = names

    for name in sensor_names:
        if name in sensor_cluster:
            continue
        singleton = f"cluster_{name}"
        sensor_cluster[name] = singleton
        cluster_members[singleton] = [name]

    return sensor_cluster, cluster_members


def analyze_hierarchy_cascade(
    *,
    sensor_names: list[str],
    subsystem: Mapping[str, object] | None,
    graph: Mapping[str, object] | None,
    causal_propagation: Mapping[str, object] | None,
    counterfactual_guidance: Mapping[str, object] | None,
    transition: Mapping[str, object] | None,
) -> dict[str, object]:
    if len(sensor_names) < 2 or not isinstance(subsystem, Mapping):
        return {
            "available": False,
            "reason": "relational_analytics_unavailable",
            "origin_scope": "LOCAL",
            "origin_subsystem": None,
            "propagation_risk": "LOW",
            "cascade_direction": [],
            "localized_vs_global_score": 0.0,
            "rationale": {
                "observation": "Hierarchy analysis unavailable because multivariate relational analytics are missing.",
            },
        }

    raw_clusters = subsystem.get("clusters")
    clusters = raw_clusters if isinstance(raw_clusters, list) else []
    if not clusters:
        return {
            "available": False,
            "reason": "missing_subsystem_structure",
            "origin_scope": "LOCAL",
            "origin_subsystem": None,
            "propagation_risk": "LOW",
            "cascade_direction": [],
            "localized_vs_global_score": 0.0,
            "rationale": {
                "observation": "Hierarchy analysis unavailable because subsystem clusters are missing.",
            },
        }

    sensor_cluster, cluster_members = _sensor_to_cluster_map(sensor_names, clusters)
    node_scores = {name: 0.0 for name in sensor_names}

    contributors = []
    if isinstance(counterfactual_guidance, Mapping):
        v = counterfactual_guidance.get("top_structural_break_contributors")
        if isinstance(v, list):
            contributors = v

    for item in contributors:
        if not isinstance(item, Mapping):
            continue
        score = _clamp01(float(item.get("contributor_score", 0.0)))
        a = item.get("sensor_a")
        b = item.get("sensor_b")
        if isinstance(a, str) and a in node_scores:
            node_scores[a] += score
        if isinstance(b, str) and b in node_scores:
            node_scores[b] += score

    spread_scores: list[float] = []
    top_pairs: list[Mapping[str, object]] = []
    if isinstance(causal_propagation, Mapping):
        raw_spread = causal_propagation.get("spread_scores")
        if isinstance(raw_spread, list):
            spread_scores = [float(v) for v in raw_spread[: len(sensor_names)]]
        raw_pairs = causal_propagation.get("top_pairs")
        if isinstance(raw_pairs, list):
            top_pairs = [p for p in raw_pairs if isinstance(p, Mapping)]

    if spread_scores:
        for idx, score in enumerate(spread_scores):
            if idx >= len(sensor_names):
                break
            node_scores[sensor_names[idx]] += 0.75 * _clamp01(score)

    max_node_score = max(node_scores.values(), default=0.0)
    cutoff = max(0.15, max_node_score * 0.42)
    affected_nodes = [name for name, score in node_scores.items() if score >= cutoff]

    cluster_scores: dict[str, float] = {}
    for cluster, members in cluster_members.items():
        vals = [node_scores.get(name, 0.0) for name in members]
        cluster_scores[cluster] = float(sum(vals) / max(1, len(vals)))

    ordered_clusters = sorted(cluster_scores.items(), key=lambda item: item[1], reverse=True)
    total_cluster_score = sum(cluster_scores.values())
    top_cluster, top_score = ordered_clusters[0]
    concentration = _clamp01(top_score / (total_cluster_score + 1e-9))
    affected_clusters = [name for name, score in ordered_clusters if score >= max(0.08, top_score * 0.4)]

    global_ratio = len(affected_nodes) / float(max(1, len(sensor_names)))
    globalness = _clamp01(0.62 * global_ratio + 0.38 * (1.0 - concentration))

    if globalness >= 0.68 or len(affected_clusters) >= max(3, int(round(0.6 * len(cluster_members)))):
        origin_scope = "SYSTEM_WIDE"
        origin_subsystem = None
    elif len(affected_clusters) >= 2:
        origin_scope = "REGIONAL"
        origin_subsystem = top_cluster
    else:
        origin_scope = "LOCAL"
        origin_subsystem = top_cluster

    cascade_direction: list[str] = []
    intercluster_pairs = 0
    for pair in top_pairs:
        src_idx = pair.get("src_idx")
        dst_idx = pair.get("dst_idx")
        try:
            src = sensor_names[int(src_idx)]
            dst = sensor_names[int(dst_idx)]
        except (TypeError, ValueError, IndexError):
            continue
        src_cluster = sensor_cluster.get(src)
        dst_cluster = sensor_cluster.get(dst)
        if not src_cluster or not dst_cluster or src_cluster == dst_cluster:
            continue
        intercluster_pairs += 1
        edge = f"{src_cluster} -> {dst_cluster}"
        if edge not in cascade_direction:
            cascade_direction.append(edge)

    intercluster_flow = _clamp01(intercluster_pairs / float(max(1, len(top_pairs))))
    transition_pressure = 0.0
    shock_activity = 0.0
    if isinstance(transition, Mapping):
        transition_pressure = _clamp01(float(transition.get("transition_pressure", 0.0)) / 1.35)
        shock_activity = _clamp01(
            max(float(transition.get("shock_triggered", 0.0)), float(transition.get("shock_boost", 0.0)))
        )

    spread_peak = _clamp01(max(spread_scores, default=0.0))
    broad_instability = _clamp01(globalness)
    risk_score = _clamp01(
        0.36 * intercluster_flow
        + 0.25 * spread_peak
        + 0.2 * transition_pressure
        + 0.12 * shock_activity
        + 0.07 * broad_instability
    )

    if risk_score >= 0.62:
        propagation_risk = "HIGH"
    elif risk_score >= 0.33:
        propagation_risk = "MODERATE"
    else:
        propagation_risk = "LOW"

    return {
        "available": True,
        "origin_scope": origin_scope,
        "origin_subsystem": origin_subsystem,
        "propagation_risk": propagation_risk,
        "cascade_direction": cascade_direction,
        "localized_vs_global_score": round(float(globalness), 4),
        "rationale": {
            "affected_node_ratio": round(float(global_ratio), 4),
            "affected_cluster_count": int(len(affected_clusters)),
            "cluster_concentration": round(float(concentration), 4),
            "intercluster_flow": round(float(intercluster_flow), 4),
            "transition_pressure_signal": round(float(transition_pressure), 4),
            "shock_activity_signal": round(float(shock_activity), 4),
        },
    }
