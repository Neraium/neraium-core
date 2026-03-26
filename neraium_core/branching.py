from __future__ import annotations

import math
from typing import Mapping


PATH_KEYS: tuple[str, ...] = ("stabilizing", "metastable", "diverging")
PATH_LABELS: dict[str, str] = {
    "stabilizing": "STABILIZING",
    "metastable": "METASTABLE",
    "diverging": "DIVERGING",
}

BRANCH_MARGIN_MAX = 0.15
BRANCH_ENTROPY_MIN = 0.85
COMMITMENT_HIGH_MARGIN = 0.40
COMMITMENT_HIGH_ENTROPY_MAX = 0.70
COMMITMENT_MODERATE_MARGIN = 0.22
COMMITMENT_MODERATE_ENTROPY_MAX = 0.90


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return max(0.0, score)


def _normalize_path_scores(path_scores: Mapping[str, object]) -> dict[str, float]:
    cleaned = {k: _safe_score(path_scores.get(k, 0.0)) for k in PATH_KEYS}
    total = sum(cleaned.values())
    if total <= 1e-12:
        uniform = 1.0 / float(len(PATH_KEYS))
        return {k: uniform for k in PATH_KEYS}
    return {k: v / total for k, v in cleaned.items()}


def _normalized_entropy(probabilities: list[float]) -> float:
    n = len(probabilities)
    if n <= 1:
        return 0.0
    entropy = 0.0
    for p in probabilities:
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(entropy / math.log(float(n)))


def _candidate_paths_from_trajectory(trajectory_analysis: Mapping[str, object], ranked: list[tuple[str, float]]) -> list[dict[str, object]]:
    candidates = trajectory_analysis.get("candidate_paths")
    if isinstance(candidates, list) and candidates:
        parsed = [c for c in candidates if isinstance(c, Mapping)]
        if parsed:
            return [dict(c) for c in parsed]
    return [
        {
            "path_label": PATH_LABELS.get(label, label.upper()),
            "support": round(float(score), 4),
            "direction_signature": {},
            "dominant_features": [],
            "confidence": round(float(max(0.15, min(1.0, score * 1.2))), 4),
        }
        for label, score in ranked[:3]
    ]


def derive_branching_analysis(
    trajectory_analysis: Mapping[str, object] | None,
    *,
    temporal_quality: Mapping[str, object] | None = None,
    temporal_features: Mapping[str, object] | None = None,
    geometry: Mapping[str, object] | None = None,
    state_graph: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    if not isinstance(trajectory_analysis, Mapping):
        return None
    raw_scores = trajectory_analysis.get("path_scores")
    if not isinstance(raw_scores, Mapping):
        return None

    normalized = _normalize_path_scores(raw_scores)
    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    top_key, top_prob = ranked[0]
    second_key, second_prob = ranked[1]
    margin = max(0.0, float(top_prob - second_prob))
    entropy = _normalized_entropy([score for _, score in ranked])

    closeness = max(0.0, min(1.0, 1.0 - (margin / BRANCH_MARGIN_MAX)))

    diagnostics = trajectory_analysis.get("diagnostics") if isinstance(trajectory_analysis.get("diagnostics"), Mapping) else {}
    directional_div = _safe_score(diagnostics.get("directional_divergence_score", 0.0))
    shape_div = _safe_score(diagnostics.get("shape_divergence_score", 0.0))
    temporal_gap_irregularity = _safe_score((temporal_quality or {}).get("timestamp_gap_irregularity", 0.0))
    temporal_consistency = _safe_score((temporal_quality or {}).get("temporal_consistency_score", 0.0))
    timing_instability = _safe_score((temporal_features or {}).get("timing_instability", temporal_gap_irregularity))
    angular_divergence = _safe_score((geometry or {}).get("angular_change", 0.0)) / 3.141592653589793
    branching_factor = _safe_score((state_graph or {}).get("branching_factor", 0.0))
    branching_intensity = _safe_score((state_graph or {}).get("branching_intensity", 0.0))
    transition_entropy = _safe_score((state_graph or {}).get("transition_entropy", 0.0))
    transition_entropy_effective = _safe_score((state_graph or {}).get("transition_entropy_effective", transition_entropy))
    graph_divergence = _safe_score((state_graph or {}).get("graph_divergence_score", 0.0))
    graph_divergence_effective = _safe_score((state_graph or {}).get("graph_divergence_effective", graph_divergence))
    directional_inconsistency = _safe_score((state_graph or {}).get("directional_inconsistency", 0.0))
    raw_divergence_score = _safe_score((state_graph or {}).get("raw_divergence_score", 0.0))
    revisit_rate = _safe_score((state_graph or {}).get("revisit_rate", 0.0))
    path_commitment_score = _safe_score((state_graph or {}).get("path_commitment_score", 0.0))
    anisotropy = _safe_score((geometry or {}).get("anisotropy", 1.0))
    anisotropy_norm = _clamp01(anisotropy - 1.0)
    curvature_norm = _clamp01(_safe_score((geometry or {}).get("curvature", 0.0) or 0.0) * 8.0)
    counterfactual_spread = _safe_score(trajectory_analysis.get("counterfactual_spread", 0.0))
    if counterfactual_spread <= 1e-9:
        counterfactual_spread = _clamp01(1.0 - margin)

    normalized_branching_signal = max(
        0.0,
        min(
            1.0,
            max(
                branching_intensity,
                min(1.0, (branching_factor - 1.0) / 3.5),
            ),
        ),
    )

    branch_tension = (
        0.28 * closeness
        + 0.16 * entropy
        + 0.14 * directional_div
        + 0.10 * transition_entropy_effective
        + 0.09 * graph_divergence_effective
        + 0.08 * normalized_branching_signal
        + 0.08 * min(1.0, angular_divergence)
        + 0.10 * shape_div
        + 0.07 * timing_instability
        + 0.04 * temporal_gap_irregularity
        + 0.03 * min(1.0, directional_inconsistency)
        + 0.03 * min(1.0, raw_divergence_score)
        + 0.04 * anisotropy_norm
        + 0.04 * counterfactual_spread
        + 0.04 * curvature_norm
    )
    branch_tension = max(0.0, min(1.0, branch_tension))

    is_branching = bool((margin <= BRANCH_MARGIN_MAX and entropy >= BRANCH_ENTROPY_MIN) or branch_tension >= 0.62)

    consistency_penalty = max(0.0, min(0.15, (1.0 - temporal_consistency) * 0.15))
    adjusted_margin = max(0.0, margin - consistency_penalty)
    if adjusted_margin >= COMMITMENT_HIGH_MARGIN and entropy <= COMMITMENT_HIGH_ENTROPY_MAX and branch_tension < 0.5:
        commitment = "HIGH"
    elif adjusted_margin >= COMMITMENT_MODERATE_MARGIN and entropy <= COMMITMENT_MODERATE_ENTROPY_MAX:
        commitment = "MODERATE"
    else:
        commitment = "LOW"

    candidate_paths = _candidate_paths_from_trajectory(trajectory_analysis, ranked)
    supports = [float(c.get("support", 0.0)) for c in candidate_paths if isinstance(c, Mapping)]
    support_total = sum(supports) + 1e-12
    support_probs = [max(0.0, s) / support_total for s in supports] if supports else [top_prob, second_prob]
    branch_entropy = _normalized_entropy(support_probs)
    support_branch_count = len([p for p in support_probs if p >= 0.15])
    branch_expressiveness = _clamp01(
        0.30 * transition_entropy_effective
        + 0.25 * graph_divergence_effective
        + 0.15 * branch_entropy
        + 0.10 * directional_inconsistency
        + 0.10 * (1.0 - path_commitment_score)
        + 0.10 * counterfactual_spread
    )
    branch_response = _clamp01(
        0.45 * normalized_branching_signal
        + 0.25 * branch_expressiveness
        + 0.15 * anisotropy_norm
        + 0.10 * curvature_norm
        + 0.05 * revisit_rate
    )
    graph_branch_count = 1 + int(round(2.0 * branch_response))
    if branch_response >= 0.50 and support_branch_count < 2:
        support_branch_count = 2
    if branch_response >= 0.76 and support_branch_count < 3:
        support_branch_count = 3
    branch_count = max(1, min(len(PATH_KEYS), max(support_branch_count, graph_branch_count)))
    path_diversity = 1.0 - max(support_probs) if support_probs else 0.0
    transition_freq = max(0.0, min(1.0, directional_div * 0.55 + shape_div * 0.45))
    branch_stability = max(0.0, min(1.0, 1.0 - transition_freq))

    dominant_branch = candidate_paths[0]["path_label"] if candidate_paths else PATH_LABELS.get(top_key)
    alternative_branch = candidate_paths[1]["path_label"] if len(candidate_paths) > 1 else PATH_LABELS.get(second_key)

    return {
        "available": True,
        "is_branching": is_branching,
        "branch_count_estimate": max(1, branch_count),
        "candidate_paths": candidate_paths,
        "commitment": commitment,
        "decision_tension": round(float(branch_tension), 4),
        "branch_tension": round(float(branch_tension), 4),
        "dominant_branch": dominant_branch,
        "alternative_branch": alternative_branch,
        "secondary_path": PATH_LABELS.get(second_key),
        "path_entropy": round(float(entropy), 4),
        "branch_entropy": round(float(branch_entropy), 4),
        "dominant_path_observed": PATH_LABELS.get(top_key),
        "top_path_probability": round(float(top_prob), 4),
        "second_path_probability": round(float(second_prob), 4),
        "top_two_margin": round(float(margin), 4),
        "normalized_path_scores": {k: round(float(v), 4) for k, v in normalized.items()},
        "diagnostics": {
            "path_diversity_score": round(float(path_diversity), 4),
            "path_prototype_count": max(1, branch_count),
            "dominant_path_concentration_ratio": round(float(max(support_probs)), 4),
            "directional_divergence_score": round(float(directional_div), 4),
            "branch_stability_over_time": round(float(branch_stability), 4),
            "branch_transition_frequency": round(float(transition_freq), 4),
            "temporal_gap_irregularity": round(float(temporal_gap_irregularity), 4),
            "temporal_consistency_score": round(float(temporal_consistency), 4),
            "timing_instability": round(float(timing_instability), 4),
            "angular_divergence": round(float(angular_divergence), 4),
            "state_graph_branching_factor": round(float(branching_factor), 4),
            "state_graph_branching_intensity": round(float(normalized_branching_signal), 4),
            "state_graph_transition_entropy": round(float(transition_entropy), 4),
            "state_graph_transition_entropy_effective": round(float(transition_entropy_effective), 4),
            "state_graph_graph_divergence_score": round(float(graph_divergence), 4),
            "state_graph_graph_divergence_effective": round(float(graph_divergence_effective), 4),
            "state_graph_directional_inconsistency": round(float(directional_inconsistency), 4),
            "state_graph_raw_divergence_score": round(float(raw_divergence_score), 4),
            "state_graph_revisit_rate": round(float(revisit_rate), 4),
            "state_graph_path_commitment_score": round(float(path_commitment_score), 4),
            "anisotropy_normalized": round(float(anisotropy_norm), 4),
            "counterfactual_spread": round(float(counterfactual_spread), 4),
            "curvature_normalized": round(float(curvature_norm), 4),
            "branch_expressiveness": round(float(branch_expressiveness), 4),
            "branch_response": round(float(branch_response), 4),
        },
        "thresholds": {
            "branch_margin_max": BRANCH_MARGIN_MAX,
            "branch_entropy_min": BRANCH_ENTROPY_MIN,
            "commitment_high_margin": COMMITMENT_HIGH_MARGIN,
            "commitment_moderate_margin": COMMITMENT_MODERATE_MARGIN,
        },
    }
