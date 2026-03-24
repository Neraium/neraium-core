from __future__ import annotations

import math
from typing import Mapping


PATH_KEYS: tuple[str, ...] = ("stabilizing", "metastable", "diverging")
PATH_LABELS: dict[str, str] = {
    "stabilizing": "STABILIZING",
    "metastable": "METASTABLE",
    "diverging": "DIVERGING",
}

# Simple thresholds intended for easy tuning as trajectory analysis matures.
BRANCH_MARGIN_MAX = 0.15
BRANCH_ENTROPY_MIN = 0.85
COMMITMENT_HIGH_MARGIN = 0.40
COMMITMENT_HIGH_ENTROPY_MAX = 0.70
COMMITMENT_MODERATE_MARGIN = 0.22
COMMITMENT_MODERATE_ENTROPY_MAX = 0.90


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
        # If incoming scores are unusable, default to a fully ambiguous split.
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


def derive_branching_analysis(trajectory_analysis: Mapping[str, object] | None) -> dict[str, object] | None:
    """
    Derive lightweight possibility-space observables from trajectory path scores.

    Expected shape from trajectory-analysis branch:
      {
        "dominant_path": "STABILIZING|METASTABLE|DIVERGING",
        "path_scores": {
          "stabilizing": float,
          "metastable": float,
          "diverging": float,
        }
      }
    """
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

    # Branch tension increases when the top two paths are close and overall spread is broad.
    closeness = max(0.0, min(1.0, 1.0 - (margin / BRANCH_MARGIN_MAX)))
    branch_tension = 0.6 * closeness + 0.4 * entropy
    branch_tension = max(0.0, min(1.0, branch_tension))

    is_branching = bool(margin <= BRANCH_MARGIN_MAX and entropy >= BRANCH_ENTROPY_MIN)

    if margin >= COMMITMENT_HIGH_MARGIN and entropy <= COMMITMENT_HIGH_ENTROPY_MAX:
        commitment = "HIGH"
    elif margin >= COMMITMENT_MODERATE_MARGIN and entropy <= COMMITMENT_MODERATE_ENTROPY_MAX:
        commitment = "MODERATE"
    else:
        commitment = "LOW"

    return {
        "is_branching": is_branching,
        "secondary_path": PATH_LABELS.get(second_key),
        "branch_tension": round(float(branch_tension), 4),
        "commitment": commitment,
        "path_entropy": round(float(entropy), 4),
        "dominant_path_observed": PATH_LABELS.get(top_key),
        "top_path_probability": round(float(top_prob), 4),
        "second_path_probability": round(float(second_prob), 4),
        "top_two_margin": round(float(margin), 4),
        "normalized_path_scores": {k: round(float(v), 4) for k, v in normalized.items()},
        "thresholds": {
            "branch_margin_max": BRANCH_MARGIN_MAX,
            "branch_entropy_min": BRANCH_ENTROPY_MIN,
            "commitment_high_margin": COMMITMENT_HIGH_MARGIN,
            "commitment_moderate_margin": COMMITMENT_MODERATE_MARGIN,
        },
    }
