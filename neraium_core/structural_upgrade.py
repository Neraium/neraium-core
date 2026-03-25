from __future__ import annotations

from typing import Any


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def derive_path_prototype_summary(
    *,
    trajectory: dict[str, Any] | None,
    branching: dict[str, Any] | None,
    constraint: dict[str, Any] | None,
    drift_noise: dict[str, Any] | None,
    multi_scale: dict[str, Any] | None,
) -> dict[str, Any]:
    tr = trajectory or {}
    br = branching or {}
    cs = constraint or {}
    dn = drift_noise or {}
    ms = multi_scale or {}

    trajectory_label = str(tr.get("trajectory_label", "unknown")).lower()
    branch_count = float(br.get("branch_count_estimate", 1.0) or 1.0)
    branch_entropy = float(br.get("branch_entropy", 0.0) or 0.0)
    lock_in = float(cs.get("lock_in_score", 0.0) or 0.0)
    drift_type = str(dn.get("interpreted_change_type", "noise")).lower()
    scale_conflict = float(ms.get("scale_conflict", 0.0) or 0.0)

    scores = {
        "monotonic_drift": 0.0,
        "oscillatory_instability": 0.0,
        "punctuated_shift": 0.0,
        "recovery_then_degradation": 0.0,
        "localized_divergence": 0.0,
        "broad_systemic_drift": 0.0,
        "abrupt_commitment": 0.0,
        "ambiguous_multimodal": 0.0,
    }

    if "osc" in trajectory_label or drift_type == "oscillatory_instability":
        scores["oscillatory_instability"] += 0.7
    if "punct" in trajectory_label or drift_type in {"transient_shift", "persistent_drift"}:
        scores["punctuated_shift"] += 0.35
    if "recover" in trajectory_label:
        scores["recovery_then_degradation"] += 0.65
    if drift_type in {"persistent_drift", "accelerating_drift"}:
        scores["monotonic_drift"] += 0.55
    if branch_count >= 1.8:
        scores["localized_divergence"] += 0.45
    if branch_entropy >= 0.35:
        scores["ambiguous_multimodal"] += 0.35
    if lock_in >= 0.75:
        scores["abrupt_commitment"] += 0.55
    if scale_conflict >= 0.34:
        scores["ambiguous_multimodal"] += 0.4
    if float(dn.get("persistent_change_score", 0.0) or 0.0) >= 0.8 and branch_count < 1.4:
        scores["broad_systemic_drift"] += 0.5

    # Generic priors to avoid all-zero outputs.
    scores["monotonic_drift"] += 0.12
    scores["ambiguous_multimodal"] += 0.08
    scores["localized_divergence"] += 0.08

    total = sum(scores.values()) + 1e-9
    normalized = {k: _clamp01(v / total) for k, v in scores.items()}
    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    dominant, dom_score = ranked[0]

    return {
        "dominant_prototype": dominant,
        "prototype_confidence": round(float(dom_score), 4),
        "candidate_prototypes": [{"label": k, "score": round(float(v), 4)} for k, v in ranked[:4]],
    }


def derive_episode_type(
    *,
    transition_pressure: float,
    drift_type: str,
    lock_in_score: float,
    horizon_category: str,
    branch_count: float,
    multi_scale: dict[str, Any] | None,
) -> tuple[str, str]:
    ms = multi_scale or {}
    short_state = str(ms.get("short_term_state", "unknown"))
    long_state = str(ms.get("long_term_state", "unknown"))
    horizon = str(horizon_category or "unknown").lower()

    if lock_in_score >= 0.85 and horizon in {"imminent", "near"}:
        return "terminal_commitment", "high_lock_in_and_near_horizon"
    if lock_in_score >= 0.65:
        return "branch_commitment", "lock_in_rising"
    if drift_type == "accelerating_drift" and transition_pressure >= 1.0:
        return "re_acceleration", "persistent_acceleration"
    if drift_type in {"persistent_drift", "accelerating_drift"} and transition_pressure >= 0.7:
        return "escalation", "transition_pressure_and_persistent_drift"
    if drift_type == "transient_shift" and short_state in {"recovering", "stable"}:
        return "temporary_recovery", "short_term_recovery_after_transient_shift"
    if long_state == "stable" and transition_pressure < 0.45:
        return "stabilization", "low_transition_pressure_and_long_term_stability"
    if branch_count >= 1.6:
        return "onset", "branching_activity_detected"
    return "onset", "warmup_or_early_transition"


def update_episode_memory(
    *,
    current_step: int,
    previous_episode: dict[str, Any] | None,
    episode_history: list[dict[str, Any]],
    transition_pressure: float,
    drift_type: str,
    lock_in_score: float,
    horizon_category: str,
    branch_count: float,
    multi_scale: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_type, reason = derive_episode_type(
        transition_pressure=transition_pressure,
        drift_type=drift_type,
        lock_in_score=lock_in_score,
        horizon_category=horizon_category,
        branch_count=branch_count,
        multi_scale=multi_scale,
    )

    prev = previous_episode or {
        "current_episode_type": "onset",
        "episode_index": 0,
        "episode_start": 0,
        "episode_duration": 0,
        "episode_transition_reason": "initialization",
    }
    if current_step <= 2:
        episode_type = "onset"
        reason = "safe_warmup"

    if episode_type == prev.get("current_episode_type"):
        current = dict(prev)
        current["episode_duration"] = int(current_step - int(prev.get("episode_start", 0)) + 1)
        current["episode_transition_reason"] = reason
        history = list(episode_history)
    else:
        history = list(episode_history)
        history.append(
            {
                "episode_type": prev.get("current_episode_type", "onset"),
                "episode_index": int(prev.get("episode_index", 0)),
                "episode_start": int(prev.get("episode_start", 0)),
                "episode_end": int(current_step - 1),
                "episode_duration": int(max(1, current_step - int(prev.get("episode_start", 0)))),
                "transition_reason": reason,
            }
        )
        history = history[-24:]
        current = {
            "current_episode_type": episode_type,
            "episode_index": int(prev.get("episode_index", 0)) + 1,
            "episode_start": int(current_step),
            "episode_duration": 1,
            "episode_transition_reason": reason,
        }

    current["episode_history"] = history
    return current, history


def build_evidence_block(
    *,
    frame_count: int,
    temporal_quality: dict[str, Any] | None,
    stability: dict[str, Any] | None,
    attribution: dict[str, Any] | None,
    explanation_count: int,
) -> dict[str, Any]:
    tq = temporal_quality or {}
    st = stability or {}
    at = attribution or {}

    input_completeness = float(tq.get("coverage_ratio", 1.0) or 1.0)
    temporal_score = float(tq.get("temporal_consistency_score", 0.0) or 0.0)
    stability_ref = float(st.get("overall_structural_stability", st.get("overall_stability", 0.0)) or 0.0)
    attribution_depth = len(at.get("top_drivers", [])) if isinstance(at, dict) else 0
    sufficiency = _clamp01(0.25 * input_completeness + 0.25 * temporal_score + 0.35 * stability_ref + 0.15 * min(1.0, attribution_depth / 5.0))

    return {
        "history_depth": int(frame_count),
        "input_completeness": round(input_completeness, 4),
        "temporal_quality": round(temporal_score, 4),
        "stability_reference": round(stability_ref, 4),
        "attribution_reference": {
            "driver_count": int(attribution_depth),
            "top_driver": (at.get("top_drivers", [{}])[0].get("feature") if attribution_depth else None),
        },
        "explanation_summary": {"count": int(explanation_count)},
        "evidence_sufficiency_score": round(float(sufficiency), 4),
    }


def fleet_comparison_summary(entity_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if not entity_results:
        return {"peer_relative_fragility": {}, "rankings": {}, "outlier_entities": [], "comparison_summary": "empty"}

    metrics: dict[str, dict[str, float]] = {}
    for entity, rows in entity_results.items():
        if not rows:
            continue
        last = rows[-1]
        exp = last.get("experimental_analytics", {}) or {}
        stability = (last.get("stability") or (last.get("robustness", {}) or {}).get("stability") or {})
        fragility = 0.4 * float(last.get("latest_drift", 0.0) or 0.0) + 0.3 * float((exp.get("constraint_analysis", {}) or {}).get("lock_in_score", 0.0) or 0.0) + 0.3 * (1.0 - float(stability.get("overall_structural_stability", stability.get("overall_stability", 0.0)) or 0.0))
        metrics[entity] = {
            "fragility": float(fragility),
            "branch_ambiguity": float((exp.get("branching_analysis", {}) or {}).get("branch_entropy", 0.0) or 0.0),
            "recoverability": float((exp.get("trajectory_analysis", {}) or {}).get("recovery_signature", 0.0) or 0.0),
            "counterfactual_shift": float((exp.get("counterfactual_simulation", {}) or {}).get("scenario_spread", 0.0) or 0.0),
        }

    if not metrics:
        return {"peer_relative_fragility": {}, "rankings": {}, "outlier_entities": [], "comparison_summary": "empty"}

    frag_vals = [m["fragility"] for m in metrics.values()]
    mu = sum(frag_vals) / len(frag_vals)
    sd = (sum((x - mu) ** 2 for x in frag_vals) / max(1, len(frag_vals))) ** 0.5
    z = {k: ((v["fragility"] - mu) / (sd + 1e-9)) for k, v in metrics.items()}

    def rank(key: str, reverse: bool = True) -> list[str]:
        return [k for k, _v in sorted(metrics.items(), key=lambda kv: kv[1][key], reverse=reverse)]

    outliers = [k for k, zv in z.items() if abs(zv) >= 1.0]
    summary = {
        "earliest_diverger": rank("fragility", True)[0],
        "highest_branch_ambiguity": rank("branch_ambiguity", True)[0],
        "most_recoverable_pattern": rank("recoverability", True)[0],
        "strongest_counterfactual_shift": rank("counterfactual_shift", True)[0],
    }

    return {
        "peer_relative_fragility": {k: round(float(v), 4) for k, v in z.items()},
        "rankings": summary,
        "outlier_entities": outliers,
        "comparison_summary": f"entities={len(metrics)} median_fragility={round(sorted(frag_vals)[len(frag_vals)//2], 4)}",
    }
