"""Trust and actionability signals for operator-facing outputs."""

from __future__ import annotations

from typing import Any

from neraium_core.product._util import clamp01, safe_float, safe_get


def _readiness_status(ready: bool, structural: bool, reason: str) -> str:
    if not structural:
        return "warming_up"
    if not ready:
        return "partial"
    if reason == "partial_subsystems_pending":
        return "partial"
    return "ready"


def _telemetry_quality(confidence: float, readiness_reason: str) -> str:
    if readiness_reason == "awaiting_baseline_and_recent_windows":
        return "limited"
    c = clamp01(confidence)
    if c >= 0.72:
        return "good"
    if c >= 0.45:
        return "fair"
    return "limited"


def _evidence_persistence(result: dict[str, Any]) -> float:
    """Proxy: stability of transition pressure signal from score spread (higher is more persistent)."""
    experimental = safe_get(result, "experimental_analytics") or {}
    horizon = experimental.get("horizon_analysis") or {}
    rat = (horizon or {}).get("rationale") or {}
    if isinstance(rat, dict):
        ew_count = safe_float(rat.get("early_warning_emerging_count", 0))
        ew_persist = clamp01(ew_count / 8.0)
    else:
        ew_persist = 0.0
    pressure = clamp01(safe_float(result.get("transition_pressure")))
    return clamp01(0.55 * pressure + 0.45 * ew_persist)


def build_trust_layer(result: dict[str, Any]) -> dict[str, Any]:
    readiness = safe_get(result, "readiness") or {}
    ready = bool(readiness.get("ready", False))
    structural = bool(readiness.get("structural_ready", False))
    transition_ready = bool(result.get("transition_outputs_actionable", False))
    reason = str(readiness.get("reason", ""))

    confidence = safe_float(result.get("confidence_score", 0.5))

    limiting: list[str] = []
    uc = readiness.get("unavailable_components") or {}
    if isinstance(uc, dict):
        for name, why in uc.items():
            limiting.append(f"{name}:{why}")

    if reason == "stabilization_period_incomplete":
        limiting.append("stabilization_period_incomplete")
    if reason == "insufficient_transition_pressure_history":
        limiting.append("insufficient_transition_pressure_history")

    actionability_reason = (
        "Full structural windows and transition history support classification."
        if transition_ready
        else "Transition classification is limited until stabilization and pressure history thresholds are met."
    )

    confidence_reason = (
        f"Confidence reflects relational stability and baseline strength (score {clamp01(confidence):.2f})."
    )

    safe_to_act = bool(ready and structural and transition_ready and len(limiting) == 0)

    trust = clamp01(
        0.28 * (1.0 if ready else 0.35)
        + 0.22 * (1.0 if structural else 0.4)
        + 0.22 * (1.0 if transition_ready else 0.45)
        + 0.18 * clamp01(confidence)
        + 0.10 * (1.0 - min(1.0, len(limiting) * 0.15))
    )

    return {
        "readiness_status": _readiness_status(ready, structural, reason),
        "evidence_persistence": round(_evidence_persistence(result), 4),
        "telemetry_quality_status": _telemetry_quality(confidence, reason),
        "actionability_reason": actionability_reason,
        "confidence_reason": confidence_reason,
        "limiting_factors": limiting[:12],
        "safe_to_act": safe_to_act,
        "trust_score": round(trust, 4),
    }
