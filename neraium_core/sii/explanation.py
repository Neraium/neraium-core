from __future__ import annotations

from typing import Mapping


def dominant_drivers(components: Mapping[str, float], top_k: int = 3) -> list[str]:
    ranked = sorted(components.items(), key=lambda kv: float(kv[1]), reverse=True)
    return [name for name, _ in ranked[: max(1, int(top_k))]]


_STATE_DESCRIPTION: dict[str, str] = {
    "STRUCTURAL_INSTABILITY_OBSERVED": "Propagation operator has shifted significantly; spectral radius approaching instability boundary.",
    "COUPLING_INSTABILITY_OBSERVED": "Directed coupling structure deviating from baseline; subsystem interactions destabilizing.",
    "REGIME_SHIFT_OBSERVED": "System operator signature has moved away from baseline regime; structural distance elevated.",
    "COHERENCE_UNDER_CONSTRAINT": "Residual coherence below baseline; constraint structure partially dissolved.",
    "NOMINAL_STRUCTURE": "Structural operator consistent with baseline; no significant drift detected.",
}

_DECISION_SUFFIX: dict[str, str] = {
    "ALERT": "Immediate inspection indicated.",
    "WATCH": "Elevated monitoring warranted.",
    "STABLE": "No action indicated.",
}


def build_explanation_text(
    *,
    interpreted_state: str,
    decision_state: str,
    dominant: list[str],
    confidence_reasoning: list[str],
) -> str:
    state_desc = _STATE_DESCRIPTION.get(interpreted_state, f"State: {interpreted_state}.")
    decision_suffix = _DECISION_SUFFIX.get(decision_state, "")
    driver_text = ", ".join(dominant) if dominant else "none identified"
    reason_text = ", ".join(confidence_reasoning) if confidence_reasoning else "none"
    parts = [f"{state_desc}"]
    if decision_suffix:
        parts.append(decision_suffix)
    parts.append(f"Dominant drivers: {driver_text}.")
    parts.append(f"Confidence evidence: {reason_text}.")
    return " ".join(parts)


# Backward-compatible alias
build_explanation = build_explanation_text

