"""Build simple causal chains from SII outputs.

Explains how we got from baseline to current state via observable cause-effect links.
"""

from __future__ import annotations

from typing import Any
from neraium_core.decision.models import CausalChain, CausalStep


def build_causal_chain(
    *,
    sii_output: dict[str, Any],
    shock_activity: float = 0.0,
    subsystem_instability: float = 0.0,
) -> CausalChain | None:
    """Build causal chain from observable signals.

    Follows: root cause → intermediate effects → current state.
    Prioritizes interpretable multi-signal patterns over single-signal explanations.
    Avoids weak or repetitive causal links.
    """
    steps: list[CausalStep] = []
    root_cause = None
    confidence = 0.5

    state = sii_output.get("state", "STABLE")
    phase = sii_output.get("system_phase", "stable")
    drift = float(sii_output.get("structural_drift_score", 0.0))
    relational = float(sii_output.get("relational_instability_score", 0.0))
    entropy = float(sii_output.get("entropy", 0.0))

    attribution = sii_output.get("attribution", {})
    top_drivers = []
    if isinstance(attribution, dict):
        drivers = attribution.get("top_drivers", [])
        if isinstance(drivers, list):
            top_drivers = drivers[:3]

    # Identify root cause (single, interpretable source)
    if shock_activity > 0.75:
        root_cause = "external_shock_or_sudden_perturbation"
        steps.append(CausalStep(
            trigger="High shock activity detected",
            effect="Disrupted signal coordination",
            strength=min(1.0, shock_activity * 0.9),
            involved_signals=top_drivers[:2],
        ))
        confidence = 0.8

    elif subsystem_instability > 0.6:
        root_cause = "subsystem_degradation"
        steps.append(CausalStep(
            trigger="Subsystem instability spike",
            effect="Structural alignment degraded",
            strength=min(1.0, subsystem_instability * 0.8),
            involved_signals=top_drivers[:2],
        ))
        confidence = 0.75

    elif drift > 0.65 and relational > 0.5:
        # Multi-indicator: both structural and relational breakdown
        root_cause = "persistent_structural_degradation"
        steps.append(CausalStep(
            trigger="Sustained baseline-to-recent misalignment",
            effect="Correlation matrices diverged; relationships unstable",
            strength=min(1.0, (drift + relational) / 2.0 * 0.8),
            involved_signals=top_drivers,
        ))
        confidence = 0.8

    elif drift > 0.5:
        root_cause = "structural_alignment_shift"
        steps.append(CausalStep(
            trigger="Baseline-to-recent structural misalignment",
            effect="Correlation patterns changed",
            strength=min(1.0, drift * 0.7),
            involved_signals=top_drivers[:2],
        ))
        confidence = 0.7

    elif relational > 0.5:
        root_cause = "relationship_breakdown"
        steps.append(CausalStep(
            trigger="Signal coupling/correlation weakened",
            effect="Relational instability metrics elevated",
            strength=min(1.0, relational * 0.8),
            involved_signals=top_drivers,
        ))
        confidence = 0.72

    # Intermediate effects (only if root cause identified and evidence strong)
    if root_cause and len(steps) > 0:
        # Add intermediate only if moving from root cause to policy state
        if entropy > 0.7 and relational > 0.3:
            steps.append(CausalStep(
                trigger="Entropy increased; signals became less predictable",
                effect="System coordination degraded",
                strength=min(1.0, entropy * 0.7),
                involved_signals=top_drivers[:2] if top_drivers else [],
            ))
            confidence = min(1.0, confidence + 0.08)

    # Policy state (only if clear progression)
    if root_cause and state in {"ALERT", "WATCH"} and len(steps) >= 1:
        effect_text = (
            "Policy thresholds exceeded on structural metrics"
            if drift > 0.5
            else "Policy thresholds exceeded"
        )
        steps.append(CausalStep(
            trigger=f"Combined indicators triggered {state} state",
            effect=effect_text,
            strength=0.85,
            involved_signals=top_drivers[:1] if top_drivers else [],
        ))
        confidence = min(1.0, confidence + 0.05)

    if not steps or not root_cause:
        return None

    return CausalChain(
        steps=steps,
        root_cause=root_cause,
        confidence=confidence,
    )


def chain_strength(chain: CausalChain | None) -> float:
    """Score the strength/confidence of a causal chain."""
    if not chain or not chain.steps:
        return 0.0

    avg_step_strength = sum(s.strength for s in chain.steps) / len(chain.steps)
    return avg_step_strength * chain.confidence
