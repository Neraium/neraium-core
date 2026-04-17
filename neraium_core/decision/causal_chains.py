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
    """Build a simple first-pass causal chain from observable signals.

    Follows: root cause → intermediate effects → current state.
    """
    steps: list[CausalStep] = []
    root_cause = None
    confidence = 0.5

    state = sii_output.get("state", "STABLE")
    phase = sii_output.get("system_phase", "stable")
    regime_name = sii_output.get("regime_name")
    drift = float(sii_output.get("structural_drift_score", 0.0))
    relational = float(sii_output.get("relational_instability_score", 0.0))

    attribution = sii_output.get("attribution", {})
    top_drivers = []
    if isinstance(attribution, dict):
        drivers = attribution.get("top_drivers", [])
        if isinstance(drivers, list):
            top_drivers = drivers[:3]

    if shock_activity > 0.6:
        root_cause = "external_shock"
        steps.append(CausalStep(
            trigger="External shock detected",
            effect="Signal relationships disrupted",
            strength=min(1.0, shock_activity * 0.8),
            involved_signals=top_drivers[:2],
        ))
        confidence = 0.75

    if drift > 0.4 and not root_cause:
        root_cause = "structural_misalignment"
        steps.append(CausalStep(
            trigger="Baseline-to-recent structural misalignment",
            effect="Correlation matrices diverged",
            strength=min(1.0, drift * 0.6),
            involved_signals=top_drivers[:3],
        ))
        confidence = 0.7

    if relational > 0.3 and len(steps) > 0:
        steps.append(CausalStep(
            trigger="Correlation breakdown",
            effect="Relational instability metrics elevated",
            strength=min(1.0, relational * 0.7),
            involved_signals=top_drivers,
        ))
        confidence = min(1.0, confidence + 0.1)

    if phase == "degrading" and len(steps) > 0:
        steps.append(CausalStep(
            trigger="Sustained structural misalignment",
            effect="System phase transitioned to degrading",
            strength=0.8,
            involved_signals=top_drivers,
        ))
        confidence = min(1.0, confidence + 0.15)

    if state in {"ALERT", "WATCH"} and len(steps) > 0:
        steps.append(CausalStep(
            trigger="Thresholds exceeded on multiple signals",
            effect=f"Policy state changed to {state}",
            strength=0.85,
            involved_signals=top_drivers,
        ))
        confidence = min(1.0, confidence + 0.1)

    if not steps:
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
