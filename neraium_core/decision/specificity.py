"""Extract specific changes from SII output.

Instead of generic "drift +0.34", explain WHAT changed and WHY it matters.
"""

from __future__ import annotations

from typing import Any
from neraium_core.decision.models import Finding


def extract_findings(
    *,
    sii_output: dict[str, Any],
    attribution: dict[str, Any] | None = None,
    prev_state: dict[str, Any] | None = None,
) -> list[Finding]:
    """Extract specific, actionable findings from SII output.

    Categories:
    - correlation_loss: relationships between signals broke
    - subsystem_instability: part of the system became unstable
    - signal_degradation: individual signal quality declined
    - regime_shift: operating regime changed
    - coordination_failure: signals stopped coordinating
    """
    findings: list[Finding] = []

    state = sii_output.get("state", "STABLE")
    drift = float(sii_output.get("structural_drift_score", 0.0))
    relational = float(sii_output.get("relational_instability_score", 0.0))
    phase = sii_output.get("system_phase", "stable")
    regime = sii_output.get("regime_name")

    top_drivers = []
    if isinstance(attribution, dict):
        drivers = attribution.get("top_drivers", [])
        if isinstance(drivers, list):
            top_drivers = drivers[:3]

    if drift > 0.5 and state in {"ALERT", "WATCH"}:
        magnitude = min(1.0, (drift - 0.5) / 0.5)
        findings.append(Finding(
            category="structural_drift",
            description=f"Structural alignment degraded (score {drift:.2f})",
            confidence=min(0.95, 0.4 + drift * 0.5),
            magnitude=magnitude,
            reversible=drift < 0.8,
            affected_signals=top_drivers,
        ))

    if relational > 0.4:
        magnitude = min(1.0, relational / 0.8)
        findings.append(Finding(
            category="coordination_failure",
            description=f"Signal relationships became unstable (instability {relational:.2f})",
            confidence=min(0.9, 0.5 + relational * 0.4),
            magnitude=magnitude,
            reversible=relational < 0.6,
            affected_signals=top_drivers,
        ))

    if phase == "degrading" and prev_state:
        prev_drift = float(prev_state.get("structural_drift_score", 0.0))
        if drift > prev_drift + 0.1:
            findings.append(Finding(
                category="trend_deterioration",
                description=f"System deteriorating: drift increased {drift - prev_drift:.2f} this frame",
                confidence=0.7,
                magnitude=min(1.0, (drift - prev_drift) * 5),
                reversible=False,
                affected_signals=top_drivers,
            ))

    if regime and isinstance(sii_output.get("regime_distance"), (int, float)):
        regime_dist = float(sii_output.get("regime_distance", 0.5))
        if regime_dist > 0.7:
            findings.append(Finding(
                category="regime_shift",
                description=f"Operating regime shifted: {regime} (distance {regime_dist:.2f})",
                confidence=0.75,
                magnitude=regime_dist,
                reversible=True,
                affected_signals=[],
            ))

    if top_drivers:
        finding_text = f"Primary signal changes: {', '.join(top_drivers[:2])}"
        if len(top_drivers) > 2:
            finding_text += f" + {len(top_drivers) - 2} others"

        findings.append(Finding(
            category="signal_focus",
            description=finding_text,
            confidence=0.8,
            magnitude=0.6,
            affected_signals=top_drivers,
        ))

    return findings


def compute_delta_summary(
    *,
    current: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> str:
    """Build a human-readable summary of what changed."""
    if not previous:
        return "Baseline established; no prior state to compare."

    prev_state = previous.get("state", "UNKNOWN")
    curr_state = current.get("state", "STABLE")

    if prev_state != curr_state:
        return f"State changed from {prev_state} to {curr_state}."

    prev_drift = float(previous.get("structural_drift_score", 0.0))
    curr_drift = float(current.get("structural_drift_score", 0.0))

    if abs(curr_drift - prev_drift) > 0.05:
        direction = "increased" if curr_drift > prev_drift else "decreased"
        return f"Drift {direction} from {prev_drift:.2f} to {curr_drift:.2f}."

    return "Marginal changes; system relatively stable."
