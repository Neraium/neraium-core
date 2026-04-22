"""Extract specific changes from SII output.

Instead of generic "drift +0.34", explain WHAT changed and WHY it matters.
Prioritize persistent relationship changes, subsystem instability, and regime transitions.
"""

from __future__ import annotations

from typing import Any
from neraium_core.decision.models import Finding
from neraium_core.decision import change_detection


def extract_findings(
    *,
    sii_output: dict[str, Any],
    attribution: dict[str, Any] | None = None,
    prev_state: dict[str, Any] | None = None,
) -> list[Finding]:
    """Extract specific, actionable findings from SII output.

    Categories (prioritized):
    - persistent_relationship_change: correlation/coupling broke and stayed broken
    - subsystem_instability: observable subsystem became unstable
    - regime_shift: operating regime changed significantly
    - coordination_failure: signals stopped coordinating
    - structural_drift: baseline to recent misalignment
    """
    findings: list[Finding] = []

    state = sii_output.get("state", "STABLE")
    drift = float(sii_output.get("structural_drift_score", 0.0))
    relational = float(sii_output.get("relational_instability_score", 0.0))
    phase = sii_output.get("system_phase", "stable")
    regime = sii_output.get("regime_name")
    subsystem_instability = float(sii_output.get("subsystem_instability", 0.0))

    top_drivers = []
    if isinstance(attribution, dict):
        drivers = attribution.get("top_drivers", [])
        if isinstance(drivers, list):
            top_drivers = drivers[:3]

    # Analyze deltas to understand what changed
    deltas = _compute_key_deltas(sii_output, prev_state) if prev_state else []

    # Priority 1: Persistent relationship changes
    if relational > 0.45 and state in {"ALERT", "WATCH"}:
        magnitude = min(1.0, relational / 0.8)
        primary = top_drivers[0] if top_drivers else "key signals"
        findings.append(Finding(
            category="persistent_relationship_change",
            description=f"Signal relationships became unstable (instability {relational:.2f}); {primary} decoupled",
            confidence=min(0.9, 0.5 + relational * 0.4),
            magnitude=magnitude,
            reversible=relational < 0.6,
            affected_signals=top_drivers,
        ))

    # Priority 2: Subsystem instability
    if subsystem_instability > 0.5:
        findings.append(Finding(
            category="subsystem_instability",
            description=f"Observable subsystem became unstable (score {subsystem_instability:.2f})",
            confidence=min(0.85, 0.6 + subsystem_instability * 0.3),
            magnitude=min(1.0, subsystem_instability / 0.8),
            reversible=subsystem_instability < 0.7,
            affected_signals=top_drivers,
        ))

    # Priority 3: Regime shift
    if regime and isinstance(sii_output.get("regime_distance"), (int, float)):
        regime_dist = float(sii_output.get("regime_distance", 0.5))
        if regime_dist > 0.7:
            findings.append(Finding(
                category="regime_shift",
                description=f"Operating regime shifted to {regime} (distance {regime_dist:.2f})",
                confidence=0.75,
                magnitude=regime_dist,
                reversible=True,
                affected_signals=top_drivers[:1] if top_drivers else [],
            ))

    # Priority 4: Structural drift with delta context
    if drift > 0.5 and state in {"ALERT", "WATCH"}:
        magnitude = min(1.0, (drift - 0.5) / 0.5)
        prev_drift = float(prev_state.get("structural_drift_score", 0.0)) if prev_state else drift
        drift_delta = drift - prev_drift
        primary_driver = _find_primary_driver(deltas) if deltas else "structural factors"

        if drift_delta > 0.1:
            desc = f"Structural alignment degraded ({prev_drift:.2f} → {drift:.2f}, +{drift_delta:.2f}); driven by {primary_driver}"
        else:
            desc = f"Structural alignment degraded (score {drift:.2f})"

        findings.append(Finding(
            category="structural_drift",
            description=desc,
            confidence=min(0.95, 0.4 + drift * 0.5),
            magnitude=magnitude,
            reversible=drift < 0.8,
            affected_signals=top_drivers,
        ))

    # Trend deterioration: check persistence of degradation
    if phase == "degrading" and prev_state:
        prev_drift = float(prev_state.get("structural_drift_score", 0.0))
        if drift > prev_drift + 0.08:  # Sustained increase
            findings.append(Finding(
                category="trend_deterioration",
                description=f"System deteriorating: drift increased {drift - prev_drift:.2f} sustained",
                confidence=0.75,
                magnitude=min(1.0, (drift - prev_drift) * 5),
                reversible=False,
                affected_signals=top_drivers,
            ))

    # Only surface key signals if no other specific findings
    if top_drivers and not findings:
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


def _compute_key_deltas(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Compute deltas for key metrics between frames."""
    if not previous:
        return []

    keys = ["structural_drift_score", "relational_instability_score", "subsystem_instability"]
    thresholds = {
        "structural_drift_score": 0.05,
        "relational_instability_score": 0.04,
        "subsystem_instability": 0.06,
    }

    current_floats = {k: float(current.get(k, 0.0)) for k in keys}
    previous_floats = {k: float(previous.get(k, 0.0)) for k in keys}

    return change_detection.detect_metric_deltas(
        current=current_floats,
        previous=previous_floats,
        keys=keys,
        thresholds=thresholds,
    )


def _find_primary_driver(deltas: list[dict[str, Any]]) -> str:
    """Identify primary driver from deltas."""
    if not deltas:
        return "unknown factors"

    primary = deltas[0]["metric"]
    if "relational" in primary:
        return "relationship breakdown"
    if "subsystem" in primary:
        return "subsystem instability"
    if "drift" in primary:
        return "structural misalignment"

    return primary.replace("_", " ")


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
