"""Decision coherence validation and enforcement.

Ensures all Decision fields are logically consistent, with no contradictions
between severity, trajectory, stage, actions, and suppression state.
"""

from __future__ import annotations

from typing import Any
from neraium_core.decision.models import Decision, SeverityLevel, DegradationStage


def _severity_rank(sev: SeverityLevel | str) -> int:
    """Map severity to numeric rank for comparison."""
    rank_map = {"LOW": 0, "MODERATE": 1, "ELEVATED": 2, "HIGH": 3}
    return rank_map.get(str(sev), -1)


def _stage_severity_baseline(stage: DegradationStage | str) -> SeverityLevel:
    """Minimum severity expected for each degradation stage."""
    stage_map: dict[str, SeverityLevel] = {
        "baseline": "LOW",
        "early_shift": "MODERATE",
        "emerging_degradation": "MODERATE",
        "persistent_degradation": "ELEVATED",
        "accelerated_deterioration": "ELEVATED",
        "chronic_degraded_state": "ELEVATED",
        "failure_approach": "HIGH",
    }
    return stage_map.get(str(stage), "LOW")


class CoherenceCheck:
    """Single coherence validation rule."""

    def __init__(self, name: str, condition_fn, error_msg_fn):
        self.name = name
        self.condition_fn = condition_fn
        self.error_msg_fn = error_msg_fn

    def check(self, decision: Decision) -> tuple[bool, str | None]:
        """Run check; return (passed, error_message)."""
        if not self.condition_fn(decision):
            return False, self.error_msg_fn(decision)
        return True, None


def _build_coherence_checks() -> list[CoherenceCheck]:
    """Build all coherence validation rules."""
    checks = []

    # Check 1: Suppress + Recommendation consistency
    checks.append(
        CoherenceCheck(
            name="suppress_recommendation_consistency",
            condition_fn=lambda d: not (d.suppress and d.primary_action is not None),
            error_msg_fn=lambda d: "suppress=True but primary_action is set",
        )
    )

    # Check 2: HIGH severity must be actionable
    checks.append(
        CoherenceCheck(
            name="high_severity_horizon",
            condition_fn=lambda d: (
                d.severity != "HIGH"
                or d.action_horizon in {"now", "soon"}
            ),
            error_msg_fn=lambda d: f"HIGH severity must have horizon 'now'|'soon', got {d.action_horizon}",
        )
    )

    # Check 3: HIGH severity not suppressed
    checks.append(
        CoherenceCheck(
            name="high_severity_not_suppressed",
            condition_fn=lambda d: d.severity != "HIGH" or not d.suppress,
            error_msg_fn=lambda d: "HIGH severity must not be suppressed",
        )
    )

    # Check 4: Stage implies minimum severity
    checks.append(
        CoherenceCheck(
            name="stage_severity_alignment",
            condition_fn=lambda d: _severity_rank(d.severity) >= _severity_rank(
                _stage_severity_baseline(d.degradation_stage)
            ),
            error_msg_fn=lambda d: (
                f"Stage {d.degradation_stage} implies min severity "
                f"{_stage_severity_baseline(d.degradation_stage)}, got {d.severity}"
            ),
        )
    )

    # Check 5: Action horizon severity alignment
    checks.append(
        CoherenceCheck(
            name="action_horizon_severity",
            condition_fn=lambda d: (
                d.action_horizon is None
                or d.action_horizon == "now"
                and _severity_rank(d.severity) >= _severity_rank("ELEVATED")
                or d.action_horizon == "soon"
                and _severity_rank(d.severity) >= _severity_rank("MODERATE")
                or d.action_horizon == "watchlist"
            ),
            error_msg_fn=lambda d: (
                f"Horizon {d.action_horizon} misaligned with severity {d.severity}"
            ),
        )
    )

    # Check 6: Failure approach stage is critical
    checks.append(
        CoherenceCheck(
            name="failure_approach_critical",
            condition_fn=lambda d: (
                d.degradation_stage != "failure_approach"
                or d.severity == "HIGH"
                or d.action_horizon == "now"
            ),
            error_msg_fn=lambda d: "failure_approach stage must be HIGH severity or 'now' horizon",
        )
    )

    # Check 7: Confidence coherence - low finding confidence with action
    checks.append(
        CoherenceCheck(
            name="finding_action_confidence",
            condition_fn=lambda d: (
                d.finding_confidence >= 0.3
                or d.action_confidence < 0.6
                or d.suppress
            ),
            error_msg_fn=lambda d: (
                f"Low finding_confidence ({d.finding_confidence:.2f}) but "
                f"high action_confidence ({d.action_confidence:.2f})"
            ),
        )
    )

    # Check 8: Persistence frames non-negative
    checks.append(
        CoherenceCheck(
            name="persistence_frames_valid",
            condition_fn=lambda d: d.persistence_frames_at_level >= 0,
            error_msg_fn=lambda d: f"persistence_frames_at_level cannot be negative: {d.persistence_frames_at_level}",
        )
    )

    # Check 9: Transient score is [0, 1]
    checks.append(
        CoherenceCheck(
            name="transient_score_bounds",
            condition_fn=lambda d: 0.0 <= d.transient_score <= 1.0,
            error_msg_fn=lambda d: f"transient_score out of bounds: {d.transient_score}",
        )
    )

    # Check 10: Confidence scores are [0, 1]
    checks.append(
        CoherenceCheck(
            name="confidence_bounds",
            condition_fn=lambda d: (
                (0.0 <= d.finding_confidence <= 1.0)
                and (0.0 <= d.action_confidence <= 1.0)
            ),
            error_msg_fn=lambda d: (
                f"Confidence out of bounds: finding={d.finding_confidence}, action={d.action_confidence}"
            ),
        )
    )

    # Check 11: Primary action consistency with suppress
    checks.append(
        CoherenceCheck(
            name="primary_action_suppress",
            condition_fn=lambda d: (
                not d.suppress or d.primary_action is None
            ),
            error_msg_fn=lambda d: (
                "When suppressed, primary_action should not be set"
            ),
        )
    )

    # Check 12: Degrading trajectory should not have LOW severity alone
    checks.append(
        CoherenceCheck(
            name="trajectory_severity_coherence",
            condition_fn=lambda d: (
                d.trajectory != "degrading"
                or d.severity != "LOW"
            ),
            error_msg_fn=lambda d: "degrading trajectory should not have LOW severity",
        )
    )

    return checks


def validate_decision_coherence(decision: Decision) -> tuple[bool, list[str]]:
    """Validate internal consistency of a Decision object.

    Args:
        decision: Decision to validate

    Returns:
        (is_valid, error_messages)
    """
    checks = _build_coherence_checks()
    errors = []

    for check in checks:
        passed, error_msg = check.check(decision)
        if not passed:
            errors.append(error_msg)

    return len(errors) == 0, errors


def enforce_decision_coherence(decision: Decision) -> tuple[Decision, list[str]]:
    """Validate and apply corrections to ensure coherence.

    This function corrects obvious inconsistencies without changing the
    fundamental decision (severity, stage, trajectory).

    Args:
        decision: Decision to enforce coherence on

    Returns:
        (corrected_decision, corrections_applied)
    """
    corrections = []

    # Correction 1: If suppress=True, clear primary_action
    if decision.suppress and decision.primary_action is not None:
        decision.primary_action = None
        decision.secondary_actions = []
        corrections.append("Cleared actions due to suppress=True")

    # Correction 2: If HIGH severity, ensure not suppressed
    if decision.severity == "HIGH":
        if decision.suppress:
            decision.suppress = False
            corrections.append("Cleared suppress flag for HIGH severity")

    # Correction 3: If HIGH severity, ensure horizon is now/soon
    if decision.severity == "HIGH":
        if decision.action_horizon not in {"now", "soon"}:
            decision.action_horizon = "now"
            corrections.append("Set horizon to 'now' for HIGH severity")

    # Correction 4: Ensure action horizon aligns with severity
    if decision.action_horizon == "now":
        if _severity_rank(decision.severity) < _severity_rank("ELEVATED"):
            decision.action_horizon = "soon"
            corrections.append(f"Downgraded horizon from 'now' to 'soon' for {decision.severity} severity")

    return decision, corrections


def format_coherence_report(decision: Decision) -> str:
    """Generate a human-readable coherence report."""
    is_valid, errors = validate_decision_coherence(decision)

    if is_valid:
        return "✓ Decision is coherent"

    report = "✗ Decision coherence issues:\n"
    for i, error in enumerate(errors, 1):
        report += f"  {i}. {error}\n"
    return report.rstrip()
