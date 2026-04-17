"""Pattern outcome influence: Use historical outcomes to modulate decision logic.

This module implements Phase 5 outcome-aware decision intelligence.
Pattern memory does NOT override core detection, but modulates decision logic.
"""

from __future__ import annotations

from typing import Optional, Any
from neraium_core.decision.models import PatternMatch, Decision, SeverityLevel


class PatternOutcomeInfluencer:
    """Modulate decision parameters based on historical pattern outcomes."""

    def adjust_suppression_for_self_resolving(
        self,
        pattern_match: Optional[PatternMatch],
        severity: SeverityLevel,
        transient_score: float,
    ) -> bool:
        """Adjust suppression decision for likely self-resolving patterns.

        If pattern is strong match to self_resolved pattern and live evidence
        is ambiguous (moderate transient_score, LOW/MODERATE severity), lean
        toward suppression. Never suppress HIGH/ELEVATED.

        Args:
            pattern_match: Historical pattern match
            severity: Current severity level
            transient_score: Likelihood this is transient [0, 1]

        Returns:
            True if should suppress based on pattern history
        """
        # Never suppress HIGH/ELEVATED regardless of pattern
        if severity in {"HIGH", "ELEVATED"}:
            return False

        if not pattern_match:
            return False

        # Only strong/moderate matches influence suppression
        if pattern_match.match_tier not in {"strong", "moderate"}:
            return False

        # Only self_resolved patterns suggest suppression
        if pattern_match.outcome_type != "self_resolved":
            return False

        # If pattern is strong match to self-resolved, suppress moderate/low severity
        if pattern_match.match_tier == "strong" and severity in {"LOW", "MODERATE"}:
            return True

        # If transient_score is high AND pattern is moderate match to self-resolved
        if transient_score > 0.6 and pattern_match.match_tier == "moderate":
            return True

        return False

    def adjust_action_confidence(
        self,
        base_action_confidence: float,
        pattern_match: Optional[PatternMatch],
        severity: SeverityLevel,
    ) -> float:
        """Adjust action confidence based on pattern history outcomes.

        Strong match to self_resolved pattern reduces confidence slightly
        (favor monitoring). Strong match to failure_progression increases it
        (favor escalation).

        Args:
            base_action_confidence: Computed action confidence [0, 1]
            pattern_match: Historical pattern match
            severity: Current severity level

        Returns:
            Adjusted action confidence [0, 1]
        """
        if not pattern_match or pattern_match.match_tier == "weak":
            return base_action_confidence

        adjustment = 0.0

        if pattern_match.outcome_type == "self_resolved":
            if pattern_match.match_tier == "strong":
                adjustment = -0.15  # Reduce confidence for monitoring vs escalation
            elif pattern_match.match_tier == "moderate":
                adjustment = -0.08

        elif pattern_match.outcome_type == "failure_progression":
            if pattern_match.match_tier == "strong":
                adjustment = +0.12  # Increase confidence for escalation
            elif pattern_match.match_tier == "moderate":
                adjustment = +0.06

        # Apply confidence weight from pattern
        weight = pattern_match.confidence_weight or 1.0
        adjustment *= weight

        return max(0.0, min(1.0, base_action_confidence + adjustment))

    def adjust_recommendation_aggressiveness(
        self,
        base_urgency: SeverityLevel,
        pattern_match: Optional[PatternMatch],
        live_evidence_severity: SeverityLevel,
    ) -> SeverityLevel:
        """Adjust recommendation urgency based on pattern outcome history.

        If pattern suggests self_resolved, don't escalate beyond live evidence.
        If pattern suggests failure_progression, escalation is justified.

        Args:
            base_urgency: Base recommendation urgency from live evidence
            pattern_match: Historical pattern match
            live_evidence_severity: Severity from current metrics

        Returns:
            Adjusted urgency level
        """
        if not pattern_match or pattern_match.match_tier == "weak":
            return base_urgency

        # Prefer live evidence over pattern memory
        if pattern_match.outcome_type == "self_resolved":
            # Don't escalate beyond what live evidence suggests
            if base_urgency == "HIGH" and live_evidence_severity in {"MODERATE", "LOW"}:
                return "ELEVATED"  # Moderate escalation, but not full HIGH
            return base_urgency

        elif pattern_match.outcome_type == "failure_progression":
            # Pattern suggests progression; live evidence justifies escalation
            if pattern_match.match_tier == "strong":
                if base_urgency == "MODERATE":
                    return "ELEVATED"
                elif base_urgency == "ELEVATED":
                    return "HIGH"

        return base_urgency

    def compute_pattern_influence_summary(
        self,
        pattern_match: Optional[PatternMatch],
        suppress: bool,
        action_confidence: float,
        severity: SeverityLevel,
    ) -> Optional[str]:
        """Generate human-readable explanation of pattern influence.

        Args:
            pattern_match: Historical pattern match
            suppress: Whether finding is suppressed
            action_confidence: Action confidence after adjustments
            severity: Current severity level

        Returns:
            Brief explanation of pattern influence, or None if no pattern
        """
        if not pattern_match:
            return None

        if pattern_match.match_tier == "weak":
            return "Pattern match weak; relying on live evidence"

        if pattern_match.outcome_type == "self_resolved":
            if pattern_match.match_tier == "strong":
                if suppress:
                    return "Similar to prior self-resolving pattern; monitoring suggested"
                else:
                    return "Similar to prior self-resolving pattern; but live evidence warrants surfacing"
            else:
                return "Historically showed self-resolution; confidence reduced for monitoring"

        elif pattern_match.outcome_type == "failure_progression":
            if pattern_match.match_tier == "strong":
                return "Historically associated with progression; escalation confidence increased"
            else:
                return "Historically associated with progression; recommend heightened vigilance"

        elif pattern_match.outcome_type == "persistent_degradation":
            if pattern_match.match_tier == "strong":
                return "Similar to persistent pattern; monitoring without escalation recommended"

        return None

    def handle_evidence_pattern_conflict(
        self,
        live_severity: SeverityLevel,
        pattern_match: Optional[PatternMatch],
        finding_confidence: float,
    ) -> tuple[bool, Optional[str]]:
        """Handle conflict between live evidence and pattern memory.

        If live evidence says severe but pattern suggests self_resolved:
        - Prefer live evidence
        - Reduce confidence slightly, but do NOT suppress justified HIGH/ELEVATED

        If live evidence is ambiguous and strong self_resolved pattern exists:
        - Favor monitoring over escalation

        Args:
            live_severity: Severity from current evidence
            pattern_match: Historical pattern match
            finding_confidence: Confidence in the current finding

        Returns:
            Tuple of (should_reduce_confidence, explanation)
        """
        if not pattern_match or pattern_match.match_tier == "weak":
            return (False, None)

        explanation = None

        if (
            live_severity in {"HIGH", "ELEVATED"}
            and pattern_match.outcome_type == "self_resolved"
            and pattern_match.match_tier == "strong"
        ):
            # Live evidence says severe, pattern says self-resolved
            explanation = "Live evidence indicates severity; pattern history suggests caution in escalation"
            return (True, explanation)  # Reduce confidence slightly

        if (
            live_severity in {"MODERATE", "LOW"}
            and pattern_match.outcome_type == "self_resolved"
            and pattern_match.match_tier == "strong"
            and finding_confidence < 0.7
        ):
            # Ambiguous evidence + strong self-resolved pattern
            explanation = "Ambiguous evidence with historical self-resolution pattern; recommend monitoring"
            return (False, explanation)

        return (False, None)
