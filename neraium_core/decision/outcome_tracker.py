"""Outcome tracker for decision evaluation and feedback loops.

Tracks decisions made by the system, their predicted context, and actual outcomes
that occur later. Enables evaluation of decision quality and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
from datetime import datetime
import uuid


OutcomeType = Literal[
    "stabilized",
    "degraded",
    "persistent_degradation",
    "failure",
    "recovered",
    "intervention_delayed_failure",
    "unknown",
]

TimingType = Literal[
    "early",      # Action recommended before needed
    "appropriate", # Action matched timing well
    "late",        # Action recommended after optimal window
    "unavailable", # Timing information not available
]


@dataclass
class DecisionPrediction:
    """What the system predicted at decision time."""
    severity: str  # "HIGH", "ELEVATED", "MODERATE", "LOW"
    trajectory: str  # "improving", "stable", "degrading", "unstable"
    degradation_stage: str  # e.g., "baseline", "early_shift", "emerging_degradation"
    pattern_match_tier: Optional[str] = None  # "weak", "moderate", "strong"
    predicted_outcome_type: Optional[str] = None  # What pattern suggested
    action_horizon: Optional[str] = None  # "now", "soon", "watchlist"
    decision_window_status: Optional[str] = None  # "OPEN", "NARROWING", "CRITICAL", "CLOSED"
    primary_action: Optional[str] = None  # Recommended action
    confidence: float = 0.5  # Overall decision confidence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OutcomeLog:
    """Actual outcome that occurred after decision."""
    outcome_type: OutcomeType
    timestamp: float  # When outcome was observed/logged
    description: str = ""  # Human-readable outcome description
    severity_at_outcome: Optional[str] = None  # Severity when outcome was observed
    trajectory_at_outcome: Optional[str] = None  # Trajectory when outcome occurred
    additional_metrics: dict[str, Any] = field(default_factory=dict)  # Any extra context

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimingDelta:
    """Timing analysis between prediction and outcome."""
    predicted_window_hours: Optional[float] = None  # Window the system predicted
    actual_time_to_outcome_hours: Optional[float] = None  # How long until outcome
    timing_type: TimingType = "unavailable"  # early/appropriate/late/unavailable
    timing_error_hours: Optional[float] = None  # Signed error (predicted - actual)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionOutcomeEvaluation:
    """Evaluation of decision quality."""
    correctness: bool  # Broadly correct or not (decision matched reality)
    correctness_reasoning: str = ""  # Why correct/incorrect
    timing_analysis: Optional[TimingDelta] = None
    action_appropriateness: str = "unknown"  # "appropriate", "early", "late", "not_applicable"
    severity_assessment: str = "unknown"  # "correct", "understated", "overstated"
    window_prediction_accuracy: Optional[float] = None  # [0, 1] how well window matched

    def to_dict(self) -> dict[str, Any]:
        d = {
            "correctness": self.correctness,
            "correctness_reasoning": self.correctness_reasoning,
            "timing_analysis": self.timing_analysis.to_dict() if self.timing_analysis else None,
            "action_appropriateness": self.action_appropriateness,
            "severity_assessment": self.severity_assessment,
            "window_prediction_accuracy": self.window_prediction_accuracy,
        }
        return d


@dataclass
class LoggedDecision:
    """A logged decision with all tracking info."""
    decision_id: str  # Unique identifier
    asset_id: str  # What asset/system this decision was about
    frame_number: int  # Which frame this decision was made
    timestamp: float  # When decision was made
    prediction: DecisionPrediction
    reasoning_summary: str  # Concise decision reasoning

    # Outcome (filled later)
    outcome: Optional[OutcomeLog] = None
    evaluation: Optional[DecisionOutcomeEvaluation] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "asset_id": self.asset_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "prediction": self.prediction.to_dict(),
            "reasoning_summary": self.reasoning_summary,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "metadata": self.metadata,
        }


@dataclass
class SummaryMetrics:
    """Summary metrics across evaluated decisions."""
    total_logged_decisions: int = 0
    total_evaluated_decisions: int = 0
    total_unevaluated_decisions: int = 0

    # Accuracy metrics
    correct_decisions: int = 0
    incorrect_decisions: int = 0
    accuracy_rate: float = 0.0  # [0, 1]

    # Timing metrics
    decisions_with_timing_data: int = 0
    average_timing_error_hours: float = 0.0  # Signed: negative = late, positive = early
    early_count: int = 0
    appropriate_count: int = 0
    late_count: int = 0

    # Per-outcome metrics
    outcomes_by_type: dict[OutcomeType, int] = field(default_factory=dict)

    # Per-severity metrics
    accuracy_by_severity: dict[str, float] = field(default_factory=dict)  # severity -> accuracy

    # Per-action metrics
    action_success_rate: dict[str, float] = field(default_factory=dict)  # action -> success_rate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OutcomeTracker:
    """Tracks decisions, outcomes, and evaluates quality."""

    def __init__(self):
        self.logged_decisions: dict[str, LoggedDecision] = {}  # decision_id -> LoggedDecision
        self.asset_decisions: dict[str, list[str]] = {}  # asset_id -> [decision_ids]

    def log_decision(
        self,
        asset_id: str,
        frame_number: int,
        severity: str,
        trajectory: str,
        degradation_stage: str,
        reasoning_summary: str,
        action_horizon: Optional[str] = None,
        primary_action: Optional[str] = None,
        decision_window_status: Optional[str] = None,
        pattern_match_tier: Optional[str] = None,
        predicted_outcome_type: Optional[str] = None,
        confidence: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log a decision for later evaluation.

        Args:
            asset_id: Asset/system identifier
            frame_number: Frame number when decision made
            severity: Decision severity
            trajectory: Predicted trajectory
            degradation_stage: Current degradation stage
            reasoning_summary: Concise decision reasoning
            action_horizon: Action timing horizon
            primary_action: Primary recommended action
            decision_window_status: Window status if available
            pattern_match_tier: Pattern match strength
            predicted_outcome_type: Predicted outcome type
            confidence: Overall decision confidence
            metadata: Additional context

        Returns:
            decision_id (str) for later outcome attachment
        """
        decision_id = str(uuid.uuid4())

        prediction = DecisionPrediction(
            severity=severity,
            trajectory=trajectory,
            degradation_stage=degradation_stage,
            action_horizon=action_horizon,
            primary_action=primary_action,
            decision_window_status=decision_window_status,
            pattern_match_tier=pattern_match_tier,
            predicted_outcome_type=predicted_outcome_type,
            confidence=confidence,
        )

        logged_decision = LoggedDecision(
            decision_id=decision_id,
            asset_id=asset_id,
            frame_number=frame_number,
            timestamp=datetime.now().timestamp(),
            prediction=prediction,
            reasoning_summary=reasoning_summary,
            metadata=metadata or {},
        )

        self.logged_decisions[decision_id] = logged_decision

        if asset_id not in self.asset_decisions:
            self.asset_decisions[asset_id] = []
        self.asset_decisions[asset_id].append(decision_id)

        return decision_id

    def attach_outcome(
        self,
        decision_id: str,
        outcome_type: OutcomeType,
        description: str = "",
        severity_at_outcome: Optional[str] = None,
        trajectory_at_outcome: Optional[str] = None,
        additional_metrics: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Attach actual outcome to a logged decision.

        Args:
            decision_id: ID from log_decision
            outcome_type: What actually happened
            description: Human-readable outcome description
            severity_at_outcome: Severity when outcome observed
            trajectory_at_outcome: Trajectory when outcome observed
            additional_metrics: Extra context about outcome

        Returns:
            True if outcome attached successfully, False if decision not found
        """
        if decision_id not in self.logged_decisions:
            return False

        outcome = OutcomeLog(
            outcome_type=outcome_type,
            timestamp=datetime.now().timestamp(),
            description=description,
            severity_at_outcome=severity_at_outcome,
            trajectory_at_outcome=trajectory_at_outcome,
            additional_metrics=additional_metrics or {},
        )

        self.logged_decisions[decision_id].outcome = outcome
        self.logged_decisions[decision_id].evaluation = self._evaluate_decision(
            self.logged_decisions[decision_id]
        )
        return True

    def _evaluate_decision(self, logged_decision: LoggedDecision) -> DecisionOutcomeEvaluation:
        """Evaluate decision quality given outcome.

        Args:
            logged_decision: Decision with both prediction and outcome

        Returns:
            DecisionOutcomeEvaluation with quality metrics
        """
        if logged_decision.outcome is None:
            return DecisionOutcomeEvaluation(
                correctness=False,
                correctness_reasoning="No outcome attached"
            )

        prediction = logged_decision.prediction
        outcome = logged_decision.outcome

        # === CORRECTNESS EVALUATION ===
        correctness, correctness_reasoning = self._evaluate_correctness(
            prediction, outcome
        )

        # === TIMING ANALYSIS ===
        timing_analysis = self._evaluate_timing(prediction, outcome)

        # === ACTION APPROPRIATENESS ===
        action_appropriateness = self._evaluate_action_appropriateness(
            prediction, outcome
        )

        # === SEVERITY ASSESSMENT ===
        severity_assessment = self._evaluate_severity(prediction, outcome)

        # === WINDOW PREDICTION ===
        window_accuracy = self._evaluate_window_prediction(prediction, outcome)

        return DecisionOutcomeEvaluation(
            correctness=correctness,
            correctness_reasoning=correctness_reasoning,
            timing_analysis=timing_analysis,
            action_appropriateness=action_appropriateness,
            severity_assessment=severity_assessment,
            window_prediction_accuracy=window_accuracy,
        )

    def _evaluate_correctness(
        self,
        prediction: DecisionPrediction,
        outcome: OutcomeLog,
    ) -> tuple[bool, str]:
        """Determine if decision was broadly correct.

        Correctness means:
        - If severity was HIGH/ELEVATED, outcome should be serious
        - If trajectory was degrading, should see degradation/failure
        - If recommended action, should see appropriate outcome type
        """
        if outcome.outcome_type == "unknown":
            return False, "Outcome unknown - cannot evaluate"

        # High/Elevated severity decisions are correct if outcome is serious
        if prediction.severity in ("HIGH", "ELEVATED"):
            serious_outcomes = {
                "degraded",
                "persistent_degradation",
                "failure",
                "intervention_delayed_failure",
            }
            is_correct = outcome.outcome_type in serious_outcomes
            reason = (
                "Predicted serious outcome, got: " + outcome.outcome_type
                if is_correct
                else "Predicted serious outcome but got: " + outcome.outcome_type
            )
            return is_correct, reason

        # Moderate/Low severity predictions should not see serious outcomes
        if prediction.severity in ("MODERATE", "LOW"):
            benign_outcomes = {"stabilized", "recovered"}
            is_correct = outcome.outcome_type in benign_outcomes
            reason = (
                "Predicted benign outcome, got: " + outcome.outcome_type
                if is_correct
                else "Predicted benign outcome but got: " + outcome.outcome_type
            )
            return is_correct, reason

        return False, "Could not determine correctness"

    def _evaluate_timing(
        self,
        prediction: DecisionPrediction,
        outcome: OutcomeLog,
    ) -> Optional[TimingDelta]:
        """Evaluate timing of decision vs actual outcome.

        Returns:
            TimingDelta with timing analysis, or None if insufficient data
        """
        # If we have time-to-outcome data in additional metrics
        actual_hours = outcome.additional_metrics.get("time_to_outcome_hours")
        predicted_hours = outcome.additional_metrics.get("predicted_window_hours")

        if actual_hours is None:
            return TimingDelta(timing_type="unavailable")

        timing_type = "unavailable"
        timing_error = None

        if predicted_hours is not None:
            timing_error = predicted_hours - actual_hours
            if timing_error > 2.0:
                timing_type = "early"
            elif timing_error < -2.0:
                timing_type = "late"
            else:
                timing_type = "appropriate"

        return TimingDelta(
            predicted_window_hours=predicted_hours,
            actual_time_to_outcome_hours=actual_hours,
            timing_type=timing_type,
            timing_error_hours=timing_error,
        )

    def _evaluate_action_appropriateness(
        self,
        prediction: DecisionPrediction,
        outcome: OutcomeLog,
    ) -> str:
        """Evaluate if recommended action was appropriate for outcome."""
        if prediction.primary_action is None:
            return "not_applicable"

        # If degradation/failure occurred, action was appropriate to recommend
        serious_outcomes = {
            "degraded",
            "persistent_degradation",
            "failure",
            "intervention_delayed_failure",
        }

        if outcome.outcome_type in serious_outcomes:
            return "appropriate"

        # If system stabilized, monitor-type actions were appropriate
        if outcome.outcome_type == "stabilized":
            if prediction.primary_action in ("monitor", "watchlist"):
                return "appropriate"
            else:
                return "early"

        return "unknown"

    def _evaluate_severity(
        self,
        prediction: DecisionPrediction,
        outcome: OutcomeLog,
    ) -> str:
        """Evaluate if severity prediction was accurate."""
        actual_severity = outcome.severity_at_outcome

        if actual_severity is None:
            return "unknown"

        predicted_severity = prediction.severity

        # Direct match
        if predicted_severity == actual_severity:
            return "correct"

        # Severity rank comparison
        severity_rank = {"LOW": 0, "MODERATE": 1, "ELEVATED": 2, "HIGH": 3}
        predicted_rank = severity_rank.get(predicted_severity, 1)
        actual_rank = severity_rank.get(actual_severity, 1)

        if predicted_rank > actual_rank:
            return "overstated"
        else:
            return "understated"

    def _evaluate_window_prediction(
        self,
        prediction: DecisionPrediction,
        outcome: OutcomeLog,
    ) -> Optional[float]:
        """Evaluate how well decision window was predicted.

        Returns:
            Accuracy score [0, 1] or None if insufficient data
        """
        window_status = prediction.decision_window_status

        if window_status is None:
            return None

        # If window was CLOSED and outcome was failure/degradation, accuracy is high
        if window_status == "CLOSED":
            serious = {
                "degraded",
                "persistent_degradation",
                "failure",
                "intervention_delayed_failure",
            }
            return 0.9 if outcome.outcome_type in serious else 0.2

        # If window was OPEN and system stabilized, accuracy is high
        if window_status == "OPEN":
            return 0.85 if outcome.outcome_type == "stabilized" else 0.4

        # NARROWING or CRITICAL should see intervention-relevant outcomes
        if window_status in ("NARROWING", "CRITICAL"):
            interventionable = {"stabilized", "recovered", "intervention_delayed_failure"}
            return 0.75 if outcome.outcome_type in interventionable else 0.3

        return None

    def get_decision(self, decision_id: str) -> Optional[LoggedDecision]:
        """Retrieve a logged decision by ID."""
        return self.logged_decisions.get(decision_id)

    def get_asset_decisions(self, asset_id: str) -> list[LoggedDecision]:
        """Get all decisions for an asset."""
        decision_ids = self.asset_decisions.get(asset_id, [])
        return [self.logged_decisions[d_id] for d_id in decision_ids]

    def get_unevaluated_decisions(self) -> list[LoggedDecision]:
        """Get decisions that haven't been evaluated yet (no outcome)."""
        return [
            d for d in self.logged_decisions.values()
            if d.outcome is None
        ]

    def get_evaluated_decisions(self) -> list[LoggedDecision]:
        """Get decisions that have been evaluated (have outcome)."""
        return [
            d for d in self.logged_decisions.values()
            if d.outcome is not None
        ]

    def compute_summary_metrics(self) -> SummaryMetrics:
        """Compute summary metrics across all logged decisions."""
        all_decisions = list(self.logged_decisions.values())
        evaluated = self.get_evaluated_decisions()

        metrics = SummaryMetrics(
            total_logged_decisions=len(all_decisions),
            total_evaluated_decisions=len(evaluated),
            total_unevaluated_decisions=len(all_decisions) - len(evaluated),
        )

        if not evaluated:
            return metrics

        # Correctness metrics
        correct_count = sum(1 for d in evaluated if d.evaluation and d.evaluation.correctness)
        metrics.correct_decisions = correct_count
        metrics.incorrect_decisions = len(evaluated) - correct_count
        metrics.accuracy_rate = correct_count / len(evaluated) if evaluated else 0.0

        # Timing metrics
        with_timing = [
            d for d in evaluated
            if d.evaluation and d.evaluation.timing_analysis
            and d.evaluation.timing_analysis.timing_error_hours is not None
        ]
        if with_timing:
            metrics.decisions_with_timing_data = len(with_timing)
            timing_errors = [
                d.evaluation.timing_analysis.timing_error_hours for d in with_timing
            ]
            metrics.average_timing_error_hours = sum(timing_errors) / len(timing_errors)

            for d in with_timing:
                timing_type = d.evaluation.timing_analysis.timing_type
                if timing_type == "early":
                    metrics.early_count += 1
                elif timing_type == "appropriate":
                    metrics.appropriate_count += 1
                elif timing_type == "late":
                    metrics.late_count += 1

        # Outcome distribution
        for d in evaluated:
            if d.outcome:
                outcome_type = d.outcome.outcome_type
                metrics.outcomes_by_type[outcome_type] = (
                    metrics.outcomes_by_type.get(outcome_type, 0) + 1
                )

        # Severity-based accuracy
        by_severity = {}
        for d in evaluated:
            severity = d.prediction.severity
            if severity not in by_severity:
                by_severity[severity] = {"correct": 0, "total": 0}
            by_severity[severity]["total"] += 1
            if d.evaluation and d.evaluation.correctness:
                by_severity[severity]["correct"] += 1

        for severity, counts in by_severity.items():
            if counts["total"] > 0:
                metrics.accuracy_by_severity[severity] = (
                    counts["correct"] / counts["total"]
                )

        # Action-based success rate
        by_action = {}
        for d in evaluated:
            action = d.prediction.primary_action
            if action is None or action == "":
                continue
            if action not in by_action:
                by_action[action] = {"successful": 0, "total": 0}
            by_action[action]["total"] += 1
            if d.evaluation and d.evaluation.correctness:
                by_action[action]["successful"] += 1

        for action, counts in by_action.items():
            if counts["total"] > 0:
                metrics.action_success_rate[action] = counts["successful"] / counts["total"]

        return metrics

    def export_decisions(self, format: str = "dict") -> Any:
        """Export all decisions in specified format.

        Args:
            format: "dict" for Python dict, "list" for list of dicts

        Returns:
            Decisions in requested format
        """
        if format == "dict":
            return {
                d_id: d.to_dict()
                for d_id, d in self.logged_decisions.items()
            }
        elif format == "list":
            return [d.to_dict() for d in self.logged_decisions.values()]
        else:
            raise ValueError(f"Unknown format: {format}")

    def clear(self):
        """Clear all logged decisions."""
        self.logged_decisions.clear()
        self.asset_decisions.clear()
