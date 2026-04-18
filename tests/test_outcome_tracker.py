"""Tests for outcome tracker: decision logging, outcome attachment, and evaluation."""

import pytest
from neraium_core.decision.outcome_tracker import (
    OutcomeTracker,
    OutcomeType,
    DecisionPrediction,
    OutcomeLog,
    LoggedDecision,
    TimingDelta,
)


@pytest.fixture
def tracker():
    """Create fresh tracker for each test."""
    return OutcomeTracker()


class TestDecisionLogging:
    """Tests for logging decisions."""

    def test_log_decision_creates_id(self, tracker):
        """Logged decision should get unique ID."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=42,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Drift increasing",
        )
        assert d_id is not None
        assert isinstance(d_id, str)
        assert len(d_id) > 0

    def test_log_decision_stored(self, tracker):
        """Logged decision should be retrievable."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=42,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Drift increasing",
            confidence=0.85,
        )
        decision = tracker.get_decision(d_id)
        assert decision is not None
        assert decision.decision_id == d_id
        assert decision.asset_id == "pump_001"
        assert decision.frame_number == 42
        assert decision.prediction.severity == "HIGH"
        assert decision.prediction.confidence == 0.85

    def test_log_decision_with_all_fields(self, tracker):
        """Log decision with all optional fields."""
        d_id = tracker.log_decision(
            asset_id="pump_002",
            frame_number=100,
            severity="ELEVATED",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Monitoring",
            action_horizon="soon",
            primary_action="inspect",
            decision_window_status="OPEN",
            pattern_match_tier="strong",
            predicted_outcome_type="self_resolved",
            confidence=0.92,
            metadata={"test": "data"},
        )
        decision = tracker.get_decision(d_id)
        assert decision.prediction.action_horizon == "soon"
        assert decision.prediction.primary_action == "inspect"
        assert decision.prediction.decision_window_status == "OPEN"
        assert decision.prediction.pattern_match_tier == "strong"
        assert decision.prediction.confidence == 0.92
        assert decision.metadata["test"] == "data"

    def test_asset_decisions_grouped(self, tracker):
        """Decisions should be grouped by asset."""
        d1 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="LOW",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Stable",
        )
        d2 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=2,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Still stable",
        )
        d3 = tracker.log_decision(
            asset_id="pump_002",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Different pump",
        )
        decisions = tracker.get_asset_decisions("pump_001")
        assert len(decisions) == 2
        assert all(d.asset_id == "pump_001" for d in decisions)

    def test_unevaluated_decisions_list(self, tracker):
        """Unevaluated decisions should be listable."""
        d1 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="LOW",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )
        d2 = tracker.log_decision(
            asset_id="pump_002",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Test",
        )
        unevaluated = tracker.get_unevaluated_decisions()
        assert len(unevaluated) == 2
        assert all(d.outcome is None for d in unevaluated)


class TestOutcomeAttachment:
    """Tests for attaching outcomes to decisions."""

    def test_attach_outcome_success(self, tracker):
        """Outcome should attach successfully."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Drift increasing",
        )
        result = tracker.attach_outcome(
            d_id,
            outcome_type="persistent_degradation",
            description="System continued degrading",
        )
        assert result is True
        decision = tracker.get_decision(d_id)
        assert decision.outcome is not None
        assert decision.outcome.outcome_type == "persistent_degradation"

    def test_attach_outcome_missing_decision(self, tracker):
        """Attaching outcome to missing decision should fail."""
        result = tracker.attach_outcome(
            "nonexistent_id",
            outcome_type="stabilized",
            description="Test",
        )
        assert result is False

    def test_attach_outcome_with_metrics(self, tracker):
        """Outcome should store additional metrics."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Monitoring",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="stabilized",
            severity_at_outcome="LOW",
            trajectory_at_outcome="improving",
            additional_metrics={
                "time_to_outcome_hours": 24.5,
                "predicted_window_hours": 48.0,
            },
        )
        decision = tracker.get_decision(d_id)
        assert decision.outcome.severity_at_outcome == "LOW"
        assert decision.outcome.additional_metrics["time_to_outcome_hours"] == 24.5

    def test_evaluated_decisions_list(self, tracker):
        """Evaluated decisions should be listable separately."""
        d1 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="LOW",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )
        d2 = tracker.log_decision(
            asset_id="pump_002",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Test",
        )
        tracker.attach_outcome(d1, outcome_type="stabilized")
        evaluated = tracker.get_evaluated_decisions()
        assert len(evaluated) == 1
        assert evaluated[0].decision_id == d1
        assert evaluated[0].outcome is not None


class TestDecisionEvaluation:
    """Tests for evaluating decision quality."""

    def test_high_severity_correct_with_serious_outcome(self, tracker):
        """HIGH severity prediction correct if outcome is serious."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="accelerated_deterioration",
            reasoning_summary="Critical degradation",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="failure",
            severity_at_outcome="HIGH",
        )
        decision = tracker.get_decision(d_id)
        assert decision.evaluation is not None
        assert decision.evaluation.correctness is True

    def test_high_severity_incorrect_with_benign_outcome(self, tracker):
        """HIGH severity prediction incorrect if outcome is benign."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="accelerated_deterioration",
            reasoning_summary="Critical degradation",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="stabilized",
            severity_at_outcome="LOW",
        )
        decision = tracker.get_decision(d_id)
        assert decision.evaluation is not None
        assert decision.evaluation.correctness is False

    def test_moderate_severity_correct_with_benign_outcome(self, tracker):
        """MODERATE severity prediction correct if outcome is benign."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Monitoring",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="stabilized",
            severity_at_outcome="LOW",
        )
        decision = tracker.get_decision(d_id)
        assert decision.evaluation is not None
        assert decision.evaluation.correctness is True

    def test_timing_evaluation_early(self, tracker):
        """Timing should be marked early if predicted before actual."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="ELEVATED",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Early warning",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="persistent_degradation",
            additional_metrics={
                "predicted_window_hours": 48.0,
                "time_to_outcome_hours": 24.0,
            },
        )
        decision = tracker.get_decision(d_id)
        assert decision.evaluation.timing_analysis is not None
        assert decision.evaluation.timing_analysis.timing_type == "early"
        assert decision.evaluation.timing_analysis.timing_error_hours == 24.0

    def test_timing_evaluation_late(self, tracker):
        """Timing should be marked late if predicted after actual."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Too late",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="degraded",
            additional_metrics={
                "predicted_window_hours": 12.0,
                "time_to_outcome_hours": 48.0,
            },
        )
        decision = tracker.get_decision(d_id)
        assert decision.evaluation.timing_analysis.timing_type == "late"

    def test_timing_evaluation_appropriate(self, tracker):
        """Timing should be marked appropriate if close match."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="ELEVATED",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Good timing",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="persistent_degradation",
            additional_metrics={
                "predicted_window_hours": 36.0,
                "time_to_outcome_hours": 38.0,
            },
        )
        decision = tracker.get_decision(d_id)
        assert decision.evaluation.timing_analysis.timing_type == "appropriate"

    def test_severity_assessment_correct(self, tracker):
        """Severity assessment should match prediction to outcome."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="ELEVATED",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Test",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="persistent_degradation",
            severity_at_outcome="ELEVATED",
        )
        decision = tracker.get_decision(d_id)
        assert decision.evaluation.severity_assessment == "correct"

    def test_severity_assessment_overstated(self, tracker):
        """Severity assessment should mark overstated predictions."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Test",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="stabilized",
            severity_at_outcome="LOW",
        )
        decision = tracker.get_decision(d_id)
        assert decision.evaluation.severity_assessment == "overstated"

    def test_window_prediction_accuracy(self, tracker):
        """Window prediction accuracy should be scored."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="ELEVATED",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Test",
            decision_window_status="CLOSED",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="failure",
        )
        decision = tracker.get_decision(d_id)
        assert decision.evaluation.window_prediction_accuracy is not None
        assert decision.evaluation.window_prediction_accuracy > 0.7

    def test_unknown_outcome_evaluation(self, tracker):
        """Unknown outcomes should be handled gracefully."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )
        tracker.attach_outcome(
            d_id,
            outcome_type="unknown",
        )
        decision = tracker.get_decision(d_id)
        assert decision.evaluation is not None
        assert decision.evaluation.correctness is False


class TestSummaryMetrics:
    """Tests for summary metrics computation."""

    def test_empty_metrics(self, tracker):
        """Empty tracker should have zero metrics."""
        metrics = tracker.compute_summary_metrics()
        assert metrics.total_logged_decisions == 0
        assert metrics.total_evaluated_decisions == 0
        assert metrics.total_unevaluated_decisions == 0
        assert metrics.accuracy_rate == 0.0

    def test_metrics_counts(self, tracker):
        """Metrics should count logged and evaluated decisions correctly."""
        # Log 3 decisions
        d1 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Test",
        )
        d2 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=2,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )
        d3 = tracker.log_decision(
            asset_id="pump_002",
            frame_number=1,
            severity="LOW",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )
        # Evaluate 2 decisions
        tracker.attach_outcome(d1, outcome_type="failure")
        tracker.attach_outcome(d2, outcome_type="stabilized")

        metrics = tracker.compute_summary_metrics()
        assert metrics.total_logged_decisions == 3
        assert metrics.total_evaluated_decisions == 2
        assert metrics.total_unevaluated_decisions == 1

    def test_accuracy_rate(self, tracker):
        """Accuracy rate should match correct vs total evaluated."""
        d1 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="accelerated_deterioration",
            reasoning_summary="Critical",
        )
        d2 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=2,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="accelerated_deterioration",
            reasoning_summary="Critical",
        )
        tracker.attach_outcome(d1, outcome_type="failure")  # Correct
        tracker.attach_outcome(d2, outcome_type="stabilized")  # Incorrect

        metrics = tracker.compute_summary_metrics()
        assert metrics.correct_decisions == 1
        assert metrics.incorrect_decisions == 1
        assert metrics.accuracy_rate == 0.5

    def test_timing_metrics(self, tracker):
        """Timing metrics should be computed correctly."""
        d1 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="ELEVATED",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Test",
        )
        d2 = tracker.log_decision(
            asset_id="pump_002",
            frame_number=1,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )
        tracker.attach_outcome(
            d1,
            outcome_type="persistent_degradation",
            additional_metrics={
                "predicted_window_hours": 48.0,
                "time_to_outcome_hours": 36.0,
            },
        )
        tracker.attach_outcome(
            d2,
            outcome_type="stabilized",
            additional_metrics={
                "predicted_window_hours": 24.0,
                "time_to_outcome_hours": 12.0,
            },
        )

        metrics = tracker.compute_summary_metrics()
        assert metrics.decisions_with_timing_data == 2
        assert metrics.early_count == 2
        assert metrics.appropriate_count == 0
        assert metrics.late_count == 0

    def test_outcome_distribution(self, tracker):
        """Outcome types should be counted."""
        outcomes = [
            ("failure", "pump_001"),
            ("degraded", "pump_002"),
            ("stabilized", "pump_003"),
            ("persistent_degradation", "pump_001"),
        ]
        d_ids = []
        for i, (outcome_type, asset_id) in enumerate(outcomes):
            d_id = tracker.log_decision(
                asset_id=asset_id,
                frame_number=i,
                severity="HIGH" if outcome_type != "stabilized" else "LOW",
                trajectory="degrading" if outcome_type != "stabilized" else "stable",
                degradation_stage="baseline",
                reasoning_summary="Test",
            )
            d_ids.append((d_id, outcome_type))

        for d_id, outcome_type in d_ids:
            tracker.attach_outcome(d_id, outcome_type=outcome_type)

        metrics = tracker.compute_summary_metrics()
        assert len(metrics.outcomes_by_type) == 4
        assert metrics.outcomes_by_type["failure"] == 1
        assert metrics.outcomes_by_type["degraded"] == 1

    def test_severity_accuracy(self, tracker):
        """Accuracy should be tracked by severity."""
        # HIGH: 1 correct, 1 incorrect
        d1 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="accelerated_deterioration",
            reasoning_summary="Test",
        )
        d2 = tracker.log_decision(
            asset_id="pump_002",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="accelerated_deterioration",
            reasoning_summary="Test",
        )
        # MODERATE: 1 correct
        d3 = tracker.log_decision(
            asset_id="pump_003",
            frame_number=1,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )

        tracker.attach_outcome(d1, outcome_type="failure")
        tracker.attach_outcome(d2, outcome_type="stabilized")
        tracker.attach_outcome(d3, outcome_type="stabilized")

        metrics = tracker.compute_summary_metrics()
        assert metrics.accuracy_by_severity["HIGH"] == 0.5
        assert metrics.accuracy_by_severity["MODERATE"] == 1.0


class TestExport:
    """Tests for exporting decision data."""

    def test_export_dict_format(self, tracker):
        """Export should work in dict format."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )
        tracker.attach_outcome(d_id, outcome_type="stabilized")

        exported = tracker.export_decisions(format="dict")
        assert isinstance(exported, dict)
        assert d_id in exported
        assert exported[d_id]["decision_id"] == d_id

    def test_export_list_format(self, tracker):
        """Export should work in list format."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )
        tracker.attach_outcome(d_id, outcome_type="stabilized")

        exported = tracker.export_decisions(format="list")
        assert isinstance(exported, list)
        assert len(exported) == 1
        assert exported[0]["decision_id"] == d_id

    def test_export_invalid_format(self, tracker):
        """Export should reject invalid formats."""
        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="LOW",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )
        with pytest.raises(ValueError):
            tracker.export_decisions(format="invalid")


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_decisions_are_deterministic(self, tracker):
        """Same decision inputs should produce consistent evaluation."""
        # Log same decision twice
        d1 = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="accelerated_deterioration",
            reasoning_summary="Critical degradation",
        )
        # Attach same outcome
        tracker.attach_outcome(d1, outcome_type="failure")

        decision = tracker.get_decision(d1)
        eval1 = decision.evaluation.to_dict()

        # Create new tracker with same decision data
        tracker2 = OutcomeTracker()
        d2 = tracker2.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="accelerated_deterioration",
            reasoning_summary="Critical degradation",
        )
        tracker2.attach_outcome(d2, outcome_type="failure")
        decision2 = tracker2.get_decision(d2)
        eval2 = decision2.evaluation.to_dict()

        # Evaluations should match (except decision_id timestamps)
        assert eval1["correctness"] == eval2["correctness"]
        assert eval1["correctness_reasoning"] == eval2["correctness_reasoning"]
