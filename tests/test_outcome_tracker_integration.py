"""Integration tests for outcome tracker with decision engine."""

import pytest
from neraium_core.decision import DecisionEngine, OutcomeTracker
from neraium_core.decision.models import Decision


class TestOutcomeTrackerIntegration:
    """Tests for outcome tracker integrated with decision engine."""

    def test_engine_instantiation_without_tracking(self):
        """Engine should work without outcome tracking."""
        engine = DecisionEngine(enable_outcome_tracking=False)
        assert engine.outcome_tracker is None

    def test_engine_instantiation_with_tracking(self):
        """Engine should have tracker when enabled."""
        engine = DecisionEngine(enable_outcome_tracking=True)
        assert engine.outcome_tracker is not None
        assert isinstance(engine.outcome_tracker, OutcomeTracker)

    def test_tracker_accessibility(self):
        """Tracker should be accessible from engine."""
        engine = DecisionEngine(enable_outcome_tracking=True)
        tracker = engine.outcome_tracker
        assert tracker is not None
        assert len(tracker.logged_decisions) == 0

    def test_manual_decision_logging(self):
        """Manual logging should work independently."""
        tracker = OutcomeTracker()

        # Log a decision
        d_id = tracker.log_decision(
            asset_id="test_pump",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="accelerated_deterioration",
            reasoning_summary="Test decision",
            confidence=0.85,
        )

        assert d_id is not None
        assert tracker.get_decision(d_id) is not None

        # Attach outcome
        success = tracker.attach_outcome(d_id, outcome_type="failure")
        assert success is True

        # Verify evaluation
        decision = tracker.get_decision(d_id)
        assert decision.evaluation is not None
        assert decision.evaluation.correctness is True

    def test_multiple_decisions_tracking(self):
        """Should track multiple decisions independently."""
        engine1 = DecisionEngine(enable_outcome_tracking=True)
        engine2 = DecisionEngine(enable_outcome_tracking=True)

        # Log to tracker 1
        engine1.outcome_tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test 1",
        )

        # Log to tracker 2
        engine2.outcome_tracker.log_decision(
            asset_id="pump_002",
            frame_number=1,
            severity="HIGH",
            trajectory="degrading",
            degradation_stage="early_shift",
            reasoning_summary="Test 2",
        )

        # Trackers should be independent
        assert len(engine1.outcome_tracker.logged_decisions) == 1
        assert len(engine2.outcome_tracker.logged_decisions) == 1

    def test_tracker_metrics_computation(self):
        """Tracker should compute metrics correctly."""
        tracker = OutcomeTracker()

        # Log and evaluate decisions
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
            severity="LOW",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )

        tracker.attach_outcome(d1, outcome_type="failure")
        tracker.attach_outcome(d2, outcome_type="stabilized")

        metrics = tracker.compute_summary_metrics()
        assert metrics.total_logged_decisions == 2
        assert metrics.total_evaluated_decisions == 2
        assert metrics.correct_decisions == 2
        assert metrics.accuracy_rate == 1.0

    def test_tracker_export(self):
        """Tracker should export decisions."""
        tracker = OutcomeTracker()

        d_id = tracker.log_decision(
            asset_id="pump_001",
            frame_number=1,
            severity="MODERATE",
            trajectory="stable",
            degradation_stage="baseline",
            reasoning_summary="Test",
        )
        tracker.attach_outcome(d_id, outcome_type="stabilized")

        # Export as dict
        exported_dict = tracker.export_decisions(format="dict")
        assert d_id in exported_dict
        assert isinstance(exported_dict, dict)

        # Export as list
        exported_list = tracker.export_decisions(format="list")
        assert len(exported_list) == 1
        assert isinstance(exported_list, list)

    def test_backward_compatibility(self):
        """Engine should work normally without outcome tracking enabled."""
        engine = DecisionEngine(enable_outcome_tracking=False)

        # Verify no outcomes logged
        assert engine.outcome_tracker is None

        # Engine attributes should be intact
        assert engine.pattern_memory is not None
        assert engine.persistence_tracker is not None
        assert engine.frame_counter == 0
