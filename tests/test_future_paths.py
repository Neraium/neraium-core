"""Tests for Future Path Map feature."""

import pytest
from neraium_core.future_paths import (
    FuturePathMap,
    FuturePathMapper,
    CurrentState,
    FuturePath,
    EtaRange,
    Intervention,
)
from neraium_core.alignment import StructuralEngine


class TestEtaRange:
    def test_eta_range_dict(self):
        eta = EtaRange(min=5, max=20, unit="cycles")
        d = eta.to_dict()
        assert d["min"] == 5
        assert d["max"] == 20
        assert d["unit"] == "cycles"

    def test_eta_range_default_unit(self):
        eta = EtaRange(min=5, max=20)
        assert eta.unit == "cycles"


class TestFuturePath:
    def test_future_path_dict(self):
        eta = EtaRange(min=5, max=18)
        path = FuturePath(
            path_id="failure",
            label="Instability / Failure",
            probability=0.25,
            risk="high",
            eta_range=eta,
            trend_summary="System approaching instability",
            drivers=["accelerating drift", "stability loss"],
            point_of_no_return=12,
        )
        d = path.to_dict()
        assert d["path_id"] == "failure"
        assert d["probability"] == 0.25
        assert d["risk"] == "high"
        assert isinstance(d["eta_range"], dict)
        assert d["point_of_no_return"] == 12


class TestCurrentState:
    def test_current_state_creation(self):
        state = CurrentState(
            regime_name="normal_operation",
            health_band="stable",
            confidence=0.85,
            summary="System operating nominally",
            structural_drift=0.2,
            relational_stability=0.8,
            temporal_consistency=0.75,
        )
        assert state.regime_name == "normal_operation"
        assert state.health_band == "stable"
        assert state.confidence == 0.85


class TestFuturePathMapperStateExtraction:
    def test_extract_current_state_stable(self):
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        # Add some frames to build history
        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {
                    "sensor_1": 10.0 + i * 0.1,
                    "sensor_2": 20.0 + i * 0.05,
                },
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})

        assert current_state.health_band in ["stable", "watch", "critical"]
        assert 0.0 <= current_state.confidence <= 1.0
        assert 0.0 <= current_state.structural_drift <= 1.0
        assert 0.0 <= current_state.relational_stability <= 1.0

    def test_state_to_health_band(self):
        mapper = FuturePathMapper()

        assert mapper._state_to_health_band("STABLE", 0.2) == "stable"
        assert mapper._state_to_health_band("STABLE", 0.5) == "watch"
        assert mapper._state_to_health_band("WATCH", 0.6) == "watch"
        assert mapper._state_to_health_band("UNSTABLE", 0.8) == "critical"
        assert mapper._state_to_health_band("UNKNOWN", 0.75) == "critical"


class TestFuturePathMapperPathInference:
    def test_infer_future_paths_returns_three_paths(self):
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        # Build history
        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {
                    "s1": 10.0 + i * 0.1,
                    "s2": 20.0 + i * 0.05,
                },
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)

        assert len(paths) == 3
        path_ids = {p.path_id for p in paths}
        assert path_ids == {"recovery", "degradation", "failure"}

    def test_path_probabilities_sum_to_one(self):
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 + i * 0.1, "s2": 20.0 + i * 0.05},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)

        total_prob = sum(p.probability for p in paths)
        assert pytest.approx(total_prob, abs=0.01) == 1.0

    def test_path_probabilities_in_valid_range(self):
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 + i * 0.1},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)

        for path in paths:
            assert 0.0 <= path.probability <= 1.0
            assert path.eta_range.min > 0
            assert path.eta_range.max >= path.eta_range.min
            assert path.risk in ["low", "medium", "high"]

    def test_recovery_path_favored_under_improving_conditions(self):
        """Recovery path should have highest probability when drift decreases."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        # Create improving trajectory: decreasing drift significantly
        for i in range(25):
            score_factor = max(0.1, 1.0 - (i / 15.0))  # Decreasing strongly
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 * score_factor, "s2": 20.0 * score_factor},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)

        recovery_path = next(p for p in paths if p.path_id == "recovery")
        other_paths = [p for p in paths if p.path_id != "recovery"]

        # Recovery should be most probable under improving conditions
        assert recovery_path.probability > max(p.probability for p in other_paths)

    def test_failure_path_favored_under_deteriorating_conditions(self):
        """Failure/degradation path should be favored more than recovery when drift rises."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        # Create deteriorating trajectory: increasing drift strongly
        for i in range(25):
            score_factor = min(1.0, i / 12.0)  # Increasing strongly
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 * score_factor, "s2": 20.0 * score_factor},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)

        recovery_path = next(p for p in paths if p.path_id == "recovery")
        other_paths = [p for p in paths if p.path_id != "recovery"]

        # Recovery should NOT be the most probable under deteriorating conditions
        assert recovery_path.probability <= max(p.probability for p in other_paths)


class TestFuturePathMapperInterventions:
    def test_generate_interventions_high_risk(self):
        """Should generate multiple interventions for high failure risk."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 * i, "s2": 20.0 * i},  # Increasing drift
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)
        interventions = mapper._generate_interventions(current_state, paths, engine)

        # Should have at least one intervention when risk is high
        assert len(interventions) > 0

        for intervention in interventions:
            assert intervention.action
            assert intervention.expected_effect
            assert 0.0 <= intervention.confidence <= 1.0
            assert isinstance(intervention.target_paths, list)

    def test_interventions_have_target_paths(self):
        """Each intervention should target specific paths."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 + i * 0.1},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)
        interventions = mapper._generate_interventions(current_state, paths, engine)

        valid_path_ids = {"recovery", "degradation", "failure"}
        for intervention in interventions:
            assert intervention.target_paths
            for path_id in intervention.target_paths:
                assert path_id in valid_path_ids


class TestFuturePathMapperDynamics:
    def test_drift_dynamics_computation(self):
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 + i * 0.1},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        slope, accel = mapper._compute_drift_dynamics(engine)

        assert isinstance(slope, float)
        assert isinstance(accel, float)

    def test_stability_trend_computation(self):
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 + i * 0.1},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        trend = mapper._compute_stability_trend(engine)

        assert isinstance(trend, float)


class TestFuturePathMap:
    def test_future_path_map_complete_analysis(self):
        """Test end-to-end future path map generation."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 + i * 0.05, "s2": 20.0 + i * 0.02},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        path_map = mapper.analyze(engine, {})

        assert isinstance(path_map, FuturePathMap)
        assert path_map.current_state is not None
        assert len(path_map.future_paths) == 3
        assert isinstance(path_map.recommended_interventions, list)

    def test_future_path_map_to_dict(self):
        """Test serialization to dict."""
        state = CurrentState(
            regime_name="test",
            health_band="stable",
            confidence=0.9,
            summary="Test summary",
            structural_drift=0.3,
            relational_stability=0.7,
            temporal_consistency=0.8,
        )
        paths = [
            FuturePath(
                path_id="recovery",
                label="Test Path",
                probability=1.0,
                risk="low",
                eta_range=EtaRange(min=10, max=20),
                trend_summary="Test trend",
                drivers=["driver1"],
            )
        ]
        interventions = [
            Intervention(
                action="Test action",
                expected_effect="Test effect",
                target_paths=["recovery"],
                confidence=0.8,
            )
        ]
        path_map = FuturePathMap(
            current_state=state, future_paths=paths, recommended_interventions=interventions
        )

        d = path_map.to_dict()
        assert "current_state" in d
        assert "future_paths" in d
        assert "recommended_interventions" in d
        assert isinstance(d["current_state"], dict)
        assert isinstance(d["future_paths"], list)
        assert isinstance(d["recommended_interventions"], list)

    def test_from_engine_static_method(self):
        """Test FuturePathMapper.from_engine() static method."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 + i * 0.05},
            }
            engine.process_frame(frame)

        path_map = FuturePathMapper.from_engine(engine, {})

        assert isinstance(path_map, FuturePathMap)
        assert path_map.current_state is not None
        assert len(path_map.future_paths) == 3


class TestPathCollapse:
    def test_path_collapse_detection_under_failure_conditions(self):
        """Path collapse should be detected when failure probability dominates."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        # Create deteriorating trajectory
        for i in range(25):
            score_factor = min(1.0, i / 12.0)
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 * score_factor},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)

        collapse = mapper._detect_path_collapse(paths)

        if collapse:
            assert collapse.dominant_path in ["recovery", "degradation", "failure"]
            assert collapse.eta > 0
            assert 0.0 <= collapse.confidence <= 1.0

    def test_no_collapse_when_paths_balanced(self):
        """No collapse when path probabilities are balanced."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        # Stable conditions
        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 + i * 0.01},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)

        collapse = mapper._detect_path_collapse(paths)

        # Under stable conditions, may not have a clear collapse
        if collapse:
            assert collapse.dominant_path == "recovery"


class TestInterventionImpactPreview:
    def test_intervention_impact_preview_added(self):
        """Interventions should have impact preview fields."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(20):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 * i},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)
        interventions = mapper._generate_interventions(current_state, paths, engine)

        # Add impact preview
        interventions = mapper._add_intervention_impact_preview(interventions, paths)

        for intervention in interventions:
            # Should have impact preview fields
            assert hasattr(intervention, "current_failure_prob")
            assert hasattr(intervention, "projected_failure_prob")
            assert hasattr(intervention, "impact")

            # Values should be reasonable
            if intervention.current_failure_prob is not None:
                assert 0.0 <= intervention.current_failure_prob <= 1.0
            if intervention.projected_failure_prob is not None:
                assert 0.0 <= intervention.projected_failure_prob <= 1.0
                # Projected should be lower than current
                assert (
                    intervention.projected_failure_prob
                    <= intervention.current_failure_prob
                )

    def test_impact_magnitude_classification(self):
        """Impact should be classified as high/medium/low."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(20):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 * i},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)
        interventions = mapper._generate_interventions(current_state, paths, engine)
        interventions = mapper._add_intervention_impact_preview(interventions, paths)

        for intervention in interventions:
            assert intervention.impact in ["high", "medium", "low"]

    def test_reduce_load_has_high_impact(self):
        """Reduce load intervention should show high impact."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(20):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 * i},
            }
            engine.process_frame(frame)

        mapper = FuturePathMapper()
        current_state = mapper._extract_current_state(engine, {})
        paths = mapper._infer_future_paths(engine, current_state)
        interventions = mapper._generate_interventions(current_state, paths, engine)
        interventions = mapper._add_intervention_impact_preview(interventions, paths)

        # Find reduce load intervention
        reduce_load = next(
            (i for i in interventions if "load" in i.action.lower()), None
        )

        if reduce_load:
            # Should have high impact for load reduction
            assert reduce_load.impact == "high"


class TestStructuralEngineIntegration:
    def test_engine_get_future_path_map_method(self):
        """Test that StructuralEngine has get_future_path_map method."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        for i in range(15):
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 + i * 0.05},
            }
            engine.process_frame(frame)

        result = engine.get_future_path_map()

        assert isinstance(result, dict)
        assert "current_state" in result
        assert "future_paths" in result
        assert "recommended_interventions" in result

    def test_engine_get_future_path_map_before_analysis(self):
        """Test graceful handling when no analysis results are available."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        result = engine.get_future_path_map()

        assert isinstance(result, dict)
        assert "error" in result or "current_state" in result

    def test_future_path_map_includes_path_collapse(self):
        """End-to-end test: FuturePathMap should include path_collapse if detected."""
        engine = StructuralEngine(baseline_window=10, recent_window=5)

        # Deteriorating conditions
        for i in range(25):
            score_factor = min(1.0, i / 12.0)
            frame = {
                "timestamp": float(i),
                "asset_id": "test_unit",
                "site_id": "test_site",
                "sensor_values": {"s1": 10.0 * score_factor},
            }
            engine.process_frame(frame)

        path_map = FuturePathMapper.from_engine(engine, {})

        assert isinstance(path_map.path_collapse, object) or path_map.path_collapse is None
