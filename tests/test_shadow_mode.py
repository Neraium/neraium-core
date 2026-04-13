"""Tests for shadow-mode runner and evidence pipeline."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from validation.shadow_mode.runner import ShadowModeRunner
from validation.shadow_mode.evidence import (
    ShadowModeEvidenceLogger,
    EvidenceFrame,
    EvidenceDataFrame,
)
from validation.shadow_mode.report import ShadowModeSummaryReport
from validation.shadow_mode.replay_diff import ReplayDiff


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dummy_decision_fn():
    """Minimal decision function for testing."""

    def fn(frame: Dict[str, Any]) -> Dict[str, Any]:
        drift = frame.get("drift_signal", 0.1)
        return {
            "state": "WATCH" if drift > 0.3 else "STABLE",
            "policy_state": "WATCH" if drift > 0.3 else "STABLE",
            "structural_drift_score": drift,
            "structural_drift_score_smoothed": drift * 0.9,
            "relational_instability_score": drift * 0.5,
            "transition_pressure": 0.0,
            "transition_detected": False,
            "transition_state": "NONE",
            "system_health": 100,
            "active_sensor_count": 10,
            "missing_sensor_count": 0,
            "data_quality_summary": {},
            "dominant_driver": "sensor_1",
        }

    return fn


@pytest.fixture
def sample_telemetry():
    """Create sample telemetry frames."""
    frames = []
    for i in range(10):
        frames.append(
            {
                "asset_id": f"asset_{i % 3}",
                "frame_index": i,
                "timestamp": f"2024-04-12T12:{i:02d}:00Z",
                "drift_signal": 0.1 + (0.1 * i % 5),
                "domain": "thermal",
                "system_type": "ims",
            }
        )
    return frames


# ============================================================================
# Tests: Evidence Logger
# ============================================================================


class TestEvidenceLogger:
    """Test evidence frame creation and logging."""

    def test_logger_initialization(self, temp_output_dir):
        """Logger initializes with metadata."""
        logger = ShadowModeEvidenceLogger(
            output_dir=str(temp_output_dir),
            run_name="test_run",
            config={"param": "value"},
            engine_version="1.0",
        )

        assert logger.frame_count == 0
        assert logger.run_name == "test_run"
        assert logger.config == {"param": "value"}
        assert logger.log_path.exists()

    def test_write_frame(self, temp_output_dir):
        """Logger writes frames to JSONL."""
        logger = ShadowModeEvidenceLogger(
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        frame = EvidenceFrame(
            timestamp_utc="2024-04-12T12:00:00Z",
            frame_index=0,
            processing_latency_ms=1.5,
            asset_id="asset_1",
            unit_id="unit_1",
            domain="thermal",
            system_type="ims",
            state="STABLE",
            policy_state="STABLE",
            structural_drift_score=0.1,
            structural_drift_score_smoothed=0.09,
            relational_instability_score=0.05,
            transition_pressure=0.0,
            confidence_score=0.95,
            data_quality_summary={},
            active_sensor_count=10,
            missing_sensor_count=0,
            transition_detected=False,
            transition_state="NONE",
            regime_name=None,
            regime_distance=None,
            dominant_driver="sensor_1",
            top_drivers=[{"name": "sensor_1", "score": 0.8}],
            validation_errors=None,
            input_validation_passed=True,
            raw_engine_output={},
        )

        logger.write_frame(frame)
        logger.close()

        # Verify file contents
        with open(logger.log_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            logged = json.loads(lines[0])
            assert logged["asset_id"] == "asset_1"
            assert logged["state"] == "STABLE"

    def test_write_raw_frame(self, temp_output_dir):
        """Logger accepts raw dictionaries."""
        logger = ShadowModeEvidenceLogger(
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        frame_data = {
            "asset_id": "asset_1",
            "state": "ALERT",
            "structural_drift_score": 0.5,
        }

        logger.write_raw_frame(frame_data)
        logger.close()

        with open(logger.log_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            logged = json.loads(lines[0])
            assert logged["state"] == "ALERT"

    def test_metadata_generation(self, temp_output_dir):
        """Logger generates metadata file."""
        logger = ShadowModeEvidenceLogger(
            output_dir=str(temp_output_dir),
            run_name="test_run",
            engine_version="2.0",
        )

        for i in range(5):
            logger.write_raw_frame(
                {
                    "asset_id": f"asset_{i % 2}",
                    "frame_index": i,
                    "processing_latency_ms": 1.0 + i * 0.5,
                }
            )

        metadata = logger.close()

        assert metadata["frame_count"] == 5
        assert metadata["asset_count"] == 2
        assert "latency_stats_ms" in metadata
        assert metadata["engine_version"] == "2.0"
        assert (
            metadata["latency_stats_ms"]["mean_ms"] > 1.0
        )  # Should average > 1ms


# ============================================================================
# Tests: Report Generation
# ============================================================================


class TestShadowModeReport:
    """Test summary report generation."""

    def test_report_from_evidence(self, temp_output_dir, sample_telemetry):
        """Report generator processes evidence log."""
        # Create evidence log
        logger = ShadowModeEvidenceLogger(
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        for i, frame in enumerate(sample_telemetry):
            drift = frame.get("drift_signal", 0.1)
            state = "WATCH" if drift > 0.3 else "STABLE"
            logger.write_raw_frame(
                {
                    "asset_id": frame["asset_id"],
                    "frame_index": i,
                    "state": state,
                    "policy_state": state,
                    "structural_drift_score": drift,
                    "relational_instability_score": drift * 0.5,
                    "transition_detected": drift > 0.4,
                    "processing_latency_ms": 0.5 + i * 0.1,
                    "validation_errors": None,
                    "input_validation_passed": True,
                }
            )

        logger.close()

        # Generate report
        report_gen = ShadowModeSummaryReport()
        report_gen.load_evidence(logger.log_path)
        report = report_gen.generate_summary()

        assert "overview" in report
        assert report["overview"]["total_frames"] == len(sample_telemetry)
        assert len(report["per_asset"]) > 0
        assert "state_distribution" in report
        assert "latency" in report

    def test_state_distribution(self, temp_output_dir):
        """Report accurately counts state distribution."""
        logger = ShadowModeEvidenceLogger(
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        # Write 10 frames: 5 STABLE, 3 WATCH, 2 ALERT
        for i in range(5):
            logger.write_raw_frame(
                {"asset_id": "asset_1", "state": "STABLE", "frame_index": i}
            )
        for i in range(5, 8):
            logger.write_raw_frame(
                {"asset_id": "asset_1", "state": "WATCH", "frame_index": i}
            )
        for i in range(8, 10):
            logger.write_raw_frame(
                {"asset_id": "asset_1", "state": "ALERT", "frame_index": i}
            )

        logger.close()

        report_gen = ShadowModeSummaryReport()
        report_gen.load_evidence(logger.log_path)
        report = report_gen.generate_summary()

        state_counts = report["state_distribution"]["state_counts"]
        assert state_counts["STABLE"] == 5
        assert state_counts["WATCH"] == 3
        assert state_counts["ALERT"] == 2

    def test_latency_statistics(self, temp_output_dir):
        """Report computes latency percentiles."""
        logger = ShadowModeEvidenceLogger(
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        # Write frames with varying latencies
        for i in range(100):
            logger.write_raw_frame(
                {
                    "asset_id": "asset_1",
                    "frame_index": i,
                    "processing_latency_ms": float(i),  # 0 to 99 ms
                }
            )

        logger.close()

        report_gen = ShadowModeSummaryReport()
        report_gen.load_evidence(logger.log_path)
        report = report_gen.generate_summary()

        latency_stats = report["latency"]
        assert latency_stats["samples"] == 100
        assert latency_stats["min_ms"] < 1.0
        assert latency_stats["max_ms"] > 90.0
        assert latency_stats["p95_ms"] > latency_stats["p50_ms"]

    def test_transitions_detected(self, temp_output_dir):
        """Report counts state transitions."""
        logger = ShadowModeEvidenceLogger(
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        states = ["STABLE", "STABLE", "WATCH", "WATCH", "ALERT", "ALERT", "STABLE"]
        for i, state in enumerate(states):
            logger.write_raw_frame(
                {
                    "asset_id": "asset_1",
                    "frame_index": i,
                    "state": state,
                    "transition_detected": state != states[i - 1] if i > 0 else False,
                }
            )

        logger.close()

        report_gen = ShadowModeSummaryReport()
        report_gen.load_evidence(logger.log_path)
        report = report_gen.generate_summary()

        # Should have transitions: STABLE->WATCH, WATCH->ALERT, ALERT->STABLE
        transitions = report["transitions"]["total_transitions"]
        assert transitions >= 2  # At least 2 transitions


# ============================================================================
# Tests: Shadow Mode Runner
# ============================================================================


class TestShadowModeRunner:
    """Test the main shadow-mode runner."""

    def test_runner_initialization(self, temp_output_dir, dummy_decision_fn):
        """Runner initializes correctly."""
        runner = ShadowModeRunner(
            decision_fn=dummy_decision_fn,
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        assert runner.frame_count == 0
        assert len(runner.errors) == 0

    def test_process_single_frame(self, temp_output_dir, dummy_decision_fn, sample_telemetry):
        """Runner processes a single frame."""
        runner = ShadowModeRunner(
            decision_fn=dummy_decision_fn,
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        runner.process_frame(sample_telemetry[0])
        assert runner.frame_count == 1
        assert len(runner.errors) == 0

    def test_process_telemetry_stream(
        self, temp_output_dir, dummy_decision_fn, sample_telemetry
    ):
        """Runner processes list of frames."""
        runner = ShadowModeRunner(
            decision_fn=dummy_decision_fn,
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        result = runner.process_telemetry_stream(sample_telemetry)
        assert result["frames_processed"] == len(sample_telemetry)
        assert result["errors"] == 0
        assert runner.frame_count == len(sample_telemetry)

    def test_process_telemetry_file(
        self, temp_output_dir, dummy_decision_fn, sample_telemetry
    ):
        """Runner processes JSONL file."""
        # Create temporary telemetry file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for frame in sample_telemetry:
                f.write(json.dumps(frame) + "\n")
            telemetry_path = f.name

        try:
            runner = ShadowModeRunner(
                decision_fn=dummy_decision_fn,
                output_dir=str(temp_output_dir),
                run_name="test_run",
            )

            result = runner.process_telemetry_file(telemetry_path)
            assert result["frames_processed"] == len(sample_telemetry)
            assert result["errors"] == 0

        finally:
            Path(telemetry_path).unlink()

    def test_finalize_generates_artifacts(
        self, temp_output_dir, dummy_decision_fn, sample_telemetry
    ):
        """Runner.finalize() generates all required artifacts."""
        runner = ShadowModeRunner(
            decision_fn=dummy_decision_fn,
            output_dir=str(temp_output_dir),
            run_name="test_run",
            engine_version="1.0",
        )

        runner.process_telemetry_stream(sample_telemetry)
        final = runner.finalize()

        assert final["success"] is True
        assert "evidence_log" in final["paths"]
        assert "summary_report" in final["paths"]
        assert final["summary_stats"]["total_frames"] == len(sample_telemetry)

        # Verify files exist
        assert Path(final["paths"]["evidence_log"]).exists()
        assert Path(final["paths"]["summary_report"]).exists()
        assert Path(final["paths"]["metadata_file"]).exists()

    def test_per_unit_isolation(self, temp_output_dir, dummy_decision_fn):
        """Runner maintains isolation per asset."""
        runner = ShadowModeRunner(
            decision_fn=dummy_decision_fn,
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        frames = [
            {"asset_id": "asset_1", "unit_id": "unit_a", "drift_signal": 0.1},
            {"asset_id": "asset_1", "unit_id": "unit_b", "drift_signal": 0.2},
            {"asset_id": "asset_2", "unit_id": "unit_c", "drift_signal": 0.3},
        ]

        runner.process_telemetry_stream(frames)
        final = runner.finalize()

        # Check that units are tracked
        assert final["summary_stats"]["unique_assets"] == 2
        assert "asset_1" in final["summary_stats"]
        assert "asset_2" in final["summary_stats"]

    def test_error_handling(self, temp_output_dir):
        """Runner handles errors gracefully."""

        def bad_decision_fn(frame):
            raise ValueError("Decision function failed")

        runner = ShadowModeRunner(
            decision_fn=bad_decision_fn,
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        frames = [{"asset_id": "asset_1"}]
        runner.process_telemetry_stream(frames)
        final = runner.finalize()

        assert len(runner.errors) > 0
        assert final["summary_stats"]["error_count"] > 0


# ============================================================================
# Tests: Replay Comparison
# ============================================================================


class TestReplayDiff:
    """Test replay comparison functionality."""

    def test_diff_identical_runs(self, temp_output_dir, dummy_decision_fn, sample_telemetry):
        """Diff detects that identical runs match perfectly."""
        # Create two identical runs
        for run_name in ["run_a", "run_b"]:
            runner = ShadowModeRunner(
                decision_fn=dummy_decision_fn,
                output_dir=str(temp_output_dir),
                run_name=run_name,
            )
            runner.process_telemetry_stream(sample_telemetry)
            runner.finalize()

        # Compare
        run_a_evidence = temp_output_dir / "run_a_evidence.jsonl"
        run_b_evidence = temp_output_dir / "run_b_evidence.jsonl"

        diff = ReplayDiff(run_a_evidence, run_b_evidence)
        report = diff.compare()

        assert report["metadata"]["frame_count"] == len(sample_telemetry)
        assert report["summary"]["frames_with_differences"] == 0

    def test_diff_different_states(self, temp_output_dir):
        """Diff detects state differences."""
        # Create two runs with different decisions
        def decision_fn_a(frame):
            return {
                "state": "STABLE",
                "structural_drift_score": 0.1,
            }

        def decision_fn_b(frame):
            return {
                "state": "WATCH",  # Different!
                "structural_drift_score": 0.1,
            }

        frames = [{"asset_id": "asset_1", "frame_index": i} for i in range(5)]

        for run_name, fn in [("run_a", decision_fn_a), ("run_b", decision_fn_b)]:
            runner = ShadowModeRunner(
                decision_fn=fn,
                output_dir=str(temp_output_dir),
                run_name=run_name,
            )
            runner.process_telemetry_stream(frames)
            runner.finalize()

        # Compare
        run_a_evidence = temp_output_dir / "run_a_evidence.jsonl"
        run_b_evidence = temp_output_dir / "run_b_evidence.jsonl"

        diff = ReplayDiff(run_a_evidence, run_b_evidence)
        report = diff.compare()

        assert report["summary"]["frames_with_differences"] == 5
        assert any(
            f["field"] == "state"
            for f in report["field_level_diff"]["critical_field_mismatches"]
        )


# ============================================================================
# Tests: CSV Export
# ============================================================================


class TestEvidenceExport:
    """Test evidence export to CSV."""

    def test_csv_export(self, temp_output_dir):
        """Evidence can be exported to CSV."""
        logger = ShadowModeEvidenceLogger(
            output_dir=str(temp_output_dir),
            run_name="test_run",
        )

        for i in range(10):
            logger.write_raw_frame(
                {
                    "asset_id": "asset_1",
                    "frame_index": i,
                    "state": "STABLE",
                    "drift_score": 0.1 * i,
                }
            )

        logger.close()

        csv_path = temp_output_dir / "evidence.csv"
        rows_written = EvidenceDataFrame.export_to_csv(logger.log_path, csv_path)

        assert rows_written == 10
        assert csv_path.exists()

        # Verify CSV content
        with open(csv_path) as f:
            lines = f.readlines()
            assert len(lines) == 11  # Header + 10 rows
            assert "asset_id" in lines[0]


# ============================================================================
# Integration Tests
# ============================================================================


class TestShadowModeIntegration:
    """Integration tests for full shadow-mode workflow."""

    def test_full_workflow(self, temp_output_dir, dummy_decision_fn, sample_telemetry):
        """Full workflow: process → evidence → report."""
        runner = ShadowModeRunner(
            decision_fn=dummy_decision_fn,
            output_dir=str(temp_output_dir),
            run_name="integration_test",
            engine_version="1.0",
            doctrine_version="2024-04",
        )

        runner.process_telemetry_stream(sample_telemetry)
        final = runner.finalize()

        # Verify all artifacts
        assert final["success"]
        evidence_path = Path(final["paths"]["evidence_log"])
        report_path = Path(final["paths"]["summary_report"])
        metadata_path = Path(final["paths"]["metadata_file"])

        assert evidence_path.exists()
        assert report_path.exists()
        assert metadata_path.exists()

        # Verify evidence content
        with open(evidence_path) as f:
            frames = [json.loads(line) for line in f if line.strip()]
            assert len(frames) == len(sample_telemetry)

        # Verify report content
        with open(report_path) as f:
            report = json.load(f)
            assert "overview" in report
            assert report["overview"]["total_frames"] == len(sample_telemetry)

        # Verify metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
            assert metadata["engine_version"] == "1.0"
            assert metadata["doctrine_version"] == "2024-04"
