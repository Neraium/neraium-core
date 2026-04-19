"""Tests for agnostic runner."""

import json
import tempfile
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from neraium_core.agnostic_runner import AgnosticRunner, EngineConfig, RunMetrics
from neraium_core.agnostic_runner.adapters.base import DataSourceAdapter, Frame, OutputSink, Result
from neraium_core.agnostic_runner.adapters.csv_source import CsvDataSource
from neraium_core.agnostic_runner.adapters.json_sink import JsonFileSink, JsonLineSink
from neraium_core.agnostic_runner.adapters.registry import AdapterRegistry, get_registry


class MockDataSource(DataSourceAdapter):
    """Mock data source for testing."""

    def __init__(self, frames: list[Frame] | None = None) -> None:
        super().__init__()
        self.frames = frames or [
            Frame(
                timestamp=1.0,
                unit_id="test_unit",
                sensors={"temp": 25.0, "pressure": 100.0},
            ),
        ]
        self.read_count = 0

    def validate(self) -> bool:
        return True

    def read_frames(self) -> Iterator[Frame]:
        self.read_count += 1
        for frame in self.frames:
            yield frame


class MockOutputSink(OutputSink):
    """Mock output sink for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[Result] = []
        self.finalized = False

    def write_result(self, result: Result) -> None:
        self.results.append(result)
        self.result_count += 1

    def finalize(self) -> dict:
        self.finalized = True
        return {"record_count": len(self.results)}


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_production_preset(self) -> None:
        config = EngineConfig.production()
        assert config.mode == "production"
        assert config.enable_advisory_layers is True
        assert config.enable_experimental_layers is False

    def test_research_preset(self) -> None:
        config = EngineConfig.research()
        assert config.mode == "research_assistive"
        assert config.enable_law_extraction is True
        assert config.enable_experimental_layers is False

    def test_experimental_preset(self) -> None:
        config = EngineConfig.experimental()
        assert config.mode == "full"
        assert config.enable_experimental_layers is True

    def test_minimal_preset(self) -> None:
        config = EngineConfig.minimal()
        assert config.mode == "production"
        assert config.enable_trajectory_memory is False
        assert config.enable_advisory_layers is False

    def test_from_dict(self) -> None:
        data = {
            "mode": "research_assistive",
            "enable_law_engine": True,
            "enable_experimental_layers": False,
        }
        config = EngineConfig.from_dict(data)
        assert config.mode == "research_assistive"
        assert config.enable_law_engine is True

    def test_to_component_flags(self) -> None:
        config = EngineConfig.production()
        flags = config.to_component_flags()
        assert isinstance(flags, dict)
        assert "trajectory_intelligence" in flags
        assert "reliability_calibration" in flags


class TestAdapterRegistry:
    """Tests for adapter registry."""

    def test_register_source(self) -> None:
        registry = AdapterRegistry()
        registry.register_source("mock", MockDataSource)
        assert "mock" in registry.list_sources()

    def test_register_sink(self) -> None:
        registry = AdapterRegistry()
        registry.register_sink("mock", MockOutputSink)
        assert "mock" in registry.list_sinks()

    def test_create_source(self) -> None:
        registry = AdapterRegistry()
        registry.register_source("mock", MockDataSource)
        source = registry.create_source("mock", {})
        assert isinstance(source, MockDataSource)

    def test_create_sink(self) -> None:
        registry = AdapterRegistry()
        registry.register_sink("mock", MockOutputSink)
        sink = registry.create_sink("mock", {})
        assert isinstance(sink, MockOutputSink)

    def test_unknown_source(self) -> None:
        registry = AdapterRegistry()
        with pytest.raises(ValueError, match="Unknown data source"):
            registry.get_source("nonexistent")

    def test_global_registry(self) -> None:
        registry = get_registry()
        assert isinstance(registry, AdapterRegistry)
        # Should have built-in adapters
        assert "csv" in registry.list_sources()
        assert "json" in registry.list_sinks()


class TestCsvDataSource:
    """Tests for CSV data source."""

    def test_read_csv(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,unit_id,sensor_temp,sensor_pressure\n")
            f.write("1.0,unit1,25.0,100.0\n")
            f.write("2.0,unit1,26.0,101.0\n")
            f.flush()
            csv_path = f.name

        try:
            source = CsvDataSource({"path": csv_path})
            frames = list(source.read_frames())

            assert len(frames) == 2
            assert frames[0].timestamp == 1.0
            assert frames[0].unit_id == "unit1"
            assert frames[0].sensors["temp"] == 25.0
            assert frames[0].sensors["pressure"] == 100.0
        finally:
            Path(csv_path).unlink()

    def test_csv_validation(self) -> None:
        source = CsvDataSource({"path": "/nonexistent/file.csv"})
        assert source.validate() is False

    def test_csv_custom_columns(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ts,equipment_id,s_temp,s_pressure\n")
            f.write("1.0,eq1,25.0,100.0\n")
            f.flush()
            csv_path = f.name

        try:
            source = CsvDataSource({
                "path": csv_path,
                "timestamp_col": "ts",
                "unit_id_col": "equipment_id",
                "sensor_prefix": "s_",
            })
            frames = list(source.read_frames())

            assert len(frames) == 1
            assert frames[0].unit_id == "eq1"
        finally:
            Path(csv_path).unlink()


class TestJsonSinks:
    """Tests for JSON output sinks."""

    def test_json_file_sink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            sink = JsonFileSink({"path": str(output_path)})

            frame = Frame(timestamp=1.0, unit_id="test", sensors={"temp": 25.0})
            result = Result(frame=frame, output={"state": "STABLE"})

            sink.write_result(result)
            summary = sink.finalize()

            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["unit_id"] == "test"
            assert summary["record_count"] == 1

    def test_jsonl_sink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.jsonl"
            sink = JsonLineSink({"path": str(output_path)})

            frame = Frame(timestamp=1.0, unit_id="test", sensors={"temp": 25.0})
            result = Result(frame=frame, output={"state": "STABLE"})

            sink.write_result(result)
            sink.finalize()

            assert output_path.exists()
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["unit_id"] == "test"


class TestAgnosticRunner:
    """Tests for agnostic runner."""

    def test_run_basic(self) -> None:
        """Test basic end-to-end execution."""
        frames = [
            Frame(timestamp=1.0, unit_id="unit1", sensors={"temp": 25.0}),
            Frame(timestamp=2.0, unit_id="unit1", sensors={"temp": 26.0}),
        ]
        source = MockDataSource(frames=frames)
        sink = MockOutputSink()

        runner = AgnosticRunner(
            data_source=source,
            output_sink=sink,
            engine_config=EngineConfig.production(),
        )

        # Mock the engine update to avoid needing full engine initialization
        runner.platform.update = MagicMock(return_value={"state": "STABLE", "risk_level": "LOW"})

        metrics = runner.run()

        assert metrics.frame_count == 2
        assert metrics.error_count == 0
        assert metrics.status == "success"
        assert sink.finalized is True
        assert len(sink.results) == 2

    def test_run_with_errors(self) -> None:
        """Test error handling."""
        frames = [
            Frame(timestamp=1.0, unit_id="unit1", sensors={"temp": 25.0}),
            Frame(timestamp=2.0, unit_id="unit1", sensors={"temp": 26.0}),
        ]
        source = MockDataSource(frames=frames)
        sink = MockOutputSink()

        runner = AgnosticRunner(
            data_source=source,
            output_sink=sink,
            engine_config=EngineConfig.production(),
            error_handler="log",
        )

        # Mock engine to raise error on first frame
        call_count = [0]

        def mock_update(data):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Test error")
            return {"state": "STABLE"}

        runner.platform.update = mock_update

        metrics = runner.run()

        assert metrics.frame_count == 1  # Only second frame processed
        assert metrics.error_count == 1
        assert metrics.status == "partial"

    def test_run_metrics(self) -> None:
        """Test metrics calculation."""
        frames = [Frame(timestamp=float(i), unit_id="unit1", sensors={"temp": 25.0 + i}) for i in range(10)]
        source = MockDataSource(frames=frames)
        sink = MockOutputSink()

        runner = AgnosticRunner(
            data_source=source,
            output_sink=sink,
            engine_config=EngineConfig.production(),
        )

        runner.platform.update = MagicMock(return_value={"state": "STABLE"})
        metrics = runner.run()

        assert metrics.frame_count == 10
        assert metrics.frames_per_second > 0
        assert metrics.elapsed_seconds > 0
        assert metrics.source_type == "MockDataSource"
        assert metrics.sink_type == "MockOutputSink"

    def test_run_custom_id(self) -> None:
        """Test custom run ID."""
        source = MockDataSource()
        sink = MockOutputSink()
        custom_id = "my-run-123"

        runner = AgnosticRunner(
            data_source=source,
            output_sink=sink,
            run_id=custom_id,
        )
        runner.platform.update = MagicMock(return_value={"state": "STABLE"})
        metrics = runner.run()

        assert metrics.run_id == custom_id

    def test_invalid_source(self) -> None:
        """Test handling of invalid data source."""
        source = MockDataSource()
        source.validate = MagicMock(return_value=False)
        sink = MockOutputSink()

        runner = AgnosticRunner(
            data_source=source,
            output_sink=sink,
        )

        metrics = runner.run()
        assert metrics.status == "failed"

    def test_invalid_sink(self) -> None:
        """Test handling of invalid output sink."""
        source = MockDataSource()
        sink = MockOutputSink()
        sink.validate = MagicMock(return_value=False)

        runner = AgnosticRunner(
            data_source=source,
            output_sink=sink,
        )

        metrics = runner.run()
        assert metrics.status == "failed"


class TestRunMetrics:
    """Tests for run metrics."""

    def test_metrics_to_dict(self) -> None:
        metrics = RunMetrics(
            run_id="test-run",
            status="success",
            frame_count=100,
            error_count=0,
            elapsed_seconds=5.0,
            frames_per_second=20.0,
            source_type="CsvDataSource",
            sink_type="JsonFileSink",
            engine_mode="production",
            summary={"output_path": "results.json"},
        )
        result_dict = metrics.to_dict()

        assert result_dict["run_id"] == "test-run"
        assert result_dict["frame_count"] == 100
        assert result_dict["frames_per_second"] == 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
