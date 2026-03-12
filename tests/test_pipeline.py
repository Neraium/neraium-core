
from datetime import datetime

from neraium_core.pipeline import TelemetryPipeline
from neraium_core.telemetry import TelemetryPayload


class DummyEngine:
    def __init__(self, system_definition):
        self.system_definition = system_definition
        self.last_signals = None

    def align(self, signals):
        self.last_signals = signals
        return [signals["cpu_usage"], signals["memory_usage"]]


def test_pipeline_constructs(monkeypatch):
    monkeypatch.setattr("neraium_core.pipeline.AlignmentEngine", DummyEngine)

    pipeline = TelemetryPipeline()

    assert pipeline is not None
    assert pipeline.engine is not None


def test_process_returns_aligned_vector(monkeypatch):
    monkeypatch.setattr("neraium_core.pipeline.AlignmentEngine", DummyEngine)

    pipeline = TelemetryPipeline()

    payload = TelemetryPayload(
        timestamp=datetime.utcnow(),
        signals={
            "cpu_usage": 20.0,
            "memory_usage": 55.0,
        },
    )

    result = pipeline.process(payload)

    assert result["aligned"] == [20.0, 55.0]
    assert "score" in result
    assert result["status"] in ["normal", "anomaly"]
    assert "anomaly" in result


def test_process_passes_signals_to_engine(monkeypatch):
    monkeypatch.setattr("neraium_core.pipeline.AlignmentEngine", DummyEngine)

    pipeline = TelemetryPipeline()

    payload = TelemetryPayload(
        timestamp=datetime.utcnow(),
        signals={
            "cpu_usage": 33.3,
            "memory_usage": 66.6,
        },
    )

    pipeline.process(payload)

    assert pipeline.engine.last_signals == {
        "cpu_usage": 33.3,
        "memory_usage": 66.6,
    }
