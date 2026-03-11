from datetime import datetime, timezone

from neraium_core.baseline import BASELINE_SYSTEM, DEMO_BASELINE
from neraium_core.models import TelemetryPayload
from neraium_core.pipeline import NeraiumPipeline


def test_pipeline_constructs():
    pipeline = NeraiumPipeline(BASELINE_SYSTEM, DEMO_BASELINE)
    assert pipeline.system_definition.system_id == "baseline"


def test_pipeline_mismatch_raises():
    bad = DEMO_BASELINE.model_copy(update={"system_id": "wrong"})
    try:
        NeraiumPipeline(BASELINE_SYSTEM, bad)
        assert False
    except ValueError:
        assert True


def test_ingest_returns_list():
    pipeline = NeraiumPipeline(BASELINE_SYSTEM, DEMO_BASELINE)
    payload = TelemetryPayload(
        system_id="baseline",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        signals={"cpu_usage": 20.0, "memory_usage": 45.0},
    )
    results = pipeline.ingest(payload)
    assert isinstance(results, list)