from neraium_core.models import SystemDefinition, SignalDefinition

# Verify the import worked
print(type(SignalDefinition))  # Should be <class 'pydantic.main.ModelMetaclass'>

BASELINE_SYSTEM = SystemDefinition(
    schema_version="1",  # Already a string ✓
    system_id="baseline",
    signals=[
    SignalDefinition(
        name="cpu_usage",
        dtype="float",
        unit="percent",
        required_for_scoring=True,
    ),
    SignalDefinition(
        name="memory_usage",
        dtype="float",
        unit="percent",
        required_for_scoring=True,
    ),
],
    inference_window_seconds=60,
    raw_sample_period_seconds=5,
    vector_order=["cpu_usage", "memory_usage"],
    max_forward_fill_windows=3,
    max_missing_signal_fraction=0.5,
)
from pydantic import BaseModel, field_validator


class BaselineDefinition(BaseModel):
    baseline_id: str
    system_id: str
    schema_version: str
    signal_order: list[str]
    mean_vector: list[float]
    covariance_matrix: list[list[float]]
    sample_count: int

    @field_validator("covariance_matrix")
    @classmethod
    def validate_covariance_square(cls, value):
        n = len(value)
        for row in value:
            if len(row) != n:
                raise ValueError("covariance_matrix must be square")
        return value


DEMO_BASELINE = BaselineDefinition(
    baseline_id="demo-baseline",
    system_id="baseline",
    schema_version="1",
    signal_order=["cpu_usage", "memory_usage"],
    mean_vector=[20.0, 45.0],
    covariance_matrix=[
        [4.0, 0.5],
        [0.5, 4.0]
    ],
    sample_count=100
)