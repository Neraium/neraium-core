from neraium_core.models import SystemDefinition, SignalDefinition

# Verify the import worked
print(type(SignalDefinition))  # Should be <class 'pydantic.main.ModelMetaclass'>

BASELINE_SYSTEM = SystemDefinition(
    schema_version="1",  # Already a string ✓
    system_id="baseline",
    signals=[
        SignalDefinition(name="cpu_usage", unit="percent"),
        SignalDefinition(name="memory_usage", unit="percent"),
    ],
    inference_window_seconds=60,
    raw_sample_period_seconds=5,
    vector_order=["cpu_usage", "memory_usage"],
    max_forward_fill_windows=3,
    max_missing_signal_fraction=0.5,
)
