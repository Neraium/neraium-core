from neraium_core.models import SystemDefinition, SignalDefinition

BASELINE_SYSTEM = SystemDefinition(
    schema_version="1",
    system_id="baseline",
    signals=[
        SignalDefinition(
            name="cpu_usage",
            dtype="float64",
            unit="percent",
            required_for_scoring=True,
        ),
        SignalDefinition(
            name="memory_usage",
            dtype="float64",
            unit="percent",
            required_for_scoring=True,
        ),
    ],
    inference_window_seconds=60,
    raw_sample_period_seconds=5,
    vector_order=[
        "cpu_usage",
        "memory_usage",
    ],
    max_forward_fill_windows=3,
    max_missing_signal_fraction=0.5,
    
)