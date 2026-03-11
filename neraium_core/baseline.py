from neraium_core.models import SystemDefinition

BASELINE_SYSTEM = SystemDefinition(
    schema_version=1,
    system_id="baseline",

    signals=[
        "cpu_usage",
        "memory_usage",
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