from neraium_core.models import SystemDefinition


BASELINE_SYSTEM = SystemDefinition(
    schema_version=1,
    system_id="baseline",

    # telemetry signals used by the system
    signals=[
        "cpu_usage",
        "memory_usage",
    ],

    # alignment window size
    inference_window_seconds=60,

    # raw telemetry sampling rate
    raw_sample_period_seconds=5,

    # order of vector features
    vector_order=[
        "cpu_usage",
        "memory_usage",
    ],

    # how far forward we fill missing values
    max_forward_fill_windows=3,

    # maximum missing signal tolerance
    max_missing_signal_fraction=0.5,
)