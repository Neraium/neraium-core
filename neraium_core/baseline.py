from neraium_core.models import SystemDefinition

BASELINE_SYSTEM = SystemDefinition(
    system_id="baseline",
    inference_window_seconds=60,
    vector_order=[
        "cpu_usage",
        "memory_usage",
    ],
    max_missing_fraction=0.5,
)