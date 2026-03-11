from neraium_core.models import SystemDefinition


BASELINE_SYSTEM = SystemDefinition(
    system_id="baseline",

    # size of the alignment window
    inference_window_seconds=60,

    # signals that will appear in the vector
    vector_order=[
        "cpu_usage",
        "memory_usage",
    ],

    # how much missing data is acceptable
    max_missing_fraction=0.5,
)