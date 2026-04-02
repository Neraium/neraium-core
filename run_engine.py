from __future__ import annotations

from datetime import datetime, timezone

from neraium_core.alignment import StructuralEngine
from neraium_core.pipeline import TelemetryPipeline
from neraium_core.telemetry import TelemetryPayload


def main() -> None:
    pipeline = TelemetryPipeline()

    payload = TelemetryPayload(
        timestamp=datetime.now(timezone.utc),
        signals={
            "cpu_usage": 42.5,
            "memory_usage": 71.2,
        },
    )

    result = pipeline.process(payload)

    print("Neraium Telemetry Result:")
    print(result)


if __name__ == "__main__":
    main()
