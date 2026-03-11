import json
from datetime import datetime

from neraium_core.models import TelemetryPayload


def load_replay_file(path: str) -> list[TelemetryPayload]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    payloads = []
    for row in rows:
        payloads.append(
            TelemetryPayload(
                system_id=row["system_id"],
                timestamp=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
                signals=row["signals"],
            )
        )
    return payloads