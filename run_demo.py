from __future__ import annotations

import os

import uvicorn

from apps.api.main import DEFAULT_GROW_ENGINE, app
from neraium_core.grow.validation import load_records_from_path


def _seed_demo() -> None:
    demo_path = os.environ.get("NERAIUM_GROW_DEMO_PATH", "examples/grow/demo_telemetry.json")
    if not os.path.exists(demo_path):
        return
    if DEFAULT_GROW_ENGINE.list_states("room"):
        return
    DEFAULT_GROW_ENGINE.ingest_records(load_records_from_path(demo_path))


if __name__ == "__main__":
    _seed_demo()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
