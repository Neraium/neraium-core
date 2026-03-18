from __future__ import annotations
import json
from pathlib import Path
from typing import Any


class RegimeStore:
    def __init__(self, path: str = "regime_library.json"):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"regimes": [], "baselines": {}}
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {"regimes": [], "baselines": {}}

    def save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2))