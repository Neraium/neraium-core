"""Evidence logging utilities for surfaced market signals."""

from __future__ import annotations

import json
from pathlib import Path

from neraium_core.markets.evidence.review_schema import EvidenceRecord, SignalOutput


class EvidenceLog:
    """Append-only JSONL evidence vault."""

    def __init__(self, path: str | Path = "artifacts/neraium_markets/evidence.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: EvidenceRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")

    def latest(self, limit: int = 20) -> list[SignalOutput]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        return [
            EvidenceRecord.model_validate(json.loads(line)).signal
            for line in lines[-limit:]
            if line.strip()
        ]

    def by_asset(self, asset: str, limit: int = 100) -> list[SignalOutput]:
        out: list[SignalOutput] = []
        if not self.path.exists():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = EvidenceRecord.model_validate(json.loads(line))
            if record.signal.asset == asset:
                out.append(record.signal)
        return out[-limit:]

    def get_evidence(self, signal_id: str) -> EvidenceRecord | None:
        if not self.path.exists():
            return None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = EvidenceRecord.model_validate(json.loads(line))
            if record.signal.signal_id == signal_id:
                return record
        return None
