from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AdmittedEvent:
    timestamp: str
    drift: float
    stability: float
    regime: str
    event_admitted: bool
    evidence_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
