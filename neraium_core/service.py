from __future__ import annotations

from typing import Any, Dict, List, Optional

from neraium_core.engine import StructuralEngine
from neraium_core.ingest import normalize_rest_payload, parse_csv_text


class StructuralIntelligenceService:
    def __init__(self, engine: Optional[StructuralEngine] = None):
        self.engine = engine or StructuralEngine()

    def ingest_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        frame = normalize_rest_payload(payload, include_sensor_quality=True)
        return self.engine.process_frame(frame)

    def ingest_csv(self, csv_text: str) -> List[Dict[str, Any]]:
        frames = parse_csv_text(csv_text, include_sensor_quality=True)
        return [self.engine.process_frame(frame) for frame in frames]

    def latest_result(self) -> Optional[Dict[str, Any]]:
        return self.engine.get_latest_result()

    def reset(self) -> None:
        self.engine.reset()
