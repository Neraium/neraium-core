from typing import Any, Dict, List, Optional

from neraium_core.engine import StructuralEngine
from neraium_core.ingest import normalize_rest_payload, parse_csv_text


class StructuralIntelligenceService:
    """Application service boundary around ingestion and engine execution."""

    def __init__(self, engine: Optional[StructuralEngine] = None) -> None:
        self.engine = engine or StructuralEngine()

    def ingest_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        frame = normalize_rest_payload(payload)
        return self.engine.process_frame(frame)

    def ingest_csv(self, csv_text: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for frame in parse_csv_text(csv_text):
            results.append(self.engine.process_frame(frame))
        return results

    def latest_result(self) -> Optional[Dict[str, Any]]:
        return self.engine.get_latest_result()

    def reset(self) -> None:
        self.engine.reset()
