from __future__ import annotations

from typing import Any

from neraium_core.alignment import StructuralEngine
from neraium_core.models import StructuralResult
from neraium_core.pipeline import normalize_rest_payload, parse_csv_text


class StructuralIntelligenceService:
    def __init__(self, baseline_window: int = 50, recent_window: int = 12):
        self._engine = StructuralEngine(
            baseline_window=baseline_window,
            recent_window=recent_window,
        )
        self._latest_result: StructuralResult | None = None

    def ingest_payload(self, payload: dict[str, Any]) -> StructuralResult:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary")

        frame = normalize_rest_payload(payload)

        if not frame.get("sensor_values"):
            raise ValueError("payload must include at least one sensor value")

        metrics = self._engine.process_frame(frame)
        self._latest_result = StructuralResult.model_validate(metrics)
        return self._latest_result

    def ingest_batch(self, list_of_payloads: list[dict[str, Any]]) -> list[StructuralResult]:
        if not isinstance(list_of_payloads, list):
            raise ValueError("list_of_payloads must be a list")

        return [self.ingest_payload(payload) for payload in list_of_payloads]

    def ingest_csv(self, csv_text: str) -> list[StructuralResult]:
        frames = parse_csv_text(csv_text)

        results: list[StructuralResult] = []
        for frame in frames:
            if not frame.get("sensor_values"):
                continue
            metrics = self._engine.process_frame(frame)
            result = StructuralResult.model_validate(metrics)
            self._latest_result = result
            results.append(result)

        return results

    def latest_result(self) -> StructuralResult | None:
        return self._latest_result

    def reset(self) -> None:
        self._engine.reset()
        self._latest_result = None
