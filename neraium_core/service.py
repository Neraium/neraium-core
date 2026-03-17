from __future__ import annotations

from typing import Any

from neraium_core.engine import StructuralEngine
from neraium_core.ingest import normalize_rest_payload, parse_csv_text
from neraium_core.models import IngestResponse, IngestSummary, StructuralResult, TelemetryFrame, ValidationIssue


class StructuralIntelligenceService:
    """Service layer that separates ingestion from analytics orchestration."""

    def __init__(self, engine: StructuralEngine | None = None) -> None:
        self.engine = engine or StructuralEngine()

    @staticmethod
    def _frame_to_engine_payload(frame: TelemetryFrame) -> dict[str, Any]:
        return {
            "timestamp": frame.timestamp,
            "site_id": frame.site_id,
            "asset_id": frame.asset_id,
            "sensor_values": {
                key: (float("nan") if value is None else value)
                for key, value in frame.sensor_values.items()
            },
        }

    def ingest_frame(self, frame_or_payload: TelemetryFrame | dict[str, Any]) -> StructuralResult:
        frame = frame_or_payload if isinstance(frame_or_payload, TelemetryFrame) else normalize_rest_payload(frame_or_payload)
        return self.engine.process_frame(self._frame_to_engine_payload(frame))

    def ingest_csv(self, csv_text: str) -> IngestResponse:
        issues: list[ValidationIssue] = []
        results: list[StructuralResult] = []
        frames = parse_csv_text(csv_text)

        for idx, frame in enumerate(frames, start=2):
            try:
                results.append(self.ingest_frame(frame))
            except ValueError as exc:
                issues.append(ValidationIssue(row=idx, field="frame", message=str(exc)))

        return IngestResponse(
            summary=IngestSummary(accepted=len(results), rejected=len(issues), issues=issues),
            results=results,
        )

    def latest_result(self) -> StructuralResult | None:
        return self.engine.get_latest_result()

    def reset(self) -> None:
        self.engine.reset()
