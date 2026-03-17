from __future__ import annotations

from typing import Any

from neraium_core.alignment import StructuralEngine
from neraium_core.pipeline import normalize_rest_payload, parse_csv_text
from neraium_core.store import ResultStore


class StructuralMonitoringService:
    """Service orchestration boundary for ingestion and persisted results."""

    def __init__(
        self,
        engine: StructuralEngine | None = None,
        store: ResultStore | None = None,
    ):
        self.engine = engine or StructuralEngine(baseline_window=24, recent_window=8)
        self.store = store or ResultStore()

    def _interpret(self, result: dict[str, Any]) -> dict[str, str]:
        drift = float(result.get("structural_drift_score", 0.0))
        state = str(result.get("state", "STABLE")).upper()

        if drift >= 3.0 or state == "ALERT":
            return {
                "risk_level": "HIGH",
                "action_state": "ALERT",
                "operator_message": "High instability detected. Immediate operator review advised.",
            }

        if drift >= 1.5 or state == "WATCH":
            return {
                "risk_level": "MEDIUM",
                "action_state": "WATCH",
                "operator_message": "Drift is elevated. Monitor closely for trend continuation.",
            }

        return {
            "risk_level": "LOW",
            "action_state": "STABLE",
            "operator_message": "System appears stable based on current heuristic interpretation.",
        }

    def _decorate_result(self, result: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(result)
        enriched["interpretation"] = {
            "heuristic": True,
            **self._interpret(result),
        }
        return enriched

    def ingest_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = normalize_rest_payload(payload)
        result = self._decorate_result(self.engine.process_frame(frame))
        self.store.save_result(result)
        self.store.save_event(frame, result)
        return result

    def ingest_batch(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.ingest_payload(payload) for payload in payloads]

    def ingest_csv(self, csv_text: str) -> list[dict[str, Any]]:
        frames = parse_csv_text(csv_text)
        return [self._ingest_normalized(frame) for frame in frames]

    def _ingest_normalized(self, frame: dict[str, Any]) -> dict[str, Any]:
        result = self._decorate_result(self.engine.process_frame(frame))
        self.store.save_result(result)
        self.store.save_event(frame, result)
        return result

    def get_latest_result(self) -> dict[str, Any] | None:
        return self.store.get_latest_result()

    def list_recent_results(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_recent_results(limit=limit)

    def reset(self) -> None:
        self.engine = StructuralEngine(
            baseline_window=self.engine.baseline_window,
            recent_window=self.engine.recent_window,
        )
        self.store.reset()
