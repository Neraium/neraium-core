from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from neraium_core.alignment import StructuralEngine
from neraium_core.pipeline import normalize_rest_payload, parse_csv_text
from neraium_core.store import ResultStore


logger = logging.getLogger(__name__)


class StructuralMonitoringService:
    """Service orchestration boundary for ingestion and persisted results."""

    def __init__(
        self,
        engine: StructuralEngine | None = None,
        store: ResultStore | None = None,
    ):
        # Template engine config; runtime engines are isolated per (site_id, asset_id)
        # so each asset gets its own baseline/model memory.
        self.engine = engine or StructuralEngine(baseline_window=24, recent_window=8)
        self.store = store or ResultStore()
        self._engines_by_asset: dict[tuple[str, str], StructuralEngine] = {}
        self._localization_by_site: dict[str, dict[str, float]] = {}

    def _engine_for_frame(self, frame: dict[str, Any]) -> StructuralEngine:
        site_id = str(frame.get("site_id", "default-site"))
        asset_id = str(frame.get("asset_id", "default-asset"))
        key = (site_id, asset_id)
        existing = self._engines_by_asset.get(key)
        if existing is not None:
            return existing

        # Reuse the initially provided engine for the first asset to preserve
        # backwards compatibility with tests/injection.
        if not self._engines_by_asset:
            self._engines_by_asset[key] = self.engine
            return self.engine

        # Clone config for additional assets with isolated regime memory file.
        template = self.engine
        regime_path = "regime_library.json"
        try:
            base_path = Path(template.regime_store.path)
            safe_site = site_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            safe_asset = asset_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            regime_path = str(base_path.with_name(f"{base_path.stem}_{safe_site}_{safe_asset}{base_path.suffix}"))
        except Exception:
            regime_path = f"regime_library_{site_id}_{asset_id}.json"

        new_engine = StructuralEngine(
            baseline_window=template.baseline_window,
            recent_window=template.recent_window,
            window_stride=template.window_stride,
            regime_store_path=regime_path,
            baseline_adaptation_alpha=template.baseline_adaptation_alpha,
        )
        self._engines_by_asset[key] = new_engine
        return new_engine

    def _localization_score(self, result: dict[str, Any]) -> float:
        site_id = str(result.get("site_id", "default-site"))
        asset_id = str(result.get("asset_id", "default-asset"))
        latest_instability = float(result.get("latest_instability", 0.0))
        site_map = self._localization_by_site.setdefault(site_id, {})
        site_map[asset_id] = max(0.0, latest_instability)
        total = sum(site_map.values())
        if total <= 1e-9:
            return 0.0
        share = latest_instability / total
        concentration = max(site_map.values()) / (total + 1e-9)
        return round(max(0.0, min(1.0, share * concentration * 2.0)), 4)

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

    def _operator_trend(self, result: dict[str, Any]) -> str:
        analytics = result.get("experimental_analytics")
        if not isinstance(analytics, dict):
            return "UNKNOWN"

        forecasting = analytics.get("forecasting")
        if not isinstance(forecasting, dict):
            return "UNKNOWN"

        trend_score = float(forecasting.get("trend", 0.0))
        if trend_score > 0.05:
            return "RISING"
        if trend_score < -0.05:
            return "FALLING"
        return "STABLE"

    def _operator_confidence(self, result: dict[str, Any]) -> float:
        # Prefer stabilized confidence_score from engine when present.
        score = result.get("confidence_score")
        if score is not None:
            try:
                return round(max(0.0, min(float(score), 1.0)), 4)
            except (TypeError, ValueError):
                pass
        stability = float(result.get("relational_stability_score", 0.0))
        return round(max(0.0, min(stability, 1.0)), 4)

    def _structural_analysis_metadata(self, result: dict[str, Any]) -> dict[str, Any]:
        signals = result.get("sensor_relationships")
        signal_count = len(signals) if isinstance(signals, list) else 0
        if signal_count < 2:
            return {
                "structural_analysis_available": False,
                "skipped_reason": "insufficient signal dimensionality",
            }

        analytics = result.get("experimental_analytics")
        if not isinstance(analytics, dict):
            return {
                "structural_analysis_available": False,
                "skipped_reason": "insufficient history",
            }

        if bool(analytics.get("relational_metrics_skipped")):
            return {
                "structural_analysis_available": False,
                "skipped_reason": "insufficient signal dimensionality",
            }

        return {
            "structural_analysis_available": True,
            "skipped_reason": None,
        }

    def _decorate_result(self, result: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(result)
        interpretation = self._interpret(result)
        structural = self._structural_analysis_metadata(result)
        localization_score = self._localization_score(result)

        enriched.update(interpretation)
        enriched.update(structural)
        enriched["trend"] = self._operator_trend(result)
        enriched["confidence"] = self._operator_confidence(result)
        enriched["localization_score"] = localization_score
        enriched["interpretation"] = {
            "heuristic": True,
            **interpretation,
            "trend": enriched["trend"],
            "confidence": enriched["confidence"],
            "localization_score": localization_score,
        }
        return enriched

    def ingest_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        logger.info(
            "ingest_payload called site_id=%s asset_id=%s",
            payload.get("site_id"),
            payload.get("asset_id"),
        )
        frame = normalize_rest_payload(payload)
        engine = self._engine_for_frame(frame)
        result = self._decorate_result(engine.process_frame(frame))
        self.store.save_ingestion(frame, result)
        return result

    def ingest_batch(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        results: list[dict[str, Any]] = []
        for payload in payloads:
            frame = normalize_rest_payload(payload)
            engine = self._engine_for_frame(frame)
            result = self._decorate_result(engine.process_frame(frame))
            results.append(result)
            pairs.append((frame, result))
        self.store.save_ingestion_batch(pairs)
        return results

    def ingest_csv(self, csv_text: str) -> list[dict[str, Any]]:
        frames = parse_csv_text(csv_text)
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        results: list[dict[str, Any]] = []
        for frame in frames:
            engine = self._engine_for_frame(frame)
            result = self._decorate_result(engine.process_frame(frame))
            results.append(result)
            pairs.append((frame, result))
        self.store.save_ingestion_batch(pairs)
        return results

    def get_latest_result(self) -> dict[str, Any] | None:
        return self.store.get_latest_result()

    def list_recent_results(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_recent_results(limit=limit)

    def reset(self) -> None:
        logger.info("reset called")
        self.engine = StructuralEngine(
            baseline_window=self.engine.baseline_window,
            recent_window=self.engine.recent_window,
            window_stride=self.engine.window_stride,
            baseline_adaptation_alpha=self.engine.baseline_adaptation_alpha,
        )
        self._engines_by_asset = {}
        self._localization_by_site = {}
        self.store.reset()
