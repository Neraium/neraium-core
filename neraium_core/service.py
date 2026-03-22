from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from neraium_core.alignment import StructuralEngine
from neraium_core.pipeline import normalize_rest_payload, parse_csv_text, pilot_hardening_enabled
from neraium_core.logging_utils import (
    Timer,
    log_structured,
    pilot_debug_enabled,
    summarize_exception_for_logs,
    summarize_payload_for_logs,
    summarize_result_for_logs,
)
from neraium_core.pilot_schema import build_pilot_output
from neraium_core.pilot_config import PilotConfig, load_pilot_config
from neraium_core.store import DEFAULT_CUSTOMER_ID, ResultStore


logger = logging.getLogger(__name__)


class StructuralMonitoringService:
    """Service orchestration boundary for ingestion and persisted results."""

    def __init__(
        self,
        engine: StructuralEngine | None = None,
        store: ResultStore | None = None,
        pilot_config: PilotConfig | None = None,
    ):
        # Template engine config; runtime engines are isolated per (site_id, asset_id)
        # so each asset gets its own baseline/model memory.
        self.engine = engine or StructuralEngine(baseline_window=24, recent_window=8)
        self.store = store or ResultStore()
        self._engines_by_asset: dict[tuple[str, str], StructuralEngine] = {}
        self._localization_by_site: dict[str, dict[str, float]] = {}
        self.pilot_config: PilotConfig = pilot_config or load_pilot_config()

    @staticmethod
    def _resolve_customer_id(customer_id: str | None) -> str:
        text = str(customer_id or "").strip()
        return text or DEFAULT_CUSTOMER_ID

    @classmethod
    def _effective_customer_id(
        cls,
        *,
        request_customer_id: str | None,
        payload_customer_id: str | None,
    ) -> str:
        request_text = str(request_customer_id or "").strip()
        payload_text = str(payload_customer_id or "").strip()
        if request_text:
            return cls._resolve_customer_id(request_text)
        if payload_text:
            return cls._resolve_customer_id(payload_text)
        return DEFAULT_CUSTOMER_ID

    def _engine_for_frame(self, frame: dict[str, Any]) -> StructuralEngine:
        customer_id = self._resolve_customer_id(frame.get("customer_id"))
        site_id = str(frame.get("site_id", "default-site"))
        asset_id = str(frame.get("asset_id", "default-asset"))
        key = (customer_id, site_id, asset_id)
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
            safe_customer = customer_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            safe_site = site_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            safe_asset = asset_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            regime_path = str(
                base_path.with_name(f"{base_path.stem}_{safe_customer}_{safe_site}_{safe_asset}{base_path.suffix}")
            )
        except Exception:
            regime_path = f"regime_library_{customer_id}_{site_id}_{asset_id}.json"

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

        if drift >= float(self.pilot_config.drift_high_threshold) or state == "ALERT":
            return {
                "risk_level": "HIGH",
                "action_state": "ALERT",
                "operator_message": "High instability detected. Immediate operator review advised.",
            }

        if drift >= float(self.pilot_config.drift_watch_threshold) or state == "WATCH":
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

    @staticmethod
    def _with_result_metadata(
        result: dict[str, Any],
        *,
        customer_id: str | None,
        run_id: str | None,
        result_id: int | None,
        persisted_at: str | None,
    ) -> dict[str, Any]:
        enriched = dict(result)
        if customer_id is not None:
            enriched["customer_id"] = customer_id
        if run_id is not None:
            enriched["run_id"] = run_id
        if result_id is not None:
            enriched["result_id"] = int(result_id)
        if persisted_at is not None:
            enriched["persisted_at"] = persisted_at
        return enriched

    def ingest_payload(
        self,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest a single telemetry payload and return the decorated result.

        When `NERAIUM_PILOT_HARDENING=1`, the response is augmented with the pilot
        schema keys: `timestamp, signals, score, status, aligned, anomaly`.
        """

        timer = Timer()
        log_structured(
            logger,
            event="ingest_payload_in",
            fields=summarize_payload_for_logs(payload),
            level=logging.INFO,
        )
        try:
            frame = normalize_rest_payload(payload)
            resolved_customer = self._effective_customer_id(
                request_customer_id=customer_id,
                payload_customer_id=frame.get("customer_id"),
            )
            frame["customer_id"] = resolved_customer
            engine = self._engine_for_frame(frame)
            result = self._decorate_result(engine.process_frame(frame))
            result["customer_id"] = resolved_customer
            if pilot_hardening_enabled():
                result.update(build_pilot_output(frame=frame, result=result))
            self.store.save_ingestion(frame, result, run_id=run_id, customer_id=resolved_customer)
            persisted = self.store.get_latest_result(run_id=run_id, customer_id=resolved_customer)
            if persisted is not None:
                result = self._with_result_metadata(
                    result,
                    customer_id=persisted.get("customer_id"),
                    run_id=persisted.get("run_id"),
                    result_id=persisted.get("result_id"),
                    persisted_at=persisted.get("persisted_at"),
                )

            out_fields = summarize_result_for_logs(result)
            out_fields["latency_ms"] = round(timer.ms(), 3)
            log_structured(logger, event="ingest_payload_out", fields=out_fields, level=logging.INFO)

            if out_fields.get("gate_passed") is False:
                log_structured(
                    logger,
                    event="data_quality_gate_failed",
                    fields=out_fields,
                    level=logging.WARNING,
                )

            if pilot_debug_enabled():
                # Debug-only: provide a slightly richer reason block without including raw signals.
                dq = result.get("data_quality_summary")
                if isinstance(dq, dict) and dq.get("statuses"):
                    log_structured(
                        logger,
                        event="data_quality_statuses_debug",
                        fields={
                            "statuses": [str(s) for s in dq.get("statuses", [])[:10]],
                            "missingness_rate": dq.get("missingness_rate"),
                            "variability_coverage": dq.get("variability_coverage"),
                        },
                        level=logging.DEBUG,
                    )

            return result
        except Exception as exc:
            err_fields = {
                **summarize_payload_for_logs(payload),
                **summarize_exception_for_logs(exc),
                "latency_ms": round(timer.ms(), 3),
            }
            log_structured(logger, event="ingest_payload_error", fields=err_fields, level=logging.ERROR)
            raise

    def ingest_batch(
        self,
        payloads: list[dict[str, Any]],
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ingest multiple telemetry payloads (batch) and return decorated results.

        When `NERAIUM_PILOT_HARDENING=1`, each result is augmented with pilot schema keys.
        """

        batch_timer = Timer()
        log_structured(
            logger,
            event="ingest_batch_in",
            fields={"items": len(payloads)},
            level=logging.INFO,
        )
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        results: list[dict[str, Any]] = []
        resolved_customer = self._effective_customer_id(
            request_customer_id=customer_id,
            payload_customer_id=payloads[0].get("customer_id") if payloads else None,
        )
        for payload in payloads:
            item_timer = Timer()
            try:
                frame = normalize_rest_payload(payload)
                frame_customer = self._effective_customer_id(
                    request_customer_id=customer_id,
                    payload_customer_id=frame.get("customer_id"),
                )
                frame["customer_id"] = frame_customer
                if frame_customer != resolved_customer:
                    raise ValueError("All batch items must share the same customer_id")
                engine = self._engine_for_frame(frame)
                result = self._decorate_result(engine.process_frame(frame))
                result["customer_id"] = resolved_customer
                if pilot_hardening_enabled():
                    result.update(build_pilot_output(frame=frame, result=result))
                results.append(result)
                pairs.append((frame, result))

                out_fields = summarize_result_for_logs(result)
                out_fields["latency_ms"] = round(item_timer.ms(), 3)
                out_fields["item_asset_id"] = payload.get("asset_id")
                log_structured(
                    logger,
                    event="ingest_batch_item_out",
                    fields=out_fields,
                    level=logging.INFO,
                )
            except Exception as exc:
                err_fields = {
                    **summarize_payload_for_logs(payload),
                    **summarize_exception_for_logs(exc),
                    "latency_ms": round(item_timer.ms(), 3),
                }
                log_structured(logger, event="ingest_batch_item_error", fields=err_fields, level=logging.ERROR)
                raise

        self.store.save_ingestion_batch(pairs, run_id=run_id, customer_id=resolved_customer)
        persisted_recent = self.store.list_recent_results(
            limit=len(results),
            run_id=run_id,
            customer_id=resolved_customer,
        )
        persisted_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for item in persisted_recent:
            key = (
                str(item.get("customer_id", "")),
                str(item.get("timestamp", "")),
                str(item.get("site_id", "")),
                str(item.get("asset_id", "")),
            )
            persisted_map[key] = item
        results_with_ids: list[dict[str, Any]] = []
        for result in results:
            key = (
                str(result.get("customer_id", "")),
                str(result.get("timestamp", "")),
                str(result.get("site_id", "")),
                str(result.get("asset_id", "")),
            )
            persisted = persisted_map.get(key)
            if persisted is None:
                results_with_ids.append(
                    self._with_result_metadata(
                        result,
                        customer_id=resolved_customer,
                        run_id=run_id,
                        result_id=None,
                        persisted_at=None,
                    )
                )
            else:
                results_with_ids.append(
                    self._with_result_metadata(
                        result,
                        customer_id=persisted.get("customer_id"),
                        run_id=persisted.get("run_id"),
                        result_id=persisted.get("result_id"),
                        persisted_at=persisted.get("persisted_at"),
                    )
                )
        log_structured(
            logger,
            event="ingest_batch_out",
            fields={"items": len(results), "latency_ms": round(batch_timer.ms(), 3)},
            level=logging.INFO,
        )
        return results_with_ids

    def ingest_csv(
        self,
        csv_text: str,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ingest CSV text and return decorated results.

        When `NERAIUM_PILOT_HARDENING=1`, each result is augmented with pilot schema keys.
        """

        timer = Timer()
        log_structured(
            logger,
            event="ingest_csv_in",
            fields={"csv_text_len": len(csv_text)},
            level=logging.INFO,
        )
        try:
            resolved_customer = self._resolve_customer_id(customer_id)
            frames = parse_csv_text(csv_text, customer_id=resolved_customer)
            pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            results: list[dict[str, Any]] = []
            for frame in frames:
                item_timer = Timer()
                engine = self._engine_for_frame(frame)
                result = self._decorate_result(engine.process_frame(frame))
                result["customer_id"] = resolved_customer
                if pilot_hardening_enabled():
                    result.update(build_pilot_output(frame=frame, result=result))
                results.append(result)
                pairs.append((frame, result))

                out_fields = summarize_result_for_logs(result)
                out_fields["latency_ms"] = round(item_timer.ms(), 3)
                out_fields["frame_asset_id"] = frame.get("asset_id")
                log_structured(
                    logger,
                    event="ingest_csv_item_out",
                    fields=out_fields,
                    level=logging.INFO,
                )

            self.store.save_ingestion_batch(pairs, run_id=run_id, customer_id=resolved_customer)
            persisted_recent = self.store.list_recent_results(
                limit=len(results),
                run_id=run_id,
                customer_id=resolved_customer,
            )
            persisted_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
            for item in persisted_recent:
                key = (
                    str(item.get("customer_id", "")),
                    str(item.get("timestamp", "")),
                    str(item.get("site_id", "")),
                    str(item.get("asset_id", "")),
                )
                persisted_map[key] = item
            results_with_ids: list[dict[str, Any]] = []
            for result in results:
                key = (
                    str(result.get("customer_id", "")),
                    str(result.get("timestamp", "")),
                    str(result.get("site_id", "")),
                    str(result.get("asset_id", "")),
                )
                persisted = persisted_map.get(key)
                if persisted is None:
                    results_with_ids.append(
                        self._with_result_metadata(
                            result,
                            customer_id=resolved_customer,
                            run_id=run_id,
                            result_id=None,
                            persisted_at=None,
                        )
                    )
                else:
                    results_with_ids.append(
                        self._with_result_metadata(
                            result,
                            customer_id=persisted.get("customer_id"),
                            run_id=persisted.get("run_id"),
                            result_id=persisted.get("result_id"),
                            persisted_at=persisted.get("persisted_at"),
                        )
                    )
            log_structured(
                logger,
                event="ingest_csv_out",
                fields={"items": len(results), "latency_ms": round(timer.ms(), 3)},
                level=logging.INFO,
            )
            return results_with_ids
        except Exception as exc:
            err_fields = {
                "csv_text_len": len(csv_text),
                **summarize_exception_for_logs(exc),
                "latency_ms": round(timer.ms(), 3),
            }
            log_structured(logger, event="ingest_csv_error", fields=err_fields, level=logging.ERROR)
            raise

    def get_latest_result(
        self,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
        site_id: str | None = None,
    ) -> dict[str, Any] | None:
        if site_id is None or not str(site_id).strip():
            return self.store.get_latest_result(run_id=run_id, customer_id=customer_id)
        results = self.list_recent_results(
            limit=1000,
            run_id=run_id,
            customer_id=customer_id,
            site_id=site_id,
        )
        return results[0] if results else None

    def list_recent_results(
        self,
        limit: int = 100,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
        site_id: str | None = None,
    ) -> list[dict[str, Any]]:
        items = self.store.list_recent_results(limit=limit, run_id=run_id, customer_id=customer_id)
        if site_id is None or not str(site_id).strip():
            return items
        target = str(site_id).strip()
        return [item for item in items if str(item.get("site_id", "")).strip() == target]

    def get_result_by_id(
        self,
        result_id: int,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self.store.get_result_by_id(result_id, run_id=run_id, customer_id=customer_id)

    def create_run(
        self,
        *,
        name: str,
        config: dict[str, Any] | None = None,
        activate: bool = True,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        return self.store.create_run(name=name, config=config, activate=activate, customer_id=customer_id)

    def list_runs(self, *, limit: int = 100, customer_id: str | None = None) -> list[dict[str, Any]]:
        return self.store.list_runs(limit=limit, customer_id=customer_id)

    def get_run(self, run_id: str, *, customer_id: str | None = None) -> dict[str, Any] | None:
        return self.store.get_run(run_id, customer_id=customer_id)

    def get_active_run(self, *, customer_id: str | None = None) -> dict[str, Any] | None:
        return self.store.get_active_run(customer_id=customer_id)

    def activate_run(self, run_id: str, *, customer_id: str | None = None) -> dict[str, Any]:
        return self.store.activate_run(run_id, customer_id=customer_id)

    def update_run(
        self,
        run_id: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        status: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        return self.store.update_run(
            run_id,
            name=name,
            config=config,
            status=status,
            customer_id=customer_id,
        )

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
