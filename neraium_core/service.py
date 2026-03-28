from __future__ import annotations

import csv
import io
import logging
import os
from pathlib import Path
from typing import Any, Callable, TextIO

from neraium_core.alignment import StructuralEngine
from neraium_core.adapters.raw_telemetry_adapter import ingest_raw_industrial_input
from neraium_core.csv_mapping import row_to_frame_kwargs, resolve_mapping
from neraium_core.pipeline import (
    build_frame,
    normalize_rest_payload,
    parse_csv_text,
    pilot_hardening_enabled,
)
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
from neraium_core.output_contract import build_canonical_output
from neraium_core.store import DEFAULT_CUSTOMER_ID, ResultStore


logger = logging.getLogger(__name__)


class StructuralMonitoringService:
    """Service orchestration boundary for ingestion and persisted results."""

    def __init__(
        self,
        engine: StructuralEngine | None = None,
        store: ResultStore | None = None,
        pilot_config: PilotConfig | None = None,
        *,
        frame_debug: bool = False,
    ):
        # Template engine config; runtime engines are isolated per (site_id, asset_id)
        # so each asset gets its own baseline/model memory.
        self.engine = engine or StructuralEngine(baseline_window=24, recent_window=8, frame_debug=frame_debug)
        self.store = store or ResultStore()
        self._engines_by_asset: dict[tuple[str, str, str], StructuralEngine] = {}
        self._localization_by_site: dict[str, dict[str, float]] = {}
        self._cycle_by_run: dict[tuple[str, str], int] = {}
        self.pilot_config: PilotConfig = pilot_config or load_pilot_config()

    def _next_cycle(self, *, run_id: str | None, customer_id: str) -> int:
        key = (customer_id, str(run_id or "__default__"))
        next_cycle = self._cycle_by_run.get(key, 0) + 1
        self._cycle_by_run[key] = next_cycle
        return next_cycle

    def _persist_product_history(
        self,
        result: dict[str, Any],
        *,
        run_id: str | None,
        customer_id: str,
    ) -> dict[str, Any]:
        previous = self.store.get_latest_service_history(run_id=run_id, customer_id=customer_id)
        cycle = self._next_cycle(run_id=run_id, customer_id=customer_id)
        canonical = build_canonical_output(
            result,
            cycle=cycle,
            run_id=run_id,
            customer_id=customer_id,
            previous=previous,
        )
        persisted = self.store.save_service_history(canonical, run_id=run_id, customer_id=customer_id)
        canonical["history_id"] = persisted["history_id"]
        canonical["persisted_at"] = persisted["persisted_at"]
        return canonical

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

    def _engine_for_frame(
        self,
        frame: dict[str, Any],
        *,
        run_config: dict[str, Any] | None = None,
    ) -> StructuralEngine:
        customer_id = self._resolve_customer_id(frame.get("customer_id"))
        site_id = str(frame.get("site_id", "default-site"))
        asset_id = str(frame.get("asset_id", "default-asset"))
        key = (customer_id, site_id, asset_id)
        existing = self._engines_by_asset.get(key)
        if existing is not None:
            return existing

        template = self.engine
        baseline_window = int(run_config.get("baseline_window", template.baseline_window)) if run_config else template.baseline_window
        recent_window = int(run_config.get("recent_window", template.recent_window)) if run_config else template.recent_window
        baseline_window = max(4, min(baseline_window, 500))
        recent_window = max(2, min(recent_window, 120))

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

        frame_debug = bool((run_config or {}).get("frame_debug", getattr(template, "_frame_debug", False)))
        new_engine = StructuralEngine(
            baseline_window=baseline_window,
            recent_window=recent_window,
            window_stride=template.window_stride,
            regime_store_path=regime_path,
            baseline_adaptation_alpha=template.baseline_adaptation_alpha,
            representation_mode=(run_config or {}).get("representation_mode", template.representation_config.mode),
            reference_strategy=(run_config or {}).get("reference_strategy", template.representation_config.reference_strategy),
            context_diagnostics_enabled=bool((run_config or {}).get("context_diagnostics_enabled", template.representation_config.enable_diagnostics)),
            feature_weights=(run_config or {}).get(
                "feature_weights",
                {
                    "raw_weight": template.representation_config.weights.raw_weight,
                    "residual_weight": template.representation_config.weights.residual_weight,
                    "delta_weight": template.representation_config.weights.delta_weight,
                    "slope_weight": template.representation_config.weights.slope_weight,
                    "drift_weight": template.representation_config.weights.drift_weight,
                    "second_diff_weight": template.representation_config.weights.second_diff_weight,
                },
            ),
            frame_debug=frame_debug,
        )
        if run_config and run_config.get("baseline_locked") is True:
            new_engine.lock_baseline(True)
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
        transition_pressure = float(result.get("transition_pressure", 0.0))
        transition_state = str(result.get("transition_state", "NONE")).upper()
        state = str(result.get("state", "STABLE")).upper()
        transition_aware_enabled = str(os.environ.get("NERAIUM_TRANSITION_AWARE", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

        transition_actionable = result.get("transition_outputs_actionable")
        if transition_actionable is None:
            transition_actionable = result.get("engine_transition_detectable")
        if transition_actionable is None and isinstance(result.get("readiness"), dict):
            rd0 = result["readiness"]
            transition_actionable = rd0.get("transition_classification_ready")
            if transition_actionable is None:
                transition_actionable = rd0.get("transition_classifiable")
        if transition_actionable is None:
            transition_actionable = transition_state != "WARMUP"

        if (
            drift >= float(self.pilot_config.drift_high_threshold)
            or state == "ALERT"
            or (
                transition_aware_enabled
                and transition_actionable is not False
                and transition_state != "WARMUP"
                and (transition_state == "SUSTAINED_TRANSITION" or transition_pressure >= 1.15)
            )
        ):
            return {
                "risk_level": "HIGH",
                "action_state": "ALERT",
                "operator_message": (
                    "High instability/transition pressure detected. "
                    "System is structurally departing from prior stable behavior; immediate operator review advised."
                ),
            }

        if (
            drift >= float(self.pilot_config.drift_watch_threshold)
            or state == "WATCH"
            or (
                transition_aware_enabled
                and transition_actionable is not False
                and transition_state != "WARMUP"
                and (transition_state == "EMERGING_TRANSITION" or transition_pressure >= 0.85)
            )
        ):
            return {
                "risk_level": "MEDIUM",
                "action_state": "WATCH",
                "operator_message": (
                    "Transition pressure is elevated. "
                    "Monitor closely for continued structural departure from recent stable behavior."
                ),
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

    def ingest_raw_telemetry_windows(
        self,
        input_path: str,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
        site_id: str = "raw-telemetry",
        preprocessing_mode: str = "auto",
        sample_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Backward-compatible wrapper for generic raw industrial ingestion."""
        output = self.ingest_raw_industrial_data(
            input_path,
            run_id=run_id,
            customer_id=customer_id,
            site_id=site_id,
            preprocessing_mode=preprocessing_mode,
            sample_size=sample_size,
        )
        return list(output["results"])

    def ingest_raw_industrial_data(
        self,
        input_path: str,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
        site_id: str = "raw-telemetry",
        preprocessing_mode: str = "auto",
        sample_size: int | None = None,
    ) -> dict[str, Any]:
        """Ingest raw industrial data (tabular rows or directory signal blocks) through runtime path."""
        resolved_customer = self._resolve_customer_id(customer_id)
        ingestion = ingest_raw_industrial_input(
            input_path,
            site_id=site_id,
            customer_id=resolved_customer,
            preprocessing_mode=preprocessing_mode,
            sample_size=sample_size,
        )
        outputs: list[dict[str, Any]] = []
        for frame in ingestion.frames:
            outputs.append(self.ingest_payload(frame, run_id=run_id, customer_id=customer_id))
        diagnostics = {
            "detected_input_type": ingestion.diagnostics.detected_input_type,
            "timestep_count": ingestion.diagnostics.timestep_count,
            "sensor_count": ingestion.diagnostics.sensor_count,
            "preprocessing_mode": ingestion.diagnostics.preprocessing_mode,
            "has_valid_signals": ingestion.diagnostics.has_valid_signals,
            "sensor_columns": ingestion.diagnostics.sensor_columns,
            "metadata_columns": ingestion.diagnostics.metadata_columns,
            "warnings": ingestion.diagnostics.warnings,
        }
        return {"diagnostics": diagnostics, "results": outputs}

    def ingest_payload(
        self,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
        log_ingest: bool = True,
    ) -> dict[str, Any]:
        """Ingest a single telemetry payload and return the decorated result.

        When `NERAIUM_PILOT_HARDENING=1`, the response is augmented with the pilot
        schema keys: `timestamp, signals, score, status, aligned, anomaly`.

        Set ``log_ingest=False`` for bulk/offline runs to skip per-frame structured
        INFO logs (JSON serialization of ingest metadata on every cycle).
        """

        timer = Timer()
        if log_ingest:
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
            run_config: dict[str, Any] | None = None
            if run_id:
                run_obj = self.store.get_run(run_id, customer_id=resolved_customer)
                if run_obj and isinstance(run_obj.get("config"), dict):
                    run_config = run_obj["config"]
            engine = self._engine_for_frame(frame, run_config=run_config)
            result = self._decorate_result(engine.process_frame(frame))
            result["customer_id"] = resolved_customer
            if pilot_hardening_enabled():
                result.update(build_pilot_output(frame=frame, result=result))
            persisted = self.store.save_ingestion(
                frame, result, run_id=run_id, customer_id=resolved_customer
            )
            result = self._with_result_metadata(
                result,
                customer_id=persisted.get("customer_id"),
                run_id=persisted.get("run_id"),
                result_id=persisted.get("result_id"),
                persisted_at=persisted.get("persisted_at"),
            )
            result["canonical_output"] = self._persist_product_history(
                result,
                run_id=run_id,
                customer_id=resolved_customer,
            )
            result["events"] = list(result["canonical_output"].get("events", []))

            out_fields = summarize_result_for_logs(result)
            out_fields["latency_ms"] = round(timer.ms(), 3)
            if log_ingest:
                log_structured(logger, event="ingest_payload_out", fields=out_fields, level=logging.INFO)

            if log_ingest and out_fields.get("gate_passed") is False:
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
                run_config = None
                if run_id:
                    run_obj = self.store.get_run(run_id, customer_id=resolved_customer)
                    if run_obj and isinstance(run_obj.get("config"), dict):
                        run_config = run_obj["config"]
                engine = self._engine_for_frame(frame, run_config=run_config)
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
        for result in results_with_ids:
            result["canonical_output"] = self._persist_product_history(
                result,
                run_id=run_id,
                customer_id=resolved_customer,
            )
            result["events"] = list(result["canonical_output"].get("events", []))
        log_structured(
            logger,
            event="ingest_batch_out",
            fields={"items": len(results), "latency_ms": round(batch_timer.ms(), 3)},
            level=logging.INFO,
        )
        return results_with_ids

    def ingest_frame(
        self,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        result = self.ingest_payload(payload, run_id=run_id, customer_id=customer_id)
        canonical = result.get("canonical_output")
        if isinstance(canonical, dict):
            return canonical
        resolved_customer = self._resolve_customer_id(customer_id or result.get("customer_id"))
        return self._persist_product_history(result, run_id=run_id, customer_id=resolved_customer)

    def ingest_csv(
        self,
        csv_text: str,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
        column_mapping: dict[str, Any] | None = None,
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
            run_config: dict[str, Any] | None = None
            if run_id:
                run_obj = self.store.get_run(run_id, customer_id=resolved_customer)
                if run_obj and isinstance(run_obj.get("config"), dict):
                    run_config = run_obj["config"]
            frames = parse_csv_text(
                csv_text,
                customer_id=resolved_customer,
                column_mapping=column_mapping,
            )
            pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            results: list[dict[str, Any]] = []
            for frame in frames:
                item_timer = Timer()
                engine = self._engine_for_frame(frame, run_config=run_config)
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
            for result in results_with_ids:
                result["canonical_output"] = self._persist_product_history(
                    result,
                    run_id=run_id,
                    customer_id=resolved_customer,
                )
                result["events"] = list(result["canonical_output"].get("events", []))
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

    @staticmethod
    def _csv_validation_error_message(exc: Exception, *, row_index: int | None = None) -> str:
        raw = str(exc)
        if "Invalid timestamp" in raw:
            base = (
                "Invalid timestamp format. Use ISO-8601 (for example "
                "2026-01-01T00:00:00+00:00)."
            )
        elif "Sensor name cannot be empty" in raw:
            base = "CSV includes an empty sensor column name. Ensure all sensor headers are non-empty."
        elif "Invalid signal value for" in raw:
            base = f"Non-numeric sensor value detected. {raw}"
        elif "Invalid signal type for" in raw:
            base = f"Unsupported sensor value type detected. {raw}"
        else:
            base = raw
        if row_index is None:
            return base
        return f"Row {row_index}: {base}"

    def ingest_csv_stream(
        self,
        csv_stream: TextIO,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
        column_mapping: dict[str, Any] | None = None,
        batch_size: int = 250,
        max_error_samples: int = 25,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Stream-ingest CSV rows with partial success support and progress callbacks.

        This path is intended for large-file ingestion where loading the full CSV into
        memory is undesirable.
        """

        timer = Timer()
        safe_batch_size = max(1, int(batch_size))
        safe_error_samples = max(1, int(max_error_samples))
        resolved_customer = self._resolve_customer_id(customer_id)

        def _emit_progress(
            *,
            status: str,
            rows_processed: int,
            rows_succeeded: int,
            rows_failed: int,
            error_samples: list[dict[str, Any]] | None = None,
            message: str | None = None,
        ) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(
                    {
                        "status": status,
                        "rows_processed": int(rows_processed),
                        "rows_succeeded": int(rows_succeeded),
                        "rows_failed": int(rows_failed),
                        "error_samples": list(error_samples or []),
                        "message": message,
                    }
                )
            except Exception:
                # Progress callbacks must never fail ingestion flow.
                logger.debug("ingest_csv_stream progress callback failure", exc_info=True)

        log_structured(
            logger,
            event="ingest_csv_stream_in",
            fields={
                "run_id": run_id,
                "customer_id": resolved_customer,
                "batch_size": safe_batch_size,
            },
            level=logging.INFO,
        )

        run_config: dict[str, Any] | None = None
        if run_id:
            run_obj = self.store.get_run(run_id, customer_id=resolved_customer)
            if run_obj and isinstance(run_obj.get("config"), dict):
                run_config = run_obj["config"]

        try:
            pos = csv_stream.tell()
        except (OSError, io.UnsupportedOperation):
            pos = None

        if pos is None:
            blob = csv_stream.read()
            csv_stream = io.StringIO(blob)
            pos = 0

        first_line = csv_stream.readline()
        if not first_line or not str(first_line).strip():
            raise ValueError(
                "CSV file is empty. Add a header row with your time column, entity/asset column, "
                "optional site/location column, and one or more numeric telemetry columns."
            )

        sample_for_infer = first_line + csv_stream.read(65536)
        csv_stream.seek(pos)

        reader = csv.DictReader(csv_stream)
        if reader.fieldnames is None:
            raise ValueError("CSV has no parseable header row.")

        header_names = [str(name).strip() for name in reader.fieldnames if name is not None]
        try:
            mapping, _map_warnings = resolve_mapping(
                header_names,
                column_mapping,
                csv_sample=sample_for_infer if column_mapping is None else None,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        buffered_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        rows_processed = 0
        rows_succeeded = 0
        rows_failed = 0
        error_samples: list[dict[str, Any]] = []
        last_result: dict[str, Any] | None = None

        _emit_progress(
            status="processing",
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            error_samples=[],
            message="CSV accepted. Processing rows...",
        )

        for row_index, row in enumerate(reader, start=2):
            if row is None:
                continue
            rows_processed += 1
            try:
                kwargs = row_to_frame_kwargs(row, mapping, customer_id=resolved_customer)
                frame = build_frame(**kwargs)
                frame["customer_id"] = resolved_customer
                engine = self._engine_for_frame(frame, run_config=run_config)
                result = self._decorate_result(engine.process_frame(frame))
                result["customer_id"] = resolved_customer
                if pilot_hardening_enabled():
                    result.update(build_pilot_output(frame=frame, result=result))
                buffered_pairs.append((frame, result))
                rows_succeeded += 1
                result["canonical_output"] = self._persist_product_history(
                    result,
                    run_id=run_id,
                    customer_id=resolved_customer,
                )
                result["events"] = list(result["canonical_output"].get("events", []))
                last_result = result
            except Exception as exc:
                rows_failed += 1
                msg = self._csv_validation_error_message(exc, row_index=row_index)
                if len(error_samples) < safe_error_samples:
                    error_samples.append({"row": row_index, "message": msg})
                log_structured(
                    logger,
                    event="ingest_csv_stream_row_error",
                    fields={
                        "row_index": row_index,
                        "run_id": run_id,
                        "customer_id": resolved_customer,
                        **summarize_exception_for_logs(exc),
                    },
                    level=logging.WARNING,
                )

            if len(buffered_pairs) >= safe_batch_size:
                self.store.save_ingestion_batch(
                    buffered_pairs,
                    run_id=run_id,
                    customer_id=resolved_customer,
                )
                buffered_pairs = []

            if rows_processed % 25 == 0:
                _emit_progress(
                    status="processing",
                    rows_processed=rows_processed,
                    rows_succeeded=rows_succeeded,
                    rows_failed=rows_failed,
                    error_samples=error_samples,
                    message=(
                        f"Processed {rows_processed} rows "
                        f"({rows_succeeded} succeeded, {rows_failed} failed)."
                    ),
                )

        if buffered_pairs:
            self.store.save_ingestion_batch(
                buffered_pairs,
                run_id=run_id,
                customer_id=resolved_customer,
            )

        persisted_latest = self.store.get_latest_result(run_id=run_id, customer_id=resolved_customer)
        latest_result: dict[str, Any] | None = None
        if persisted_latest is not None and last_result is not None:
            latest_result = self._with_result_metadata(
                last_result,
                customer_id=persisted_latest.get("customer_id"),
                run_id=persisted_latest.get("run_id"),
                result_id=persisted_latest.get("result_id"),
                persisted_at=persisted_latest.get("persisted_at"),
            )
        elif persisted_latest is not None:
            latest_result = persisted_latest

        partial_success = rows_succeeded > 0 and rows_failed > 0
        message = (
            f"Processed {rows_processed} rows: "
            f"{rows_succeeded} succeeded, {rows_failed} failed."
        )
        if partial_success:
            message += " Ingest completed with partial success."
        elif rows_failed > 0 and rows_succeeded == 0:
            message += " No rows were successfully ingested."

        summary = {
            "run_id": run_id,
            "customer_id": resolved_customer,
            "rows_processed": rows_processed,
            "rows_succeeded": rows_succeeded,
            "rows_failed": rows_failed,
            "partial_success": partial_success,
            "error_samples": error_samples,
            "latest_result": latest_result,
            "message": message,
            "latency_ms": round(timer.ms(), 3),
        }

        log_level = logging.INFO
        if partial_success:
            log_level = logging.WARNING
        if rows_failed > 0 and rows_succeeded == 0:
            log_level = logging.ERROR

        log_structured(
            logger,
            event="ingest_csv_stream_out",
            fields={
                "run_id": run_id,
                "customer_id": resolved_customer,
                "rows_processed": rows_processed,
                "rows_succeeded": rows_succeeded,
                "rows_failed": rows_failed,
                "partial_success": partial_success,
                "latency_ms": summary["latency_ms"],
            },
            level=log_level,
        )

        _emit_progress(
            status="completed",
            rows_processed=rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            error_samples=error_samples,
            message=message,
        )
        return summary

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

    def get_current_state(
        self,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self.store.get_latest_service_history(run_id=run_id, customer_id=customer_id)

    def get_recent_history(
        self,
        limit: int = 100,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.store.list_service_history(limit=limit, run_id=run_id, customer_id=customer_id)

    def get_latest_decision(
        self,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any] | None:
        latest = self.get_current_state(run_id=run_id, customer_id=customer_id)
        if not isinstance(latest, dict):
            return None
        decision = latest.get("decision")
        return decision if isinstance(decision, dict) else None

    def get_latest_explanation(
        self,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> str | None:
        latest = self.get_current_state(run_id=run_id, customer_id=customer_id)
        if not isinstance(latest, dict):
            return None
        text = latest.get("explanation_text")
        return str(text) if text is not None else None

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

    def _engine_keys_for_run(
        self,
        run_id: str,
        customer_id: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """Return (customer_id, site_id, asset_id) keys for engines used by this run."""
        resolved_customer = self._resolve_customer_id(customer_id)
        results = self.store.list_recent_results(limit=1000, run_id=run_id, customer_id=resolved_customer)
        seen: set[tuple[str, str, str]] = set()
        keys: list[tuple[str, str, str]] = []
        for r in results:
            site_id = str(r.get("site_id") or "default-site")
            asset_id = str(r.get("asset_id") or "default-asset")
            key = (resolved_customer, site_id, asset_id)
            if key not in seen:
                seen.add(key)
                keys.append(key)
        return keys if keys else [(resolved_customer, "default-site", "default-asset")]

    def reset_baseline_for_run(
        self,
        run_id: str,
        customer_id: str | None = None,
    ) -> None:
        """Reset baseline and calibration for all engines used by this run."""
        for key in self._engine_keys_for_run(run_id, customer_id):
            engine = self._engines_by_asset.get(key)
            if engine is not None:
                engine.reset_baseline()
                log_structured(
                    logger,
                    event="baseline_reset",
                    fields={"run_id": run_id, "site_id": key[1], "asset_id": key[2]},
                    level=logging.INFO,
                )

    def lock_baseline_for_run(
        self,
        run_id: str,
        locked: bool,
        customer_id: str | None = None,
    ) -> None:
        """Lock or unlock baseline for this run. Updates run config and existing engines."""
        resolved_customer = self._resolve_customer_id(customer_id)
        run_obj = self.store.get_run(run_id, customer_id=resolved_customer)
        if run_obj is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        config = dict(run_obj.get("config") or {})
        config["baseline_locked"] = bool(locked)
        self.store.update_run(run_id, config=config, customer_id=resolved_customer)
        for key in self._engine_keys_for_run(run_id, customer_id):
            engine = self._engines_by_asset.get(key)
            if engine is not None:
                engine.lock_baseline(locked)

    def get_baseline_info_for_run(
        self,
        run_id: str,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        """Return baseline metadata for this run (from engines or run config)."""
        resolved_customer = self._resolve_customer_id(customer_id)
        run_obj = self.store.get_run(run_id, customer_id=resolved_customer)
        if run_obj is None:
            return {
                "baseline_set_at": None,
                "baseline_coverage_samples": 0,
                "baseline_window_config": 24,
                "baseline_locked": False,
                "baseline_mode": "unknown",
            }
        config = run_obj.get("config") or {}
        keys = self._engine_keys_for_run(run_id, customer_id)
        # Use first engine's info if available
        for key in keys:
            engine = self._engines_by_asset.get(key)
            if engine is not None:
                info = dict(engine.get_baseline_info())
                info["baseline_locked"] = config.get("baseline_locked", info.get("baseline_locked", False))
                return info
        return {
            "baseline_set_at": None,
            "baseline_coverage_samples": 0,
            "baseline_window_config": int(config.get("baseline_window", 24)),
            "baseline_locked": bool(config.get("baseline_locked", False)),
            "baseline_mode": "unknown",
        }
