from __future__ import annotations

import os
from typing import Any

from neraium_core.explanation_layer import build_memory_context_text
from neraium_core.output_contract import build_canonical_output
from neraium_core.pilot_config import PilotConfig


class CanonicalOutputProjector:
    """Pure-ish result projection and canonical output shaping boundary."""

    def __init__(self, pilot_config: PilotConfig) -> None:
        self.pilot_config = pilot_config
        self._localization_by_site: dict[str, dict[str, float]] = {}

    def project_for_output(self, result: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(result)
        interpretation = self.interpret(result)
        structural = self.structural_analysis_metadata(result)
        localization_score = self.localization_score(result)

        enriched.update(interpretation)
        enriched.update(structural)
        enriched["trend"] = self.operator_trend(result)
        enriched["confidence"] = self.operator_confidence(result)
        enriched["localization_score"] = localization_score
        enriched["interpretation"] = {
            "heuristic": True,
            **interpretation,
            "trend": enriched["trend"],
            "confidence": enriched["confidence"],
            "localization_score": localization_score,
        }
        return enriched

    def project_canonical_output(
        self,
        result: dict[str, Any],
        *,
        cycle: int,
        run_id: str | None,
        customer_id: str,
        previous: dict[str, Any] | None,
        memory_recall: dict[str, Any] | None,
        alert_control: dict[str, Any] | None = None,
        alert_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result_for_projection = dict(result)
        if alert_control:
            result_for_projection["alert_control"] = dict(alert_control)
        if alert_policy:
            result_for_projection["alert_policy"] = dict(alert_policy)
        canonical = build_canonical_output(
            result_for_projection,
            cycle=cycle,
            run_id=run_id,
            customer_id=customer_id,
            previous=previous,
            memory_recall=memory_recall,
        )
        memory_context = build_memory_context_text(canonical.get("memory_recall"))
        if memory_context:
            base_explanation = str(canonical.get("explanation_text", "")).strip()
            if memory_context not in base_explanation:
                canonical["explanation_text"] = f"{base_explanation} {memory_context}".strip()
        return canonical

    def localization_score(self, result: dict[str, Any]) -> float:
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

    def interpret(self, result: dict[str, Any]) -> dict[str, str]:
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

    @staticmethod
    def operator_trend(result: dict[str, Any]) -> str:
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

    @staticmethod
    def operator_confidence(result: dict[str, Any]) -> float:
        score = result.get("confidence_score")
        if score is not None:
            try:
                return round(max(0.0, min(float(score), 1.0)), 4)
            except (TypeError, ValueError):
                pass
        stability = float(result.get("relational_stability_score", 0.0))
        return round(max(0.0, min(stability, 1.0)), 4)

    @staticmethod
    def structural_analysis_metadata(result: dict[str, Any]) -> dict[str, Any]:
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
