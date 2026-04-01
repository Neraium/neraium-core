from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from validation.drift import DriftDetector
from validation.feedback import FeedbackIntegrator
from validation.metrics import compute_backtest_metrics
from validation.outcomes import OutcomeAttributor
from validation.replay import HistoricalReplayEngine

CALIBRATION_METRIC_DOC = {
    "canonical_metric": "calibration_error",
    "definition": "mean_absolute_error between decision confidence and binary outcome correctness",
    "direction": "lower_is_better",
    "quality_transform": "calibration_quality = 1 - calibration_error (bounded to [0,1])",
}


class RealWorldValidationPipeline:
    def __init__(self, decision_fn) -> None:
        self.replay = HistoricalReplayEngine(decision_fn=decision_fn)
        self.outcomes = OutcomeAttributor()
        self.feedback = FeedbackIntegrator()
        self.drift = DriftDetector()

    def run(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        replay_result = self.replay.replay(records)
        step_logs = list(replay_result["step_logs"])
        outcomes = self.outcomes.attribute(step_logs)
        feedback_records = self.feedback.infer_feedback(step_logs, outcomes)
        metrics = compute_backtest_metrics(step_logs, outcomes)
        drift = self.drift.detect(metrics.get("per_step", []))
        core_validation = self._build_core_validation_artifact(
            step_logs=step_logs,
            outcomes=outcomes,
            metrics=metrics,
            drift=drift,
        )

        return {
            "replay": replay_result,
            "outcomes": outcomes,
            "feedback": [f.to_dict() for f in feedback_records],
            "feedback_summary": self.feedback.summarize(feedback_records),
            "metrics": metrics,
            "core_validation_report": core_validation,
            "real_world_validation": {
                "decision_accuracy": metrics["decision_accuracy"],
                "harm_rate": metrics["harm_rate"],
                "calibration_error": metrics["calibration_error"],
                "calibration_quality": metrics["calibration_quality"],
                "law_validation": {},
                "drift_signals": drift,
            },
        }

    @staticmethod
    def _safe_float(value: Any, fallback: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @classmethod
    def _build_corpus_summary(cls, step_logs: list[dict[str, Any]]) -> dict[str, Any]:
        domains = Counter()
        assets = Counter()
        scenarios = Counter()
        system_types = Counter()
        timestamps: list[float] = []

        for row in step_logs:
            domain = str(row.get("domain") or "unknown")
            asset = str(row.get("asset_id") or "unknown")
            scenario = str(row.get("scenario_family") or "unknown")
            system_type = str(row.get("system_type") or "unknown")
            domains[domain] += 1
            assets[asset] += 1
            scenarios[scenario] += 1
            system_types[system_type] += 1
            timestamps.append(cls._safe_float(row.get("timestamp"), fallback=0.0))

        time_range = None
        if timestamps:
            time_range = {"min_timestamp": min(timestamps), "max_timestamp": max(timestamps)}

        total = max(1, len(step_logs))
        dominant_asset_count = max(assets.values()) if assets else 0
        dominant_domain_count = max(domains.values()) if domains else 0
        dominant_asset_share = dominant_asset_count / total
        dominant_domain_share = dominant_domain_count / total
        warnings: list[str] = []
        if dominant_asset_share >= 0.55:
            warnings.append("dominant_asset_skew")
        if len(domains) < 2:
            warnings.append("weak_domain_diversity")
        if any(count < 5 for count in assets.values()):
            warnings.append("low_cohort_support")

        return {
            "total_records": len(step_logs),
            "domain_count": len(domains),
            "asset_count": len(assets),
            "scenario_family_count": len(scenarios),
            "system_type_count": len(system_types),
            "by_domain": dict(sorted(domains.items())),
            "by_asset": dict(sorted(assets.items())),
            "by_scenario_family": dict(sorted(scenarios.items())),
            "by_system_type": dict(sorted(system_types.items())),
            "time_range": time_range,
            "representativeness": {
                "dominant_asset_share": round(dominant_asset_share, 6),
                "dominant_domain_share": round(dominant_domain_share, 6),
                "warnings": warnings,
            },
        }

    @classmethod
    def _cohort_metrics(cls, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "count": 0,
                "decision_accuracy": 0.0,
                "harm_rate": 0.0,
                "calibration_error": 1.0,
                "calibration_quality": 0.0,
                "false_confidence_rate": 0.0,
            }
        n = len(rows)
        correct = sum(1 for r in rows if str(r.get("outcome_label")) in {"helpful", "recovery-associated"})
        harms = sum(1 for r in rows if str(r.get("outcome_label")) == "harmful")
        false_conf = sum(1 for r in rows if cls._safe_float(r.get("confidence")) >= 0.7 and str(r.get("outcome_label")) == "harmful")
        calibration_error = sum(abs(cls._safe_float(r.get("confidence")) - (1.0 if str(r.get("outcome_label")) in {"helpful", "recovery-associated"} else 0.0)) for r in rows) / n
        return {
            "count": n,
            "decision_accuracy": round(correct / n, 6),
            "harm_rate": round(harms / n, 6),
            "calibration_error": round(calibration_error, 6),
            "calibration_quality": round(max(0.0, 1.0 - calibration_error), 6),
            "false_confidence_rate": round(false_conf / n, 6),
        }

    @classmethod
    def _build_cohort_breakdowns(cls, timeline: list[dict[str, Any]]) -> dict[str, Any]:
        by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_system_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        sparse_data: list[dict[str, Any]] = []
        drifted: list[dict[str, Any]] = []
        intervention_present: list[dict[str, Any]] = []
        intervention_missing: list[dict[str, Any]] = []
        novelty_blockers: list[dict[str, Any]] = []

        for row in timeline:
            by_domain[str(row.get("domain") or "unknown")].append(row)
            by_asset[str(row.get("asset_id") or "unknown")].append(row)
            by_system_type[str(row.get("system_type") or "unknown")].append(row)
            support_count = int(row.get("support_count", 0) or 0)
            novelty = cls._safe_float(row.get("novelty"), fallback=0.0)
            if support_count <= 1:
                sparse_data.append(row)
            if bool(row.get("drift_warning")) or novelty >= 0.7:
                drifted.append(row)
            if novelty >= 0.75 and support_count <= 2 and bool(row.get("drift_warning")):
                novelty_blockers.append(row)
            if row.get("actual_intervention"):
                intervention_present.append(row)
            else:
                intervention_missing.append(row)

        per_domain = {k: cls._cohort_metrics(v) for k, v in sorted(by_domain.items())}
        per_asset = {k: cls._cohort_metrics(v) for k, v in sorted(by_asset.items())}
        return {
            "overall": cls._cohort_metrics(timeline),
            "per_domain": per_domain,
            "per_asset": per_asset,
            "per_system_type": {k: cls._cohort_metrics(v) for k, v in sorted(by_system_type.items())},
            "macro_metrics": {
                "domain_macro_decision_accuracy": round(sum(v["decision_accuracy"] for v in per_domain.values()) / max(1, len(per_domain)), 6),
                "asset_macro_decision_accuracy": round(sum(v["decision_accuracy"] for v in per_asset.values()) / max(1, len(per_asset)), 6),
            },
            "sparse_data_subset": cls._cohort_metrics(sparse_data),
            "drifted_or_high_novelty_subset": cls._cohort_metrics(drifted),
            "novelty_weak_support_drift_subset": cls._cohort_metrics(novelty_blockers),
            "intervention_present_subset": cls._cohort_metrics(intervention_present),
            "no_intervention_subset": cls._cohort_metrics(intervention_missing),
        }

    @staticmethod
    def _build_core_validation_artifact(
        *,
        step_logs: list[dict[str, Any]],
        outcomes: list[dict[str, Any]],
        metrics: dict[str, Any],
        drift: dict[str, Any],
    ) -> dict[str, Any]:
        timeline = []
        outcome_by_step = {int(o.get("timestep", -1)): o for o in outcomes}
        harmful_cases = []
        for row in step_logs:
            step = int(row.get("timestep", -1))
            outcome = outcome_by_step.get(step, {})
            label = str(outcome.get("outcome_label", "neutral"))
            confidence = float(row.get("confidence", 0.0) or 0.0)
            entry = {
                "timestep": step,
                "timestamp": row.get("timestamp"),
                "asset_id": row.get("asset_id"),
                "domain": row.get("domain") or "unknown",
                "scenario_family": row.get("scenario_family") or "unknown",
                "scenario_id": row.get("scenario_id") or f"step-{step}",
                "system_type": row.get("system_type") or "unknown",
                "recommended_intervention": row.get("recommended_intervention"),
                "actual_intervention": row.get("actual_intervention"),
                "outcome_label": label,
                "confidence": round(confidence, 6),
                "advisory_mode": row.get("advisory_mode", "standard_advisory"),
                "fallback_triggered": bool(row.get("fallback_triggered", False)),
                "fallback_reasons": list(row.get("fallback_reasons") or []),
                "intervention_memory_contribution": row.get("intervention_memory_contribution"),
                "attribution_confidence": round(float(outcome.get("attribution_confidence", 0.0) or 0.0), 6),
                "novelty": round(float(row.get("novelty", 0.0) or 0.0), 6),
                "support_count": int(row.get("support_count", 0) or 0),
                "drift_warning": bool(row.get("drift_warning", False)),
                "novelty_blocker_case": bool(
                    float(row.get("novelty", 0.0) or 0.0) >= 0.75
                    and int(row.get("support_count", 0) or 0) <= 2
                    and bool(row.get("drift_warning", False))
                ),
            }
            timeline.append(entry)
            if label == "harmful":
                harmful_cases.append(
                    {
                        "scenario_id": entry["scenario_id"],
                        "asset_id": row.get("asset_id"),
                        "predicted_recommendation": row.get("recommended_intervention"),
                        "actual_outcome": label,
                        "actual_intervention": row.get("actual_intervention"),
                        "confidence": round(confidence, 6),
                        "failure_reason": "high_confidence_harm" if confidence >= 0.7 else "harmful_outcome",
                    }
                )

        accepted_helpful = 0
        accepted_total = 0
        ignored_harmful = 0
        for row in timeline:
            label = str(row.get("outcome_label", "neutral"))
            rec = row.get("recommended_intervention")
            actual = row.get("actual_intervention")
            if rec and actual and rec == actual:
                accepted_total += 1
                if label in {"helpful", "recovery-associated"}:
                    accepted_helpful += 1
            if rec and actual and rec != actual and label == "harmful":
                ignored_harmful += 1
        n = max(1, len(timeline))
        false_confidence_rate = float(metrics.get("false_confidence_events", 0) or 0) / n
        drift_warning_rate = round(len(drift.get("warnings", [])) / n, 6)
        memory_samples = [
            dict(row.get("intervention_memory_contribution") or {})
            for row in timeline
            if isinstance(row.get("intervention_memory_contribution"), dict)
            and str((row.get("intervention_memory_contribution") or {}).get("status")) == "available"
        ]
        memory_contribution = RealWorldValidationPipeline._summarize_intervention_memory_contribution(memory_samples)

        cohort_breakdowns = RealWorldValidationPipeline._build_cohort_breakdowns(timeline)
        corpus_summary = RealWorldValidationPipeline._build_corpus_summary(step_logs)

        top_failures = sorted(harmful_cases, key=lambda x: x["confidence"], reverse=True)[:20]

        return {
            "corpus_summary": corpus_summary,
            "summary": {
                "decision_accuracy": metrics.get("decision_accuracy", 0.0),
                "harm_rate": metrics.get("harm_rate", 0.0),
                "calibration_error": metrics.get("calibration_error", 1.0),
                "calibration_quality": metrics.get("calibration_quality", 0.0),
                "false_confidence_events": metrics.get("false_confidence_events", 0),
                "false_confidence_rate": round(false_confidence_rate, 6),
                "intervention_memory_contribution": memory_contribution,
                "drift_status": drift.get("status"),
                "drift_warning_count": len(drift.get("warnings", [])),
                "drift_warning_rate": drift_warning_rate,
            },
            "cohort_breakdowns": cohort_breakdowns,
            "metric_semantics": {"calibration": CALIBRATION_METRIC_DOC},
            "timeline": timeline[-200:],
            "law_changes": [],
            "drift_summary": drift,
            "drift_flags": drift.get("warnings", []),
            "top_failure_cases": top_failures,
            "major_failure_cases": top_failures[:10],
        }

    @staticmethod
    def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
        return out

    @classmethod
    def _summarize_intervention_memory_contribution(cls, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if len(samples) < 5:
            return {
                "status": "insufficient_evidence",
                "method": "candidate_score_ablation_delta",
                "sample_count": len(samples),
                "message": "Not enough recommendation steps with memory ablation diagnostics.",
            }
        deltas = [cls._safe_float(s.get("best_confidence_delta"), fallback=0.0) for s in samples]
        changed = [1.0 if bool(s.get("choice_changed_without_memory")) else 0.0 for s in samples]
        return {
            "status": "available",
            "method": "candidate_score_ablation_delta",
            "sample_count": len(samples),
            "mean_best_confidence_delta": round(sum(deltas) / len(deltas), 6),
            "positive_delta_rate": round(sum(1.0 for d in deltas if d > 0) / len(deltas), 6),
            "top_choice_change_rate_without_memory": round(sum(changed) / len(changed), 6),
        }
