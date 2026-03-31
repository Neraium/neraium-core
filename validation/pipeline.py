from __future__ import annotations

from pathlib import Path
from typing import Any

from validation.drift import DriftDetector
from validation.feedback import FeedbackIntegrator
from validation.metrics import compute_backtest_metrics
from validation.outcomes import OutcomeAttributor
from validation.replay import HistoricalReplayEngine


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
                "calibration": metrics["calibration"],
                "law_validation": {},
                "drift_signals": drift,
            },
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
            timeline.append(
                {
                    "timestep": step,
                    "asset_id": row.get("asset_id"),
                    "recommended_intervention": row.get("recommended_intervention"),
                    "outcome_label": label,
                    "confidence": round(confidence, 6),
                    "attribution_confidence": round(float(outcome.get("attribution_confidence", 0.0) or 0.0), 6),
                }
            )
            if label == "harmful":
                harmful_cases.append(
                    {
                        "timestep": step,
                        "asset_id": row.get("asset_id"),
                        "recommended_intervention": row.get("recommended_intervention"),
                        "actual_intervention": row.get("actual_intervention"),
                        "confidence": round(confidence, 6),
                    }
                )

        accepted_helpful = 0
        accepted_total = 0
        ignored_harmful = 0
        for row in step_logs:
            step = int(row.get("timestep", -1))
            outcome = outcome_by_step.get(step, {})
            label = str(outcome.get("outcome_label", "neutral"))
            rec = row.get("recommended_intervention")
            actual = row.get("actual_intervention")
            if rec and actual and rec == actual:
                accepted_total += 1
                if label in {"helpful", "recovery-associated"}:
                    accepted_helpful += 1
            if rec and actual and rec != actual and label == "harmful":
                ignored_harmful += 1
        memory_contribution = round((accepted_helpful / max(1, accepted_total)) - (ignored_harmful / max(1, len(step_logs))), 6)

        return {
            "summary": {
                "decision_accuracy": metrics.get("decision_accuracy", 0.0),
                "harm_rate": metrics.get("harm_rate", 0.0),
                "calibration_quality": metrics.get("calibration", 0.0),
                "intervention_memory_contribution": memory_contribution,
                "drift_status": drift.get("status"),
            },
            "timeline": timeline[-50:],
            "law_changes": [],
            "drift_flags": drift.get("warnings", []),
            "major_failure_cases": harmful_cases[:10],
        }

    @staticmethod
    def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
        return out
