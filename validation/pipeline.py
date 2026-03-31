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

        return {
            "replay": replay_result,
            "outcomes": outcomes,
            "feedback": [f.to_dict() for f in feedback_records],
            "feedback_summary": self.feedback.summarize(feedback_records),
            "metrics": metrics,
            "real_world_validation": {
                "decision_accuracy": metrics["decision_accuracy"],
                "harm_rate": metrics["harm_rate"],
                "calibration": metrics["calibration"],
                "law_validation": {},
                "drift_signals": drift,
            },
        }

    @staticmethod
    def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
        return out
