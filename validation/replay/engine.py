from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class ReplayStepResult:
    timestep: int
    timestamp: float
    asset_id: str
    predicted_state: dict[str, Any]
    recommended_intervention: str | None
    confidence: float
    law_usage: list[str]
    actual_intervention: str | None
    actual_outcome: str | None


class HistoricalReplayEngine:
    def __init__(self, decision_fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.decision_fn = decision_fn

    def replay(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        step_logs: list[ReplayStepResult] = []
        trajectory_logs: dict[str, list[dict[str, Any]]] = {}

        for idx, row in enumerate(records):
            observation = dict(row.get("observation") or {})
            observation["asset_id"] = row.get("asset_id", "unknown")
            output = self.decision_fn(observation)

            transition = dict(output.get("transition_dynamics") or {})
            intervention = dict((output.get("intervention_intelligence") or {}).get("recommendation") or {})
            best = dict(intervention.get("best_intervention") or {})
            recommended = best.get("name")
            confidence = float(best.get("confidence", (output.get("reliability_intelligence") or {}).get("risk_advisory", {}).get("calibrated_confidence", 0.0)))
            governance = dict((output.get("structural_law_intelligence") or {}).get("structural_law_governance") or {})
            law_usage = [str(l.get("law_id")) for l in list(governance.get("laws") or []) if l.get("current_stage") != "rejected_or_falsified"]

            step = ReplayStepResult(
                timestep=idx,
                timestamp=float(row.get("timestamp", idx)),
                asset_id=str(row.get("asset_id", "unknown")),
                predicted_state=transition,
                recommended_intervention=str(recommended) if recommended else None,
                confidence=confidence,
                law_usage=law_usage,
                actual_intervention=row.get("actual_intervention"),
                actual_outcome=row.get("outcome_label"),
            )
            step_logs.append(step)
            trajectory_logs.setdefault(step.asset_id, []).append(asdict(step))

        return {
            "step_logs": [asdict(s) for s in step_logs],
            "trajectory_logs": trajectory_logs,
        }
