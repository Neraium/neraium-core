from __future__ import annotations

from typing import Any


class OutcomeAttributor:
    """Conservative attribution for intervention outcomes with delayed windows."""

    def __init__(self, *, window: int = 3, helpful_threshold: float = 0.1, harmful_threshold: float = -0.1) -> None:
        self.window = max(1, int(window))
        self.helpful_threshold = float(helpful_threshold)
        self.harmful_threshold = float(harmful_threshold)

    def _fallback_from_predicted_state(self, row: dict[str, Any]) -> tuple[str, float]:
        state = dict(row.get("predicted_state") or {})
        escalation = float(state.get("escalation_probability", 0.5) or 0.5)
        reversibility = float(state.get("reversibility_score", 0.5) or 0.5)
        score = reversibility - escalation
        if score >= self.helpful_threshold:
            return "helpful", score
        if score <= self.harmful_threshold:
            return "harmful", score
        return "neutral", score

    def attribute(self, decision_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, row in enumerate(decision_logs):
            actual = row.get("actual_outcome")
            if actual in {"helpful", "neutral", "harmful", "recovery-associated"}:
                label = str(actual)
                score = 1.0 if label in {"helpful", "recovery-associated"} else -1.0 if label == "harmful" else 0.0
            else:
                delayed_scores: list[float] = []
                for j in range(idx + 1, min(len(decision_logs), idx + self.window + 1)):
                    future = decision_logs[j]
                    if future.get("asset_id") != row.get("asset_id"):
                        continue
                    state = dict(future.get("predicted_state") or {})
                    delayed_scores.append(float(state.get("reversibility_score", 0.5)) - float(state.get("escalation_probability", 0.5)))
                if delayed_scores:
                    score = sum(delayed_scores) / len(delayed_scores)
                    if score >= self.helpful_threshold:
                        label = "helpful"
                    elif score <= self.harmful_threshold:
                        label = "harmful"
                    elif any(s >= self.helpful_threshold * 0.8 for s in delayed_scores):
                        label = "recovery-associated"
                    else:
                        label = "neutral"
                else:
                    label, score = self._fallback_from_predicted_state(row)

            out.append({
                "timestep": row.get("timestep"),
                "asset_id": row.get("asset_id"),
                "recommended_intervention": row.get("recommended_intervention"),
                "actual_intervention": row.get("actual_intervention"),
                "outcome_label": label,
                "outcome_score": round(float(score), 6),
            })
        return out
