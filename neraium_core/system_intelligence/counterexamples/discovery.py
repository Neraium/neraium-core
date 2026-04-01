from __future__ import annotations

from collections import defaultdict
from typing import Any


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class CounterexampleDiscoveryEngine:
    """Inspectable counterexample discovery across historical and synthetic traces."""

    def __init__(self, *, max_history: int = 1200) -> None:
        self.max_history = int(max_history)
        self._records: list[dict[str, Any]] = []

    def scan(
        self,
        *,
        domain: str,
        transition: dict[str, Any],
        trajectory: dict[str, Any],
        laws: list[dict[str, Any]],
        synthetic_cases: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        observed_escalation = bool(float(transition.get("escalation_probability", 0.0)) >= 0.6 or transition.get("transition_path") == "escalating")
        regime = str(transition.get("regime") or "unknown")
        path = str(trajectory.get("current_trajectory_path_family") or "unknown")
        archetype = str(trajectory.get("current_trajectory_archetype") or "unknown")

        events: list[dict[str, Any]] = []
        for law in laws:
            law_id = str(law.get("law_id") or "law::unknown")
            predicted_escalation = bool(float(law.get("estimated_effect", 0.0)) >= 0.0)
            contradiction = predicted_escalation != observed_escalation
            severity = 0.6 * abs(float(law.get("estimated_effect", 0.0))) + 0.4 * float(law.get("confidence", 0.0))
            event = {
                "kind": "law_outcome_violation" if contradiction else "supporting_case",
                "law_id": law_id,
                "archetype": archetype,
                "domain": domain,
                "regime": regime,
                "trajectory_family": path,
                "predicted_escalation": predicted_escalation,
                "observed_escalation": observed_escalation,
                "contradiction": contradiction,
                "severity_of_violation": round(float(_clip01(severity if contradiction else 0.0)), 6),
            }
            events.append(event)
            self._records.append(event)

        for syn in list(synthetic_cases or []):
            syn_event = {
                "kind": "synthetic_counterexample" if bool(syn.get("contradiction")) else "synthetic_support",
                "law_id": str(syn.get("law_id") or "law::synthetic"),
                "archetype": str(syn.get("archetype") or archetype),
                "domain": str(syn.get("domain") or domain),
                "regime": str(syn.get("regime") or regime),
                "trajectory_family": str(syn.get("trajectory_family") or path),
                "predicted_escalation": bool(syn.get("predicted_escalation", True)),
                "observed_escalation": bool(syn.get("observed_escalation", False)),
                "contradiction": bool(syn.get("contradiction", False)),
                "severity_of_violation": round(float(_clip01(float(syn.get("severity_of_violation", 0.5)))), 6),
            }
            events.append(syn_event)
            self._records.append(syn_event)

        self._records = self._records[-self.max_history :]
        counterexamples = [e for e in events if e["contradiction"]]
        clusters = self._cluster(counterexamples)
        return {
            "counterexample_events": counterexamples,
            "counterexample_clusters": clusters,
            "context_difference_analysis": self._context_difference(counterexamples),
        }

    def history_for_law(self, law_id: str) -> list[dict[str, Any]]:
        return [r for r in self._records if r.get("law_id") == law_id]

    def _cluster(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in events:
            key = f"{e.get('law_id')}|{e.get('trajectory_family')}|{e.get('regime')}"
            grouped[key].append(e)
        clusters = []
        for key, rows in grouped.items():
            failures = len(rows)
            severity = sum(float(r.get("severity_of_violation", 0.0)) for r in rows) / max(1, failures)
            clusters.append(
                {
                    "cluster_id": f"cx::{key}",
                    "size": failures,
                    "severity_of_violation": round(float(_clip01(severity)), 6),
                    "domains": sorted({str(r.get("domain") or "unknown") for r in rows}),
                    "example": rows[0],
                }
            )
        clusters.sort(key=lambda row: (row["severity_of_violation"], row["size"]), reverse=True)
        return clusters

    def _context_difference(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        if not events:
            return {
                "domains": [],
                "regimes": [],
                "trajectory_families": [],
                "note": "No contradictions detected in this step.",
            }
        return {
            "domains": sorted({str(e.get("domain") or "unknown") for e in events}),
            "regimes": sorted({str(e.get("regime") or "unknown") for e in events}),
            "trajectory_families": sorted({str(e.get("trajectory_family") or "unknown") for e in events}),
            "note": "Contradictions are surfaced and grouped by context.",
        }
