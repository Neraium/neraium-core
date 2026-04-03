from __future__ import annotations

from typing import Any


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class AdversarialTrajectoryGenerator:
    """Deterministic adversarial synthetic trajectory cases used for falsification stress tests."""

    def generate(
        self,
        *,
        trajectory: dict[str, Any],
        transition: dict[str, Any],
        laws: list[dict[str, Any]],
        domain: str,
    ) -> dict[str, Any]:
        path = str(trajectory.get("current_trajectory_path_family") or "unknown")
        archetype = str(trajectory.get("current_trajectory_archetype") or "unknown")
        observed = bool(float(transition.get("escalation_probability", 0.0)) >= 0.6 or transition.get("transition_path") == "escalating")

        cases: list[dict[str, Any]] = []
        for law in laws[:4]:
            law_id = str(law.get("law_id") or "law::unknown")
            predicted = bool(float(law.get("estimated_effect", 0.0)) >= 0.0)
            conf = float(law.get("confidence", 0.0))
            cases.extend(
                [
                    {
                        "case_type": "shape_mimic_outcome_flip",
                        "law_id": law_id,
                        "archetype": archetype,
                        "domain": domain,
                        "trajectory_family": path,
                        "predicted_escalation": predicted,
                        "observed_escalation": not predicted,
                        "contradiction": True,
                        "severity_of_violation": round(float(_clip01(0.55 + 0.45 * conf)), 6),
                    },
                    {
                        "case_type": "descriptor_match_dynamic_shift",
                        "law_id": law_id,
                        "archetype": archetype,
                        "domain": domain,
                        "trajectory_family": path,
                        "predicted_escalation": predicted,
                        "observed_escalation": observed,
                        "contradiction": predicted != observed,
                        "severity_of_violation": round(float(_clip01(abs(float(predicted) - float(observed)) * (0.3 + 0.5 * conf))), 6),
                    },
                    {
                        "case_type": "noise_injected_phase_shifted",
                        "law_id": law_id,
                        "archetype": archetype,
                        "domain": domain,
                        "trajectory_family": path,
                        "predicted_escalation": predicted,
                        "observed_escalation": not observed,
                        "contradiction": predicted == observed,
                        "severity_of_violation": round(float(_clip01((0.25 + 0.5 * conf) if predicted == observed else 0.15)), 6),
                    },
                ]
            )

        total = max(1, len(cases))
        false_positives = sum(1 for c in cases if c["predicted_escalation"] and not c["observed_escalation"])
        contradictions = sum(1 for c in cases if c["contradiction"])
        confusion = {
            "true_positive": sum(1 for c in cases if c["predicted_escalation"] and c["observed_escalation"]),
            "true_negative": sum(1 for c in cases if (not c["predicted_escalation"]) and (not c["observed_escalation"])),
            "false_positive": false_positives,
            "false_negative": sum(1 for c in cases if (not c["predicted_escalation"]) and c["observed_escalation"]),
        }
        return {
            "synthetic_cases": cases,
            "adversarial_test_results": {
                "cases_tested": len(cases),
                "contradictions_triggered": contradictions,
                "false_positive_rate": round(float(false_positives / total), 6),
                "adversarial_resistance": round(float(1.0 - contradictions / total), 6),
                "confusion_metrics": confusion,
            },
        }
