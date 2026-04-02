from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ..adversarial.generator import AdversarialTrajectoryGenerator
from ..counterexamples.discovery import CounterexampleDiscoveryEngine
from ..refinement.splitter import LawRefinementEngine


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class StructuralFalsificationEngine:
    """Experimental validation layer that tries to disprove discovered structural hypotheses."""

    def __init__(self) -> None:
        self.counterexamples = CounterexampleDiscoveryEngine()
        self.adversarial = AdversarialTrajectoryGenerator()
        self.refinement = LawRefinementEngine()

    def update(
        self,
        *,
        domain: str,
        transition: dict[str, Any],
        trajectory: dict[str, Any],
        laws: dict[str, Any],
        intervention: dict[str, Any],
        universal: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = list(laws.get("law_candidates") or [])
        adv = self.adversarial.generate(
            trajectory=trajectory,
            transition=transition,
            laws=candidates,
            domain=domain,
        )
        cx = self.counterexamples.scan(
            domain=domain,
            transition=transition,
            trajectory=trajectory,
            laws=candidates,
            synthetic_cases=list(adv.get("synthetic_cases") or []),
        )
        law_evals = []
        epistemic = {}
        robustness_scores = {}
        warnings = []

        for law in candidates:
            law_id = str(law.get("law_id") or "law::unknown")
            history = self.counterexamples.history_for_law(law_id)
            support_count = sum(1 for row in history if not row.get("contradiction"))
            contradiction_count = sum(1 for row in history if row.get("contradiction"))
            total = max(1, support_count + contradiction_count)
            contradiction_rate = contradiction_count / total

            rates = []
            by_domain: dict[str, list[bool]] = defaultdict(list)
            for row in history:
                by_domain[str(row.get("domain") or "unknown")].append(bool(row.get("observed_escalation")))
            for values in by_domain.values():
                rates.append(sum(1 for v in values if v) / max(1, len(values)))
            domain_variance = float((sum((r - (sum(rates) / max(1, len(rates)))) ** 2 for r in rates) / max(1, len(rates))) if rates else 0.0)

            boundary_conditions = self._boundary_conditions(history)
            domain_failures = [d for d, rows in by_domain.items() if rows and (sum(1 for r in rows if r) / len(rows)) < 0.35]
            transfer_metrics = self._transfer_failure_metrics(universal)
            adversarial_results = adv.get("adversarial_test_results") or {}
            adversarial_resistance = float(adversarial_results.get("adversarial_resistance", 0.0))
            support_strength = min(1.0, support_count / 12.0)
            cross_domain_consistency = 1.0 - min(1.0, math.sqrt(domain_variance))
            transfer_stability = 1.0 - float(transfer_metrics.get("transfer_failure_rate", 0.0))

            robustness_score = _clip01(
                0.18 * support_strength
                + 0.37 * (1.0 - contradiction_rate)
                + 0.20 * cross_domain_consistency
                + 0.15 * adversarial_resistance
                + 0.10 * transfer_stability
            )
            falsification_score = _clip01(0.65 * contradiction_rate + 0.2 * domain_variance + 0.15 * (1.0 - adversarial_resistance))
            confidence_adjustment = round(float(_clip01(0.5 + 0.5 * robustness_score - 0.35 * contradiction_rate)), 6)
            epistemic_state = self._classify_epistemic(
                support_count=support_count,
                contradiction_rate=contradiction_rate,
                cross_domain_consistency=cross_domain_consistency,
                transfer_success=1.0 - float(transfer_metrics.get("transfer_failure_rate", 0.0)),
            )
            refinement = self.refinement.refine(law=law, evidence=history)

            contradiction_examples = [row for row in history if row.get("contradiction")][:3]
            evaluation = {
                "law_id": law_id,
                "support_count": support_count,
                "contradiction_count": contradiction_count,
                "contradiction_rate": round(float(contradiction_rate), 6),
                "domain_variance": round(float(domain_variance), 6),
                "boundary_conditions": boundary_conditions,
                "falsification_score": round(float(falsification_score), 6),
                "robustness_score": round(float(robustness_score), 6),
                "confidence_adjustment_factor": confidence_adjustment,
                "contradiction_examples": contradiction_examples,
                "domain_failure_flags": domain_failures,
                "transfer_failure": transfer_metrics,
                "epistemic_status": epistemic_state,
                "law_refinement": refinement,
            }
            law_evals.append(evaluation)
            robustness_scores[law_id] = {
                "robustness_score": evaluation["robustness_score"],
                "confidence_adjustment_factor": confidence_adjustment,
            }
            epistemic[law_id] = epistemic_state
            if epistemic_state in {"fragile_pattern", "falsified_pattern"}:
                warnings.append(f"{law_id} is {epistemic_state}; contradictions should be reviewed before trusting transfer.")

        warnings.extend(self._transfer_warnings(universal, intervention))
        return {
            "falsification_analysis": {
                "status": "experimental",
                "law_evaluations": law_evals,
                "counterexamples": {
                    "counterexample_clusters": cx.get("counterexample_clusters", []),
                    "counterexample_events": cx.get("counterexample_events", []),
                    "context_difference_analysis": cx.get("context_difference_analysis", {}),
                    "adversarial_test_results": adv.get("adversarial_test_results", {}),
                },
                "robustness_scores": robustness_scores,
                "epistemic_classifications": epistemic,
                "warnings": warnings,
            }
        }

    def _boundary_conditions(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = f"{row.get('regime')}|{row.get('trajectory_family')}"
            grouped[key].append(row)
        out = []
        for key, records in grouped.items():
            contradictions = sum(1 for r in records if r.get("contradiction"))
            if contradictions == 0:
                continue
            regime, family = key.split("|", 1)
            out.append(
                {
                    "regime": regime,
                    "trajectory_family": family,
                    "contradiction_rate": round(float(contradictions / max(1, len(records))), 6),
                    "support": len(records),
                }
            )
        return out

    def _transfer_failure_metrics(self, universal: dict[str, Any]) -> dict[str, Any]:
        section = (universal.get("experimental_universal_structural_intelligence") or {})
        transfer = section.get("intervention_transfer") or {}
        failed = list(transfer.get("failed_transfers") or [])
        ok = list(transfer.get("transferable_interventions") or [])
        total = max(1, len(failed) + len(ok))
        failure_rate = len(failed) / total
        return {
            "transfer_success_rate": round(float(len(ok) / total), 6),
            "transfer_failure_rate": round(float(failure_rate), 6),
            "mismatch_penalty": round(float(_clip01(0.65 * failure_rate)), 6),
            "transfer_failure_cases": failed[:4],
            "domain_dependency_flags": [r.get("intervention") for r in list(transfer.get("domain_specific_interventions") or [])],
        }

    def _transfer_warnings(self, universal: dict[str, Any], intervention: dict[str, Any]) -> list[str]:
        warnings = []
        section = universal.get("experimental_universal_structural_intelligence") or {}
        safety = section.get("safety_boundary") or {}
        for w in list(safety.get("warnings") or []):
            warnings.append(f"universal-layer warning: {w}")
        ranked = list((intervention.get("recommendation") or {}).get("ranked_interventions") or [])
        if ranked and float(((ranked[0].get("evidence_sources") or {}).get("historical_evidence") or {}).get("uncertainty", 1.0)) > 0.55:
            warnings.append("intervention evidence uncertainty is high; transfer claims are fragile.")
        return warnings

    def _classify_epistemic(
        self,
        *,
        support_count: int,
        contradiction_rate: float,
        cross_domain_consistency: float,
        transfer_success: float,
    ) -> str:
        if contradiction_rate >= 0.55 and support_count >= 3:
            return "falsified_pattern"
        if contradiction_rate >= 0.4:
            return "fragile_pattern"
        if support_count < 3:
            return "descriptive_pattern"
        if contradiction_rate < 0.2 and cross_domain_consistency > 0.7 and transfer_success > 0.65:
            return "robust_predictor"
        if transfer_success < 0.5:
            return "intervention_sensitive_pattern"
        if contradiction_rate < 0.3:
            return "context_conditioned_pattern"
        return "descriptive_pattern"
