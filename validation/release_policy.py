from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from validation.release_gates import ReleaseGateThresholds, evaluate_release_gates

CORPUS_TYPES = {
    "baseline_clean",
    "noisy_realistic",
    "adversarial",
    "transfer_cross_domain",
}


@dataclass(frozen=True)
class CorpusClassRule:
    required: bool
    max_failure_ratio: float
    catastrophic_gates: tuple[str, ...] = ()


CORPUS_CLASS_RULES: dict[str, CorpusClassRule] = {
    "baseline_clean": CorpusClassRule(required=True, max_failure_ratio=0.0),
    "noisy_realistic": CorpusClassRule(required=True, max_failure_ratio=0.0),
    "adversarial": CorpusClassRule(
        required=False,
        max_failure_ratio=0.34,
        catastrophic_gates=("maximum_harm_rate", "maximum_calibration_error", "maximum_false_confidence_rate"),
    ),
    "transfer_cross_domain": CorpusClassRule(
        required=True,
        max_failure_ratio=0.0,
        catastrophic_gates=("minimum_decision_accuracy", "maximum_harm_rate"),
    ),
}


CORPUS_TYPE_THRESHOLDS: dict[str, ReleaseGateThresholds] = {
    "baseline_clean": ReleaseGateThresholds(
        min_decision_accuracy=0.74,
        max_harm_rate=0.18,
        max_calibration_error=0.30,
        max_false_confidence_rate=0.08,
        max_drift_warning_rate=0.22,
    ),
    "noisy_realistic": ReleaseGateThresholds(
        min_decision_accuracy=0.69,
        max_harm_rate=0.22,
        max_calibration_error=0.37,
        max_false_confidence_rate=0.12,
        max_drift_warning_rate=0.30,
    ),
    "adversarial": ReleaseGateThresholds(
        min_decision_accuracy=0.62,
        max_harm_rate=0.27,
        max_calibration_error=0.42,
        max_false_confidence_rate=0.15,
        max_drift_warning_rate=0.36,
    ),
    "transfer_cross_domain": ReleaseGateThresholds(
        min_decision_accuracy=0.66,
        max_harm_rate=0.25,
        max_calibration_error=0.40,
        max_false_confidence_rate=0.14,
        max_drift_warning_rate=0.32,
    ),
}


def classify_failure_modes(corpus_type: str, gate_breakdown: list[dict[str, Any]]) -> list[str]:
    tags: set[str] = set()
    failed = {str(g.get("gate")) for g in gate_breakdown if not g.get("passed")}
    if "minimum_decision_accuracy" in failed:
        tags.add("trajectory_misclassification")
    if "maximum_harm_rate" in failed:
        tags.add("intervention_ranking_error")
    if "maximum_calibration_error" in failed or "maximum_false_confidence_rate" in failed:
        tags.add("calibration_failure")
    if "minimum_domain_coverage" in failed or "minimum_asset_coverage" in failed:
        tags.add("attribution_error")
    if corpus_type == "transfer_cross_domain" and failed:
        tags.add("transfer_failure")
    return sorted(tags)


def evaluate_corpus_release(core_validation_report: dict[str, Any], *, corpus_type: str) -> dict[str, Any]:
    thresholds = CORPUS_TYPE_THRESHOLDS.get(corpus_type, ReleaseGateThresholds())
    gate = evaluate_release_gates(core_validation_report, thresholds=thresholds)
    gate["corpus_type"] = corpus_type
    gate["policy_thresholds"] = asdict(thresholds)
    tags = set(classify_failure_modes(corpus_type, gate.get("gate_breakdown", [])))
    novelty_subset = ((core_validation_report.get("cohort_breakdowns") or {}).get("novelty_weak_support_drift_subset") or {})
    if float(novelty_subset.get("false_confidence_rate", 0.0) or 0.0) > 0.08 or float(novelty_subset.get("harm_rate", 0.0) or 0.0) > 0.2:
        tags.add("novelty_fallback_failure")
    gate["failure_mode_tags"] = sorted(tags)
    return gate


DEFAULT_MIN_CREDIBLE_RECORDS = 50


def _is_credible_for_release(row: dict[str, Any], *, min_credible_records: int) -> bool:
    explicit = row.get("eligible_for_release_decision")
    if explicit is not None:
        return bool(explicit)
    if "corpus_summary" not in row:
        return True
    corpus_summary = dict(row.get("corpus_summary") or {})
    total_records = int(corpus_summary.get("total_records", 0) or 0)
    return total_records >= min_credible_records


def aggregate_multi_corpus_release(
    results: list[dict[str, Any]],
    *,
    min_credible_records: int = DEFAULT_MIN_CREDIBLE_RECORDS,
    include_ineligible: bool = False,
) -> dict[str, Any]:
    decision_rows = list(results)
    excluded_non_credible_corpora: list[str] = []
    if not include_ineligible:
        decision_rows = []
        for row in results:
            if _is_credible_for_release(row, min_credible_records=min_credible_records):
                decision_rows.append(row)
            else:
                excluded_non_credible_corpora.append(str(row.get("corpus_id")))

    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in decision_rows:
        corpus_type = str(row.get("corpus_type") or "unknown")
        by_class.setdefault(corpus_type, []).append(row)

    blocking_classes: list[str] = []
    class_summaries: dict[str, Any] = {}
    failing_corpora: list[str] = []
    failure_frequency: dict[str, int] = {}

    for corpus_type, rows in by_class.items():
        rule = CORPUS_CLASS_RULES.get(corpus_type, CorpusClassRule(required=False, max_failure_ratio=0.0))
        total = len(rows)
        failed_rows = [r for r in rows if not r.get("release_passed")]
        for r in failed_rows:
            failing_corpora.append(str(r.get("corpus_id")))
            for tag in r.get("failure_mode_tags", []):
                failure_frequency[tag] = failure_frequency.get(tag, 0) + 1

        catastrophic = []
        for row in failed_rows:
            failed_gates = {g.get("gate") for g in row.get("gate_breakdown", []) if not g.get("passed")}
            if any(g in failed_gates for g in rule.catastrophic_gates):
                catastrophic.append(str(row.get("corpus_id")))

        failure_ratio = len(failed_rows) / max(1, total)
        class_passed = failure_ratio <= rule.max_failure_ratio and not catastrophic
        if rule.required and (total == 0 or not class_passed):
            blocking_classes.append(corpus_type)
        if not rule.required and not class_passed:
            blocking_classes.append(corpus_type)

        class_summaries[corpus_type] = {
            "required": rule.required,
            "total_corpora": total,
            "failed_corpora": [str(r.get("corpus_id")) for r in failed_rows],
            "failure_ratio": round(failure_ratio, 6),
            "max_failure_ratio": rule.max_failure_ratio,
            "catastrophic_failures": catastrophic,
            "class_passed": class_passed,
        }

    missing_required = sorted(k for k, v in CORPUS_CLASS_RULES.items() if v.required and k not in by_class)
    release_passed = len(blocking_classes) == 0 and not missing_required

    return {
        "release_passed": release_passed,
        "release_recommendation": "ship" if release_passed else "no-ship",
        "corpus_results": results,
        "release_decision_corpus_results": decision_rows,
        "excluded_non_credible_corpora": sorted(set(excluded_non_credible_corpora)),
        "release_decision_minimum_records": min_credible_records,
        "release_decision_include_ineligible": include_ineligible,
        "class_summary": class_summaries,
        "blocking_corpus_classes": sorted(set(blocking_classes)),
        "missing_required_corpus_classes": missing_required,
        "failing_corpora": sorted(set(failing_corpora)),
        "failure_mode_frequency": dict(sorted(failure_frequency.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_failure_modes": [k for k, _ in sorted(failure_frequency.items(), key=lambda kv: (-kv[1], kv[0]))[:3]],
    }
