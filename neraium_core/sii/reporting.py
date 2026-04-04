from __future__ import annotations

import csv
import json
from math import sqrt
from pathlib import Path
from typing import Any

from .errors import SIIIOError

STRUCTURAL_SIGNAL_FIELDS: tuple[str, ...] = (
    "operator_deformation_energy",
    "residual_energy",
    "regime_distance",
    "coherence_margin_raw",
    "structural_drift_magnitude",
)


def results_to_json(results: list[dict[str, Any]]) -> str:
    return json.dumps(results, indent=2)


def write_json_report(path: str | Path, results: list[dict[str, Any]]) -> None:
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(results_to_json(results), encoding="utf-8")
    except Exception as exc:
        raise SIIIOError(f"Failed to write JSON report: {p}") from exc


def results_to_csv_text(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    fields = [
        "timestamp",
        "site_id",
        "asset_id",
        "state",
        "interpreted_state",
        "confidence",
        "structural_drift_score",
        "relational_instability_score",
        "regime_distance",
        "coherence_score",
        "graph_deformation_score",
        "risk_level",
        "risk_trend",
        "inspection_target",
        "explanation",
    ]
    # csv.DictWriter needs file-like object; create via StringIO
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for item in results:
        writer.writerow(
            {
                "timestamp": item.get("timestamp"),
                "site_id": item.get("site_id"),
                "asset_id": item.get("asset_id"),
                "state": item.get("state"),
                "interpreted_state": item.get("interpreted_state"),
                "confidence": item.get("confidence"),
                "structural_drift_score": item.get("structural_drift_score"),
                "relational_instability_score": item.get("relational_instability_score"),
                "regime_distance": item.get("regime_distance"),
                "coherence_score": item.get("coherence_score"),
                "graph_deformation_score": item.get("graph_deformation_score"),
                "risk_level": ((item.get("risk_assessment") or {}) if isinstance(item.get("risk_assessment"), dict) else {}).get(
                    "current_risk_level"
                ),
                "risk_trend": ((item.get("risk_assessment") or {}) if isinstance(item.get("risk_assessment"), dict) else {}).get(
                    "projected_near_term_trend"
                ),
                "inspection_target": (
                    ((item.get("operator_guidance") or {}) if isinstance(item.get("operator_guidance"), dict) else {}).get(
                        "first_inspection_target"
                    )
                ),
                "explanation": item.get("explanation"),
            }
        )
    return buf.getvalue()


def write_csv_report(path: str | Path, results: list[dict[str, Any]]) -> None:
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(results_to_csv_text(results), encoding="utf-8")
    except Exception as exc:
        raise SIIIOError(f"Failed to write CSV report: {p}") from exc


def compute_structural_temporal_validation(
    results: list[dict[str, Any]],
    *,
    lead_window: int = 3,
) -> dict[str, Any]:
    if not results:
        return {
            "sample_count": 0,
            "transition_count": 0,
            "transitions": [],
            "signals": {},
        }

    lead_window = max(1, int(lead_window))

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(results):
        structural = item.get("structural_state") if isinstance(item.get("structural_state"), dict) else {}
        row = {"step": idx, "state": str(item.get("state") or ""), "interpreted_state": str(item.get("interpreted_state") or "")}
        for field in STRUCTURAL_SIGNAL_FIELDS:
            source = item.get(field, structural.get(field, 0.0))
            try:
                row[field] = float(source)
            except (TypeError, ValueError):
                row[field] = 0.0
        rows.append(row)

    transitions: list[dict[str, Any]] = []
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        current = rows[idx]
        if prev["state"] != current["state"] or prev["interpreted_state"] != current["interpreted_state"]:
            transitions.append(
                {
                    "step": idx,
                    "from_state": prev["state"],
                    "to_state": current["state"],
                    "from_interpreted_state": prev["interpreted_state"],
                    "to_interpreted_state": current["interpreted_state"],
                }
            )

    transition_steps = {t["step"] for t in transitions}
    diagnostics: dict[str, Any] = {}
    for field in STRUCTURAL_SIGNAL_FIELDS:
        diffs: list[float] = [abs(rows[idx][field] - rows[idx - 1][field]) for idx in range(1, len(rows))]
        if not diffs:
            diagnostics[field] = {
                "mean": round(rows[0][field], 6),
                "peak": round(rows[0][field], 6),
                "mean_abs_change": 0.0,
                "spike_threshold": 0.0,
                "spike_steps": [],
                "leading_transition_hits": 0,
                "concurrent_transition_hits": 0,
                "lagging_transition_hits": 0,
                "lead_precision": 0.0,
                "lead_recall": 0.0,
            }
            continue

        mean_diff = sum(diffs) / len(diffs)
        variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
        spike_threshold = mean_diff + 1.0 * sqrt(max(0.0, variance))
        spike_steps = [idx for idx in range(1, len(rows)) if abs(rows[idx][field] - rows[idx - 1][field]) >= spike_threshold]
        spike_set = set(spike_steps)

        leading_hits = 0
        concurrent_hits = 0
        lagging_hits = 0
        for step in transition_steps:
            has_lead = any((step - k) in spike_set for k in range(1, lead_window + 1))
            has_concurrent = step in spike_set
            has_lag = any((step + k) in spike_set for k in range(1, lead_window + 1))
            leading_hits += int(has_lead)
            concurrent_hits += int(has_concurrent)
            lagging_hits += int(has_lag)

        diagnostics[field] = {
            "mean": round(sum(r[field] for r in rows) / len(rows), 6),
            "peak": round(max(r[field] for r in rows), 6),
            "mean_abs_change": round(mean_diff, 6),
            "spike_threshold": round(spike_threshold, 6),
            "spike_steps": spike_steps,
            "leading_transition_hits": leading_hits,
            "concurrent_transition_hits": concurrent_hits,
            "lagging_transition_hits": lagging_hits,
            "lead_precision": round(leading_hits / max(1, len(spike_steps)), 6),
            "lead_recall": round(leading_hits / max(1, len(transition_steps)), 6),
        }

    return {
        "sample_count": len(rows),
        "transition_count": len(transitions),
        "transitions": transitions,
        "signals": diagnostics,
    }


def compute_structural_signal_ranking(
    temporal_validation: dict[str, Any],
) -> dict[str, Any]:
    signals = temporal_validation.get("signals")
    if not isinstance(signals, dict) or not signals:
        return {
            "transition_count": int(temporal_validation.get("transition_count") or 0),
            "ranking": [],
        }

    transition_count = int(temporal_validation.get("transition_count") or 0)
    ranking: list[dict[str, Any]] = []
    for signal_name, metrics in signals.items():
        signal_metrics = metrics if isinstance(metrics, dict) else {}
        lead_recall = float(signal_metrics.get("lead_recall") or 0.0)
        lead_precision = float(signal_metrics.get("lead_precision") or 0.0)
        leading_hits = int(signal_metrics.get("leading_transition_hits") or 0)
        concurrent_hits = int(signal_metrics.get("concurrent_transition_hits") or 0)
        lagging_hits = int(signal_metrics.get("lagging_transition_hits") or 0)
        spike_steps = signal_metrics.get("spike_steps")
        spike_count = len(spike_steps) if isinstance(spike_steps, list) else 0

        alignment_hits = leading_hits + concurrent_hits + lagging_hits
        transition_coverage = (
            round(alignment_hits / max(1, transition_count), 6) if transition_count > 0 else 0.0
        )
        spike_to_transition_ratio = (
            round(spike_count / max(1, transition_count), 6) if transition_count > 0 else 0.0
        )

        temporal_role = "weak"
        if leading_hits > max(concurrent_hits, lagging_hits) and lead_recall > 0.0:
            temporal_role = "leading"
        elif concurrent_hits > max(leading_hits, lagging_hits):
            temporal_role = "concurrent"
        elif lagging_hits > max(leading_hits, concurrent_hits):
            temporal_role = "lagging"

        ranking.append(
            {
                "signal": signal_name,
                "predictive_metrics": {
                    "lead_recall": round(lead_recall, 6),
                    "lead_precision": round(lead_precision, 6),
                    "transition_coverage": transition_coverage,
                },
                "behavior_metrics": {
                    "spike_count": spike_count,
                    "spike_to_transition_ratio": spike_to_transition_ratio,
                    "mean_abs_change": float(signal_metrics.get("mean_abs_change") or 0.0),
                    "spike_threshold": float(signal_metrics.get("spike_threshold") or 0.0),
                },
                "temporal_alignment": {
                    "leading_transition_hits": leading_hits,
                    "concurrent_transition_hits": concurrent_hits,
                    "lagging_transition_hits": lagging_hits,
                    "dominant_role": temporal_role,
                },
            }
        )

    ranking.sort(
        key=lambda item: (
            item["predictive_metrics"]["lead_recall"],
            item["predictive_metrics"]["lead_precision"],
            item["predictive_metrics"]["transition_coverage"],
            -item["behavior_metrics"]["spike_to_transition_ratio"],
        ),
        reverse=True,
    )
    for index, item in enumerate(ranking, start=1):
        item["rank"] = index

    return {
        "transition_count": transition_count,
        "ranking": ranking,
    }


def evaluate_structural_risk_decision_policy(
    *,
    signal_ranking: dict[str, Any],
    raw_structural_magnitudes: dict[str, Any],
    min_lead_recall: float = 0.3,
    min_lead_precision: float = 0.3,
    max_eligible_rank: int = 2,
) -> dict[str, Any]:
    ranking = signal_ranking.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return {
            "risk_flag": False,
            "risk_level": "normal",
            "decision_posture": "monitor_structural_baseline",
            "contributing_signals": [],
            "signal_evaluation": [],
            "supporting_raw_magnitudes": {},
        }

    eligible_limit = max(1, int(max_eligible_rank))
    evaluation: list[dict[str, Any]] = []
    contributing: list[dict[str, Any]] = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        signal = str(item.get("signal") or "")
        if not signal:
            continue
        rank = int(item.get("rank") or 0)
        predictive = item.get("predictive_metrics") if isinstance(item.get("predictive_metrics"), dict) else {}
        behavior = item.get("behavior_metrics") if isinstance(item.get("behavior_metrics"), dict) else {}
        temporal = item.get("temporal_alignment") if isinstance(item.get("temporal_alignment"), dict) else {}

        role = str(temporal.get("dominant_role") or "weak")
        lead_recall = float(predictive.get("lead_recall") or 0.0)
        lead_precision = float(predictive.get("lead_precision") or 0.0)
        spike_threshold = abs(float(behavior.get("spike_threshold") or 0.0))
        baseline_mean = float(behavior.get("mean_abs_change") or 0.0)

        signal_raw = float(raw_structural_magnitudes.get(signal) or 0.0)
        threshold_distance = abs(signal_raw - baseline_mean)
        threshold_crossed = threshold_distance >= spike_threshold and spike_threshold > 0.0

        eligible = False
        if role == "leading" and rank <= eligible_limit and lead_recall >= min_lead_recall:
            eligible = True
        elif role == "concurrent" and rank == 1 and lead_recall >= min_lead_recall and lead_precision >= min_lead_precision:
            eligible = True

        signal_entry = {
            "signal": signal,
            "rank": rank,
            "temporal_role": role,
            "eligible_for_early_risk": eligible,
            "threshold_comparison": {
                "threshold": round(spike_threshold, 6),
                "baseline_reference": round(baseline_mean, 6),
                "current_raw_magnitude": round(signal_raw, 6),
                "distance_to_threshold": round(threshold_distance, 6),
                "threshold_crossed": threshold_crossed,
            },
        }
        evaluation.append(signal_entry)
        if eligible and threshold_crossed:
            contributing.append(signal_entry)

    elevated = len(contributing) > 0
    risk_level = "normal"
    decision_posture = "monitor_structural_baseline"
    if elevated and len(contributing) >= 2:
        risk_level = "high"
        decision_posture = "elevated_structural_risk_confirmed"
    elif elevated:
        risk_level = "elevated"
        decision_posture = "elevated_structural_risk_watch"

    return {
        "risk_flag": elevated,
        "risk_level": risk_level,
        "decision_posture": decision_posture,
        "contributing_signals": contributing,
        "signal_evaluation": evaluation,
        "supporting_raw_magnitudes": {
            field: round(float(raw_structural_magnitudes.get(field) or 0.0), 6) for field in STRUCTURAL_SIGNAL_FIELDS
        },
    }
