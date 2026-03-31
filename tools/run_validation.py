from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neraium_core.system_intelligence.platform import StructuralSystemIntelligencePlatform
from neraium_core.system_intelligence.law_governance import StructuralLawGovernance
from validation import RealWorldValidationPipeline
from validation.replay import load_dataset


def _build_feedback_rows(report: dict) -> list[dict]:
    rows = []
    for rec in report.get("feedback", []):
        rows.append(
            {
                "outcome_label": rec.get("outcome_label", "neutral"),
                "confidence": 0.8 if rec.get("action") == "accepted" else 0.4,
                "trajectory_family": "unknown",
                "transition_path": "unknown",
                "regime": "unknown",
                "novelty": 0.5,
                "support_count": 1,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-world replay + validation loop")
    parser.add_argument("--input", required=True, help="Path to dataset")
    parser.add_argument("--format", required=True, choices=["csv", "json", "event"], help="Dataset format")
    parser.add_argument("--output", default="reports/validation/real_world_validation_report.json", help="Output report path")
    args = parser.parse_args()

    rows = load_dataset(args.input, args.format)
    platform = StructuralSystemIntelligencePlatform(operating_mode="production")
    law_governance = StructuralLawGovernance()
    pipeline = RealWorldValidationPipeline(decision_fn=platform.update)
    report = pipeline.run(rows)

    # Feedback integration loop A: intervention memory.
    intervention_memory = platform.production.intervention_intelligence.memory
    for rec in report.get("feedback", []):
        intervention_memory.ingest_feedback_event(
            intervention_type=str(rec.get("recommended_intervention") or "other"),
            action=str(rec.get("action") or "ignored"),
            outcome_label=str(rec.get("outcome_label") or "neutral"),
        )

    # Feedback integration loop B: reliability.
    reliability = platform.production.reliability
    reliability.ingest_feedback_records(
        asset_id="validation_batch",
        step=len(rows) + 1,
        feedback_records=_build_feedback_rows(report),
    )

    # Feedback integration loop C: law governance real evidence.
    law_counts = defaultdict(lambda: {"helpful": 0, "harmful": 0, "neutral": 0})
    for step in report.get("replay", {}).get("step_logs", []):
        label = str((next((o for o in report.get("outcomes", []) if o.get("timestep") == step.get("timestep")), {}) or {}).get("outcome_label", "neutral"))
        for law_id in step.get("law_usage", []) or []:
            if label not in {"helpful", "harmful", "neutral", "recovery-associated"}:
                label = "neutral"
            bucket = "helpful" if label in {"helpful", "recovery-associated"} else "harmful" if label == "harmful" else "neutral"
            law_counts[law_id][bucket] += 1

    law_validation_summary = {}
    for law_id, c in law_counts.items():
        evidence = law_governance.ingest_real_world_validation(
            law_id=law_id,
            helpful_count=c["helpful"],
            harmful_count=c["harmful"],
            neutral_count=c["neutral"],
        )
        law_validation_summary[law_id] = {
            "support_count": int(evidence["support_count"]),
            "contradiction_count": int(evidence["contradiction_count"]),
            "real_world_validation_score": round(float(evidence["validation_score"]), 6),
            "intervention_outcome_consistency": round(c["helpful"] / max(1, sum(c.values())), 6),
        }

    report["real_world_validation"]["law_validation"] = law_validation_summary
    report["evidence_summaries"] = {
        "intervention_memory": intervention_memory.support_summary(),
        "reliability_store": reliability.store.inspect(limit=32),
        "law_validation": law_validation_summary,
    }

    platform.set_real_world_validation(report["real_world_validation"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
