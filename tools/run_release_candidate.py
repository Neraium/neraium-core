from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_validation import execute_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical release-candidate evaluation")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--corpus-root", default="validation/corpus")
    parser.add_argument("--history-root", default="reports/validation/history")
    parser.add_argument("--output", default="reports/validation/real_world_validation_report.json")
    parser.add_argument("--core-output", default="reports/validation/core_validation_report.json")
    parser.add_argument("--release-gate-output", default="reports/validation/release_gate_report.json")
    parser.add_argument("--mark-baseline", action="store_true")
    args = parser.parse_args()

    run_args = argparse.Namespace(
        input=None,
        format=None,
        corpus_id=args.corpus_id,
        corpus_root=args.corpus_root,
        output=args.output,
        core_output=args.core_output,
        release_gate_output=args.release_gate_output,
        prior_core_report=None,
        history_root=args.history_root,
        write_history=True,
        mark_baseline=args.mark_baseline,
    )

    report = execute_validation(run_args)
    gate = report["release_gate_report"]
    status = "RELEASE_PASS" if gate.get("release_passed") else "RELEASE_FAIL"
    print(status)
    print(json.dumps({
        "status": status,
        "blocking_reasons": gate.get("blocking_reasons", []),
        "summary_metrics": report.get("core_validation_report", {}).get("summary", {}),
        "run_metadata": report.get("run_metadata", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
