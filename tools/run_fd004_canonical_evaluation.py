#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neraium_core.fd004_canonical_evaluation import run_fd004_canonical_evaluation


def main() -> int:
    p = argparse.ArgumentParser(description="Run canonical FD004 evaluation metrics")
    p.add_argument("--num-units", type=int, default=20)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--confidence-threshold", type=float, default=0.90)
    p.add_argument("--early-lead-cycles", type=int, default=15)
    p.add_argument("--warmup-cycles", type=int, default=20)
    p.add_argument("--output-dir", type=str, default="fd004_outputs")
    args = p.parse_args()

    report = run_fd004_canonical_evaluation(
        num_units=args.num_units,
        num_steps=args.num_steps,
        seed=args.seed,
        confidence_threshold=args.confidence_threshold,
        early_lead_cycles=args.early_lead_cycles,
        warmup_cycles=args.warmup_cycles,
        output_dir=args.output_dir,
    )

    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
