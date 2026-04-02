"""End-to-end script for Neraium Markets signal generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neraium_core.markets.app.api import run_signal_pipeline
from neraium_core.markets.evidence.evidence_log import EvidenceLog


def main() -> None:
    data_dir = Path("neraium_core/markets/sample_data")
    evidence = EvidenceLog("artifacts/neraium_markets/evidence.jsonl")
    signals = run_signal_pipeline(data_dir, evidence)
    out = Path("artifacts/neraium_markets/latest_signals.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(signals, indent=2), encoding="utf-8")
    print(f"Generated {len(signals)} signals -> {out}")


if __name__ == "__main__":
    main()
