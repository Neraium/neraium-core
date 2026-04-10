from __future__ import annotations

from pathlib import Path
from pprint import pprint
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.config import UIConfig
from ui.core_integration import build_system_state
from ui.layouts.operations_view import build_operations_view


def _row(**overrides):
    base = {
        "timestamp": "2026-04-10T00:00:00Z",
        "regime_name": "transition",
        "system_health": "watch",
        "confidence_score": 0.74,
        "structural_drift_score": 0.62,
        "relational_stability_score": 0.39,
    }
    base.update(overrides)
    return base


def run_demo() -> None:
    rows = [_row()]
    state = build_system_state(rows, config=UIConfig())

    scenarios = [
        {
            "name": "clear_admit",
            "gate_decision": {
                "decision": "ADMIT",
                "doctrine_version": "doctrine.v1",
                "timestamp": "2026-04-10T00:10:00Z",
                "confidence_label": "high",
                "reason": "Persistent corroborated structural reorganization meets doctrine admission criteria.",
                "criteria_summary": ["coherence confirmed", "persistence confirmed", "multiple corroborating signals"],
            },
        },
        {
            "name": "suppress",
            "gate_decision": {
                "decision": "SUPPRESS",
                "doctrine_version": "doctrine.v1",
                "timestamp": "2026-04-10T00:15:00Z",
                "confidence_label": "medium",
                "refusal_reason": "Signal is transient and lacks corroborating evidence.",
                "criteria_summary": ["insufficient persistence", "single-source evidence"],
            },
        },
        {
            "name": "admissibility_void",
            "gate_decision": {
                "decision": "ADMISSIBILITY_VOID",
                "doctrine_version": "doctrine.v1",
                "timestamp": "2026-04-10T00:20:00Z",
                "confidence_label": "low",
                "refusal_reason": "Evidence stream is incoherent and cannot be admitted.",
                "criteria_summary": ["coherence breakdown", "instrument instability"],
            },
        },
    ]

    for scenario in scenarios:
        surface = build_operations_view(
            state,
            reasoning_context={"recent_admitted_events": []},
            gate_decision=scenario["gate_decision"],
        )
        print(f"\n=== UI DEMO: {scenario['name']} ===")
        pprint(surface["zones"]["gate"]["content"], sort_dicts=False)


if __name__ == "__main__":
    run_demo()
