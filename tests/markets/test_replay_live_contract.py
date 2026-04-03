from __future__ import annotations

from pathlib import Path

from neraium_core.markets.evidence.evidence_log import EvidenceLog
from neraium_core.markets.replay import run_signal_replay

REQUIRED_FIELDS = {
    "timestamp",
    "ticker",
    "action_permission",
    "best_action",
    "confidence",
    "fragility",
    "validity_score",
    "size_guidance",
    "one_line_reason",
    "invalidation_conditions",
    "market_state",
    "transition_state",
    "structural_drift_score",
    "latest_instability",
    "cooldown_status",
}


def test_replay_output_contract_fields_present(tmp_path: Path):
    evidence = EvidenceLog(tmp_path / "evidence.jsonl")
    rows = run_signal_replay(
        Path("neraium_core/markets/sample_data"),
        evidence,
        timeframe="15m",
        symbols=["SPY", "QQQ", "IWM"],
    )
    assert rows
    missing = REQUIRED_FIELDS - set(rows[0].keys())
    assert not missing
