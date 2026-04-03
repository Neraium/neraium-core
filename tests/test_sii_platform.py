from __future__ import annotations

import math
from pathlib import Path

from neraium_core.sii import SIIEngine, SIIConfig, write_json_report


def _frame(t: int, s1: float, s2: float, s3: float) -> dict[str, object]:
    return {
        "timestamp": str(t),
        "site_id": "site-A",
        "asset_id": "asset-A",
        "sensor_values": {"s1": s1, "s2": s2, "s3": s3},
    }


def test_sii_nominal_structure_and_allowed_outputs(tmp_path: Path) -> None:
    engine = SIIEngine(
        SIIConfig(
            baseline_window=24,
            recent_window=8,
            regime_store_path=str(tmp_path / "sii_regimes.json"),
        )
    )
    out: dict[str, object] | None = None
    for t in range(70):
        base = math.sin(0.06 * t)
        out = engine.process_payload(_frame(t, base, 0.95 * base, 1.03 * base))

    assert out is not None
    assert out["state"] in {"STABLE", "WATCH", "ALERT"}
    assert out["interpreted_state"] in {
        "NOMINAL_STRUCTURE",
        "REGIME_SHIFT_OBSERVED",
        "COUPLING_INSTABILITY_OBSERVED",
        "STRUCTURAL_INSTABILITY_OBSERVED",
        "COHERENCE_UNDER_CONSTRAINT",
    }
    assert out["confidence"] in {"low", "medium", "high"}
    assert "structural_drift_score" in out
    assert "relational_instability_score" in out
    assert "regime_distance" in out
    assert "graph_deformation_score" in out
    assert "coherence_score" in out
    assert "dominant_drivers" in out


def test_sii_report_export_json(tmp_path: Path) -> None:
    engine = SIIEngine(
        SIIConfig(
            baseline_window=18,
            recent_window=6,
            regime_store_path=str(tmp_path / "sii_regimes.json"),
        )
    )
    outputs = []
    for t in range(40):
        base = math.sin(0.05 * t)
        outputs.append(engine.process_payload(_frame(t, base, 0.9 * base, 1.0 * base)))

    out_file = tmp_path / "report.json"
    write_json_report(str(out_file), outputs)
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").strip().startswith("[")
