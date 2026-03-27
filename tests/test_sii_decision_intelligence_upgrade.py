from __future__ import annotations

import math

import numpy as np

from neraium_core.sii import SIIConfig, SIIEngine
from neraium_core.sii.attribution import build_structural_attribution
from neraium_core.sii.regime_memory import summarize_regime_memory
from neraium_core.sii.regime_model import RegimeModel, RegimeObservation
from neraium_core.sii.risk import assess_forward_risk


def test_structural_attribution_returns_grounded_fields() -> None:
    names = ["A", "B", "C"]
    baseline_corr = np.array(
        [
            [1.0, 0.85, 0.8],
            [0.85, 1.0, 0.78],
            [0.8, 0.78, 1.0],
        ]
    )
    recent_corr = np.array(
        [
            [1.0, 0.10, -0.70],
            [0.10, 1.0, 0.22],
            [-0.70, 0.22, 1.0],
        ]
    )
    base_adj = (np.abs(baseline_corr) >= 0.6).astype(float)
    np.fill_diagonal(base_adj, 0.0)
    recent_adj = (np.abs(recent_corr) >= 0.6).astype(float)
    np.fill_diagonal(recent_adj, 0.0)

    out = build_structural_attribution(
        sensor_names=names,
        baseline_corr=baseline_corr,
        recent_corr=recent_corr,
        baseline_adj=base_adj,
        recent_adj=recent_adj,
        baseline_mean=np.array([0.0, 0.0, 0.0]),
        recent_mean=np.array([0.1, 0.8, 0.7]),
        structural_score=0.8,
        relational_score=0.7,
        graph_score=0.6,
        coupling_score=0.65,
        recent_composite=[0.2, 0.24, 0.3, 0.37, 0.45, 0.53, 0.62],
    )

    assert len(out.top_sensors) >= 1
    assert out.top_sensors[0]["sensor"] in {"A", "B", "C"}
    assert any(r["change"] in {"break", "weakened", "strengthened", "new"} for r in out.top_relationships)
    assert "primary_cluster" in out.subsystem_impact
    assert out.change_character["scope"] in {"local", "distributed"}
    assert out.change_character["tempo"] in {"abrupt", "gradual"}


def test_regime_similarity_reports_novelty_and_matches(tmp_path) -> None:
    cfg = SIIConfig(regime_store_path=str(tmp_path / "regimes.json"))
    model = RegimeModel(config=cfg)

    g0 = np.array([0.0, 0.1, 0.2], dtype=float)
    gr0 = np.array([0.1, 0.2], dtype=float)
    model.observe(RegimeObservation(geometry_signature=g0, graph_signature=gr0, feature_names=["s1", "s2"]))

    matches = model.nearest_matches(geometry_signature=g0, graph_signature=gr0, top_k=2)
    summary = summarize_regime_memory(current_regime="regime_0", nearest_matches=matches, threshold=cfg.regime_distance_threshold)

    assert summary.current_regime == "regime_0"
    assert summary.nearest_matches
    assert summary.best_similarity > 0.8
    assert summary.is_novel is False


def test_forward_risk_flags_deteriorating_trend() -> None:
    out = assess_forward_risk(
        composite_history=[0.2, 0.26, 0.3, 0.36, 0.42, 0.5, 0.58, 0.64],
        structural_score=0.62,
        regime_score=0.55,
        coupling_score=0.61,
    )
    assert out.current_risk_level in {"medium", "high"}
    assert out.projected_near_term_trend == "increasing"
    assert out.trajectory == "deteriorating"
    assert out.projected_score >= out.risk_score


def test_end_to_end_engine_emits_decision_intelligence_fields(tmp_path) -> None:
    engine = SIIEngine(
        SIIConfig(
            baseline_window=24,
            recent_window=8,
            regime_store_path=str(tmp_path / "sii_regimes.json"),
            min_samples_for_alerts=20,
        )
    )

    warmup_out = engine.process_payload(
        {
            "timestamp": "0",
            "site_id": "site-A",
            "asset_id": "asset-A",
            "sensor_values": {"s1": 0.0, "s2": 0.0, "s3": 0.0},
        }
    )
    assert "attribution" in warmup_out
    assert "regime_memory" in warmup_out
    assert "risk_assessment" in warmup_out
    assert "operator_guidance" in warmup_out
    assert "causal_analysis" in warmup_out
    assert "decision" in warmup_out
    assert warmup_out["attribution"]["status"] in {"warming_up", "insufficient_history"}
    assert warmup_out["regime_memory"]["status"] in {"warming_up", "insufficient_history"}
    assert warmup_out["risk_assessment"]["status"] in {"warming_up", "insufficient_history"}
    assert warmup_out["operator_guidance"]["status"] in {"warming_up", "insufficient_history"}
    assert warmup_out["causal_analysis"]["status"] in {"warming_up", "insufficient_history"}
    assert warmup_out["decision"]["status"]["available"] is False
    assert "structural_attribution" not in warmup_out
    assert "forward_risk" not in warmup_out

    out = None
    for t in range(1, 80):
        base = math.sin(0.05 * t)
        if t < 40:
            payload = {
                "timestamp": str(t),
                "site_id": "site-A",
                "asset_id": "asset-A",
                "sensor_values": {"s1": base, "s2": 0.95 * base, "s3": 1.05 * base},
            }
        else:
            payload = {
                "timestamp": str(t),
                "site_id": "site-A",
                "asset_id": "asset-A",
                "sensor_values": {"s1": base, "s2": ((-1) ** t) * 0.7 * base + 0.4, "s3": 0.2 * base - 0.5},
            }
        out = engine.process_payload(payload)

    assert out is not None
    assert "attribution" in out
    assert "regime_memory" in out
    assert "risk_assessment" in out
    assert "operator_guidance" in out
    assert "causal_analysis" in out
    assert "decision" in out
    assert "structural_attribution" not in out
    assert "forward_risk" not in out

    attribution = out["attribution"]
    assert attribution["status"] == "ready"
    assert attribution["top_sensors"]
    assert attribution["top_relationships"]
    assert out["regime_memory"]["status"] == "ready"
    assert out["risk_assessment"]["status"] == "ready"

    guidance = out["operator_guidance"]
    assert guidance["status"] == "ready"
    assert guidance["recommended_actions"]
    assert out["causal_analysis"]["status"] == "ready"
    assert out["decision"]["status"]["available"] is True
