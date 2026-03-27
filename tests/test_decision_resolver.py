from __future__ import annotations

import ast
import math
from pathlib import Path

from neraium_core.decision_resolver import compute_decision_confidence, resolve_best_action
from neraium_core.sii import SIIConfig, SIIEngine


def _base_inputs() -> dict[str, dict]:
    return {
        "attribution": {
            "top_sensors": [{"sensor": "pump_A", "score": 0.82}],
            "contribution_scores": {"pump_A": 0.82, "pump_B": 0.35},
        },
        "regime_memory": {"status": "ready", "is_novel": False},
        "risk_assessment": {
            "status": "ready",
            "projected_near_term_trend": "increasing",
            "risk_score": 0.68,
            "projected_score": 0.76,
        },
        "operator_guidance": {
            "status": "ready",
            "first_inspection_target": "cluster_A",
            "recommended_actions": ["Inspect subsystem/cluster first: cluster_A."],
        },
    }


def test_resolver_prefers_best_next_action() -> None:
    payload = _base_inputs()
    payload["causal_analysis"] = {
        "status": "ready",
        "best_next_action": {
            "action": "Inspect cooling manifold",
            "target": "subsystem_X",
            "priority": 1,
            "confidence": 0.73,
        },
        "recommended_sequence": [
            {"action": "Fallback action", "target": "subsystem_Y", "priority": 2},
        ],
        "top_hypotheses": [{"hypothesis": "x", "confidence": 0.71, "robustness": 0.66, "counterfactual_strength": 0.58}],
    }
    out = resolve_best_action(**payload)
    assert out["status"]["available"] is True
    assert out["action"] == "Inspect cooling manifold"
    assert out["target"] == "subsystem_X"
    assert out["source"]["from_causal_analysis"] is True


def test_resolver_falls_back_to_recommended_sequence() -> None:
    payload = _base_inputs()
    payload["causal_analysis"] = {
        "status": "ready",
        "best_next_action": None,
        "recommended_sequence": [
            {"action": "Inspect redundancy pair", "target": "cluster_B", "priority": 1},
        ],
        "top_hypotheses": [{"hypothesis": "y", "confidence": 0.63, "robustness": 0.6, "counterfactual_strength": 0.55}],
    }
    out = resolve_best_action(**payload)
    assert out["status"]["available"] is True
    assert out["action"] == "Inspect redundancy pair"
    assert out["target"] == "cluster_B"


def test_resolver_falls_back_to_operator_guidance_without_causal_action() -> None:
    payload = _base_inputs()
    payload["causal_analysis"] = {"status": "ready", "best_next_action": None, "recommended_sequence": [], "top_hypotheses": []}
    out = resolve_best_action(**payload)
    assert out["status"]["available"] is True
    assert out["action"] == "Inspect subsystem/cluster first: cluster_A."
    assert out["source"]["from_operator_guidance"] is True


def test_resolver_structured_fallback_during_warmup() -> None:
    payload = _base_inputs()
    payload["causal_analysis"] = {"status": "warming_up", "best_next_action": None, "recommended_sequence": [], "top_hypotheses": []}
    out = resolve_best_action(**payload, readiness={"ready": False, "reason": "Warmup window is incomplete."})
    assert out["status"]["available"] is False
    assert out["action"] is None
    assert out["confidence"] <= 0.05
    assert "Warmup" in out["reason"]


def test_confidence_is_deterministic_and_bounded() -> None:
    source = {
        "from_causal_analysis": True,
        "from_operator_guidance": True,
        "from_risk_assessment": True,
        "from_attribution": True,
    }
    causal = {
        "top_hypotheses": [
            {
                "confidence": 0.77,
                "robustness": 0.68,
                "counterfactual_strength": 0.61,
            }
        ]
    }
    risk = {"projected_near_term_trend": "increasing", "risk_score": 0.62, "projected_score": 0.71}
    attribution = {"top_sensors": [{"sensor": "A", "score": 0.8}], "contribution_scores": {"A": 0.8}}

    c1 = compute_decision_confidence(causal_analysis=causal, risk_assessment=risk, attribution=attribution, source=source)
    c2 = compute_decision_confidence(causal_analysis=causal, risk_assessment=risk, attribution=attribution, source=source)
    assert math.isclose(c1, c2, rel_tol=0.0, abs_tol=1e-12)
    assert 0.0 <= c1 <= 1.0


def test_evidence_and_source_fields_are_populated() -> None:
    payload = _base_inputs()
    payload["causal_analysis"] = {
        "status": "ready",
        "best_next_action": {"action": "Inspect cooling manifold", "target": "subsystem_X", "priority": 1, "confidence": 0.7},
        "recommended_sequence": [],
        "top_hypotheses": [{"hypothesis": "localized degradation", "confidence": 0.7, "robustness": 0.65, "counterfactual_strength": 0.5}],
    }
    out = resolve_best_action(**payload)
    assert out["evidence"]
    assert out["source"]["from_causal_analysis"] is True
    assert out["source"]["from_risk_assessment"] is True


def test_engine_process_frame_emits_decision_in_warmup_and_ready(tmp_path: Path) -> None:
    engine = SIIEngine(
        SIIConfig(
            baseline_window=16,
            recent_window=8,
            regime_store_path=str(tmp_path / "sii_regimes.json"),
            min_samples_for_alerts=16,
        )
    )

    warm = engine.process_payload(
        {
            "timestamp": "0",
            "site_id": "site-A",
            "asset_id": "asset-A",
            "sensor_values": {"s1": 0.0, "s2": 0.0, "s3": 0.0},
        }
    )
    assert "decision" in warm
    assert warm["decision"]["status"]["available"] is False
    assert "causal_analysis" in warm

    out = None
    for t in range(1, 70):
        base = math.sin(0.08 * t)
        payload = {
            "timestamp": str(t),
            "site_id": "site-A",
            "asset_id": "asset-A",
            "sensor_values": {
                "s1": base,
                "s2": (0.95 * base) if t < 35 else ((-1) ** t) * 0.85 * base + 0.35,
                "s3": (1.03 * base) if t < 35 else (0.2 * base - 0.55),
            },
        }
        out = engine.process_payload(payload)

    assert out is not None
    assert "decision" in out
    assert "causal_analysis" in out
    assert out["decision"]["status"]["available"] is True
    assert out["decision"]["action"] is not None
    causal_action = ((out.get("causal_analysis") or {}).get("best_next_action") or {}).get("action")
    if causal_action:
        assert out["decision"]["action"] == causal_action
    else:
        assert out["decision"]["action"] == out["operator_guidance"]["recommended_actions"][0]


def test_internal_modules_do_not_import_casual_shim() -> None:
    root = Path("neraium_core")
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "casual.py":
            continue
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module == "neraium_core.casual":
                violations.append(str(path))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "neraium_core.casual":
                        violations.append(str(path))
    assert violations == []
