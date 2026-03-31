from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from neraium_core.sii.config import SIIConfig
from neraium_core.sii.engine import SystemicInfrastructureIntelligenceEngine
from neraium_core.sii.types import SIIFrame


def _frame(ts: datetime, s1: float, s2: float, s3: float) -> SIIFrame:
    return SIIFrame(
        timestamp=ts.isoformat(),
        site_id="site-a",
        system_id="sys-a",
        asset_id="asset-a",
        node_id="node-a",
        sensor_values={"s1": s1, "s2": s2, "s3": s3},
    )


def test_structural_system_intelligence_layer_integrates_and_responds() -> None:
    cfg = SIIConfig(
        baseline_window=12,
        recent_window=6,
        max_history=160,
        min_samples_for_alerts=12,
    )
    engine = SystemicInfrastructureIntelligenceEngine(config=cfg)

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stable_out = None
    unstable_out = None

    for i in range(55):
        t = start + timedelta(minutes=i)
        base = np.sin(i / 6.0)
        if i < 34:
            s1 = base
            s2 = 0.95 * base + 0.02
            s3 = -0.85 * base
        else:
            # Regime break: decouple/flip relationships + stronger variance.
            s1 = base + 0.2 * np.sin(i)
            s2 = -0.35 * base + (0.8 if i % 3 == 0 else -0.8)
            s3 = 0.9 * base + 0.4 * np.cos(i / 2.0)

        out = engine.process_frame(_frame(t, s1, s2, s3))
        if i == 33:
            stable_out = out
        if i == 54:
            unstable_out = out

    assert stable_out is not None and unstable_out is not None

    intelligence = unstable_out.get("structural_system_intelligence")
    assert isinstance(intelligence, dict)
    assert set(intelligence.keys()) >= {
        "production_intelligence",
        "advisory_intelligence",
        "experimental_intelligence",
        "compatibility",
        "capability_boundaries",
    }
    assert set(intelligence["production_intelligence"].keys()) >= {
        "latent_structural_state",
        "transition_dynamics",
        "counterfactuals",
        "archetype_intelligence",
    }
    assert intelligence["advisory_intelligence"] == {}

    stable_prob = float(stable_out["structural_system_intelligence"]["transition_dynamics"].get("escalation_probability", 0.0))
    unstable_prob = float(unstable_out["structural_system_intelligence"]["transition_dynamics"].get("escalation_probability", 0.0))
    assert unstable_prob >= stable_prob

    latent = intelligence["production_intelligence"]["latent_structural_state"]
    assert len(latent.get("embedding", [])) == 3
    assert len(latent.get("trajectory", [])) >= 2

    counterfactual = intelligence["production_intelligence"]["counterfactuals"]
    observed = float(counterfactual["observed_evidence"]["base_composite_instability"])
    projected = float(counterfactual["best_intervention"]["projected_risk_score"])
    assert projected <= observed

    archetype = intelligence["production_intelligence"]["archetype_intelligence"]
    assert archetype.get("nearest_archetypes")

    boundary = intelligence["capability_boundaries"]["latent_structural_state"]
    assert boundary["status"] == "production"
    assert boundary["can_drive_operator_recommendations"] is True

    # Legacy-facing outputs remain intact.
    assert unstable_out.get("risk_assessment")
    assert unstable_out.get("operator_guidance")
    assert unstable_out.get("causal_analysis")
