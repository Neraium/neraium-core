"""Signal generation orchestration for Neraium Markets."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from neraium_core.markets.evidence.review_schema import EvidenceRecord, ScoreBundle, SignalOutput
from neraium_core.markets.regime.confidence import score_confidence
from neraium_core.markets.regime.interpretive_gate import should_emit_signal
from neraium_core.markets.regime.regime_rules import classify_regime, classify_state, map_action_posture
from neraium_core.markets.signals.explanation_engine import build_explanation
from neraium_core.markets.state.baseline_recent_windows import split_baseline_recent
from neraium_core.markets.structure.drift_metrics import (
    compute_drift_components,
    compute_instability_score,
    compute_structural_drift_score,
)


def generate_signal_for_asset(asset: str, timeframe: str, state: pd.DataFrame, data_quality: float) -> SignalOutput | None:
    baseline, recent = split_baseline_recent(state)
    components = compute_drift_components(baseline, recent)
    drift = compute_structural_drift_score(components)
    instability = compute_instability_score(state)
    coherence = float(np.clip(1.0 - np.std(list(components.values())), 0.0, 1.0))
    persistence = float(np.clip(recent.mean().abs().mean() * 20.0, 0.0, 1.0))
    latest = state.iloc[-1]
    breadth = float(latest.get("advancer_ratio", 0.5))
    vix_change = float(latest.get("vix_change", 0.0))
    equity_return = float(latest.get(f"ret_{asset}", 0.0))

    regime, contradiction = classify_regime(drift, instability, breadth, vix_change, equity_return)
    confidence = score_confidence(coherence, persistence, data_quality, contradiction)

    if not should_emit_signal(confidence, drift, instability):
        return None

    state_label = classify_state(drift, instability, coherence)
    risk_score = float(np.clip(0.6 * instability + 0.4 * drift, 0.0, 1.0))
    opp_score = float(np.clip((1.0 - risk_score) * confidence, 0.0, 1.0))
    action = map_action_posture(regime, confidence)
    explanation = build_explanation(asset, regime.value, components, breadth, vix_change)

    signal = SignalOutput(
        signal_id=f"sig_{uuid4().hex[:12]}",
        asset=asset,
        timeframe=timeframe,
        timestamp=datetime.now(timezone.utc),
        state=state_label,
        regime=regime,
        scores=ScoreBundle(
            structural_drift_score=drift,
            instability_score=instability,
            coherence_score=coherence,
            confidence_score=confidence,
            opportunity_score=opp_score,
            risk_score=risk_score,
        ),
        action_posture=action,
        explanation=explanation,
        evidence_refs=["drift_components", "state_latest"],
        evidence_payload={
            "top_components": sorted(components.items(), key=lambda x: x[1], reverse=True)[:3],
            "latest_features": latest.to_dict(),
        },
    )
    return signal


def to_evidence(signal: SignalOutput, input_snapshot: dict) -> EvidenceRecord:
    return EvidenceRecord(signal=signal, input_snapshot=input_snapshot, created_at=datetime.now(timezone.utc))
