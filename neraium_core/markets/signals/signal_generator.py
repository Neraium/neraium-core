"""Signal generation orchestration for Neraium Markets."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from neraium_core.engine_stages import (
    ActionGovernanceInput,
    DecisionValidityInput,
    FragilityInput,
    RiskGuidanceInput,
    evaluate_action_governance,
    evaluate_decision_validity,
    evaluate_fragility,
    evaluate_risk_guidance,
)
from neraium_core.markets.evidence.review_schema import (
    ActionGovernanceSummary,
    CompactTraderOutput,
    DecisionValiditySummary,
    EvidenceRecord,
    FragilitySummary,
    RiskGuidanceSummary,
    ScoreBundle,
    SignalOutput,
)
from neraium_core.markets.regime.confidence import score_confidence
from neraium_core.markets.regime.regime_rules import classify_regime, classify_state, map_action_posture
from neraium_core.markets.signals.explanation_engine import build_explanation
from neraium_core.markets.state.baseline_recent_windows import split_baseline_recent
from neraium_core.markets.structure.drift_metrics import (
    compute_drift_components,
    compute_instability_score,
    compute_structural_drift_score,
)


def _build_context_notes(asset: str, breadth: float, vix_change: float, equity_return: float) -> list[str]:
    notes: list[str] = []
    if breadth < 0.45:
        notes.append("breadth support is narrow")
    elif breadth > 0.58:
        notes.append("breadth support is broad")
    if vix_change > 0.02:
        notes.append("volatility context is expanding")
    elif vix_change < -0.01:
        notes.append("volatility context is easing")
    if equity_return > 0.003:
        notes.append(f"{asset} return leadership is positive")
    elif equity_return < -0.003:
        notes.append(f"{asset} return leadership is negative")
    return notes


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

    state_label = classify_state(drift, instability, coherence)
    risk_score = float(np.clip(0.6 * instability + 0.4 * drift, 0.0, 1.0))
    opp_score = float(np.clip((1.0 - risk_score) * confidence, 0.0, 1.0))
    context_notes = _build_context_notes(asset, breadth, vix_change, equity_return)
    decision_validity = evaluate_decision_validity(
        DecisionValidityInput(
            asset=asset,
            timeframe=timeframe,
            regime=regime.value,
            state=state_label.value,
            structural_drift_score=drift,
            instability_score=instability,
            coherence_score=coherence,
            confidence_score=confidence,
            data_quality_score=data_quality,
            persistence_score=persistence,
            contradiction_penalty=contradiction,
            breadth_score=breadth,
            vix_change=vix_change,
            sample_count=len(state),
            drift_components=components,
            context_notes=context_notes,
        )
    )
    fragility = evaluate_fragility(
        FragilityInput(
            regime=regime.value,
            state=state_label.value,
            drift_score=drift,
            instability_score=instability,
            coherence_score=coherence,
            confidence_score=confidence,
            data_quality_score=data_quality,
            validity_score=decision_validity.validity_score,
            transition_state=decision_validity.transition_state,
            breadth_score=breadth,
            vix_change=vix_change,
            invalidating_factors=decision_validity.invalidating_factors,
        )
    )
    action_governance = evaluate_action_governance(
        ActionGovernanceInput(
            regime=regime.value,
            state=state_label.value,
            action_permission=decision_validity.action_permission,
            validity_score=decision_validity.validity_score,
            confidence_score=confidence,
            fragility_score=fragility.fragility_score,
            structural_drift_score=drift,
            instability_score=instability,
            transition_state=decision_validity.transition_state,
            abstain=decision_validity.abstain,
            invalidation_conditions=fragility.invalidation_conditions,
        )
    )
    regime_stability = float(np.clip(1.0 - 0.65 * drift - 0.35 * instability, 0.0, 1.0))
    risk_guidance = evaluate_risk_guidance(
        RiskGuidanceInput(
            action_permission=decision_validity.action_permission,
            validity_score=decision_validity.validity_score,
            confidence_score=confidence,
            fragility_score=fragility.fragility_score,
            transition_state=decision_validity.transition_state,
            data_quality_score=data_quality,
            regime_stability_score=regime_stability,
        )
    )
    action = map_action_posture(regime, confidence)
    trader_timestamp = datetime.now(timezone.utc)
    trader_output = CompactTraderOutput(
        timestamp=trader_timestamp,
        ticker=asset,
        market_state=regime.value,
        transition_state=decision_validity.transition_state,
        action_permission=decision_validity.action_permission,
        best_action=action_governance.best_action,
        trading_signal=f"{decision_validity.action_permission}:{action_governance.best_action}",
        confidence=confidence,
        fragility=fragility.fragility_score,
        validity_score=decision_validity.validity_score,
        structural_drift_score=drift,
        latest_instability=instability,
        one_line_reason=decision_validity.justification,
        invalidation_conditions=fragility.invalidation_conditions[:4],
        size_guidance=risk_guidance.exposure_guidance,
        cooldown_status=None,
    )
    explanation = build_explanation(asset, regime.value, components, breadth, vix_change)

    signal = SignalOutput(
        signal_id=f"sig_{uuid4().hex[:12]}",
        asset=asset,
        timeframe=timeframe,
        timestamp=trader_timestamp,
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
            "context_notes": context_notes,
        },
        decision_validity=DecisionValiditySummary.model_validate(asdict(decision_validity)),
        action_governance=ActionGovernanceSummary.model_validate(asdict(action_governance)),
        fragility=FragilitySummary.model_validate(asdict(fragility)),
        risk_guidance=RiskGuidanceSummary.model_validate(asdict(risk_guidance)),
        trader_output=trader_output,
    )
    return signal


def to_evidence(signal: SignalOutput, input_snapshot: dict) -> EvidenceRecord:
    return EvidenceRecord(signal=signal, input_snapshot=input_snapshot, created_at=datetime.now(timezone.utc))
