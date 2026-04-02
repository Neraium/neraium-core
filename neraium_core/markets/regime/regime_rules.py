"""Rule-based market regime classifier."""

from __future__ import annotations

from neraium_core.markets.evidence.review_schema import ActionPosture, RegimeLabel, StateLabel


def classify_state(drift: float, instability: float, coherence: float) -> StateLabel:
    if instability > 0.75:
        return StateLabel.UNSTABLE
    if drift > 0.7 and instability > 0.5:
        return StateLabel.TRANSITIONING
    if drift > 0.6:
        return StateLabel.DIVERGING
    if drift > 0.4:
        return StateLabel.DRIFTING
    if drift < 0.25 and coherence > 0.7:
        return StateLabel.STABLE
    return StateLabel.RESTABILIZING


def classify_regime(
    drift: float,
    instability: float,
    breadth: float,
    vix_change: float,
    equity_return: float,
) -> tuple[RegimeLabel, float]:
    contradiction = 0.0
    if drift > 0.65 and instability > 0.55 and breadth < 0.45 and vix_change > 0.0:
        return RegimeLabel.RISK_OFF_TRANSITION, contradiction
    if drift > 0.6 and breadth < 0.5 and equity_return > 0:
        return RegimeLabel.FRAGILE_RALLY, 0.1
    if instability > 0.7 and vix_change > 0.02:
        return RegimeLabel.HIGH_VOL_TRANSITION, contradiction
    if drift < 0.3 and instability < 0.3 and breadth > 0.55:
        return RegimeLabel.TREND, contradiction
    if drift < 0.35 and instability < 0.4 and abs(equity_return) < 0.002:
        return RegimeLabel.MEAN_REVERSION, contradiction
    if drift < 0.25 and instability < 0.25 and breadth < 0.45:
        return RegimeLabel.FALSE_CALM, 0.15
    if drift > 0.5 and instability > 0.6 and breadth < 0.4:
        return RegimeLabel.LIQUIDITY_STRESS, contradiction
    return RegimeLabel.CROWDED_TREND, 0.05


def map_action_posture(regime: RegimeLabel, confidence: float) -> ActionPosture:
    if confidence < 0.5:
        return ActionPosture.WAIT
    mapping = {
        RegimeLabel.TREND: ActionPosture.FAVOR_LONG,
        RegimeLabel.CROWDED_TREND: ActionPosture.WATCH,
        RegimeLabel.MEAN_REVERSION: ActionPosture.WATCH,
        RegimeLabel.FRAGILE_RALLY: ActionPosture.REDUCE,
        RegimeLabel.RISK_OFF_TRANSITION: ActionPosture.AVOID_RISK,
        RegimeLabel.LIQUIDITY_STRESS: ActionPosture.HEDGE,
        RegimeLabel.FALSE_CALM: ActionPosture.WAIT,
        RegimeLabel.HIGH_VOL_TRANSITION: ActionPosture.REDUCE,
    }
    return mapping[regime]
