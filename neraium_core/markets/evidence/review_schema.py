"""Pydantic schemas for signal and evidence review."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StateLabel(str, Enum):
    STABLE = "stable"
    DRIFTING = "drifting"
    DIVERGING = "diverging"
    UNSTABLE = "unstable"
    TRANSITIONING = "transitioning"
    RESTABILIZING = "restabilizing"


class RegimeLabel(str, Enum):
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    CROWDED_TREND = "crowded_trend"
    FRAGILE_RALLY = "fragile_rally"
    RISK_OFF_TRANSITION = "risk_off_transition"
    LIQUIDITY_STRESS = "liquidity_stress"
    FALSE_CALM = "false_calm"
    HIGH_VOL_TRANSITION = "high_vol_transition"


class ActionPosture(str, Enum):
    WATCH = "watch"
    FAVOR_LONG = "favor_long_setups"
    FAVOR_SHORT = "favor_short_setups"
    REDUCE = "reduce_exposure"
    AVOID_RISK = "avoid_new_risk"
    HEDGE = "hedge_bias"
    WAIT = "wait_for_confirmation"


class ScoreBundle(BaseModel):
    structural_drift_score: float = Field(ge=0.0, le=1.0)
    instability_score: float = Field(ge=0.0, le=1.0)
    coherence_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    opportunity_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)


class DecisionValiditySummary(BaseModel):
    is_valid: bool
    action_permission: str
    validity_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    fragility_score: float = Field(ge=0.0, le=1.0)
    abstain: bool
    invalidating_factors: list[str]
    justification: str
    structural_summary: str
    transition_state: str


class ActionAssessment(BaseModel):
    action_type: str
    validity: bool
    confidence: float = Field(ge=0.0, le=1.0)
    fragility: float = Field(ge=0.0, le=1.0)
    reason: str
    invalidation_conditions: list[str]


class ActionGovernanceSummary(BaseModel):
    allowed_actions: list[ActionAssessment]
    blocked_actions: list[ActionAssessment]
    best_action: str
    action_quality: dict[str, float]
    action_notes: dict[str, str]


class FragilitySummary(BaseModel):
    fragility_score: float = Field(ge=0.0, le=1.0)
    fragility_state: str
    fragility_sources: list[str]
    invalidation_conditions: list[str]
    dependency_map: dict[str, float]
    edge_dependencies: list[str]
    why_not_robust: str
    edge_stability_window: str


class RiskGuidanceSummary(BaseModel):
    exposure_guidance: str
    risk_budget_score: float = Field(ge=0.0, le=1.0)
    size_multiplier: float = Field(ge=0.0, le=1.0)
    rationale: str


class CompactTraderOutput(BaseModel):
    timestamp: datetime
    ticker: str
    market_state: str
    transition_state: str
    action_permission: str
    best_action: str
    trading_signal: str
    confidence: float = Field(ge=0.0, le=1.0)
    fragility: float = Field(ge=0.0, le=1.0)
    validity_score: float = Field(ge=0.0, le=1.0)
    structural_drift_score: float = Field(ge=0.0, le=1.0)
    latest_instability: float = Field(ge=0.0, le=1.0)
    one_line_reason: str
    invalidation_conditions: list[str]
    size_guidance: str | None = None
    cooldown_status: str | None = None


class SignalOutput(BaseModel):
    signal_id: str
    asset: str
    timeframe: str
    timestamp: datetime
    state: StateLabel
    regime: RegimeLabel
    scores: ScoreBundle
    action_posture: ActionPosture
    explanation: str
    evidence_refs: list[str]
    evidence_payload: dict[str, Any]
    decision_validity: DecisionValiditySummary | None = None
    action_governance: ActionGovernanceSummary | None = None
    fragility: FragilitySummary | None = None
    risk_guidance: RiskGuidanceSummary | None = None
    trader_output: CompactTraderOutput | None = None


class EvidenceRecord(BaseModel):
    signal: SignalOutput
    input_snapshot: dict[str, Any]
    created_at: datetime
