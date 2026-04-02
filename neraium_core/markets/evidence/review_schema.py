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


class EvidenceRecord(BaseModel):
    signal: SignalOutput
    input_snapshot: dict[str, Any]
    created_at: datetime
