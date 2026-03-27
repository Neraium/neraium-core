"""
Neraium engine readiness — domain-general stream and subsystem state.

Separates “signals exist” from “outputs are actionable” for higher-order analytics
and transition classification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EngineReadinessState:
    """Structured readiness for structural analytics and transition classification."""

    ready: bool
    """True when baseline/recent windows are populated (Tier-1 structural context is defined)."""

    structural_ready: bool
    """Windows are full (baseline + recent available)."""

    transition_classification_ready: bool
    """True when EMERGING/SUSTAINED transition labels are meaningful (post-stabilization)."""

    history_points: int
    """Frames (or samples) observed in the current stream."""

    min_history_required: int
    """Minimum stream length before transition outputs are treated as actionable."""

    stabilization_progress: float
    """Progress toward ``min_history_required`` in [0, 1] (was ``warmup_progress``)."""

    unavailable_components: dict[str, str] = field(default_factory=dict)
    """Subsystem name -> reason when not yet available."""

    reason: str = ""
    """Human-readable summary."""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Deprecated aliases (do not remove — external integrations may rely on names)
        d["warmup_progress"] = d["stabilization_progress"]
        d["transition_classifiable"] = d["transition_classification_ready"]
        return d


def compute_engine_readiness(
    *,
    frame_count: int,
    baseline_window: int,
    recent_window: int,
    transition_pressure_history_len: int,
    warmup_margin_frames: int = 8,
    min_transition_history: int = 6,
    geometry_available: bool | None = None,
    geometry_reason: str | None = None,
    state_space_available: bool | None = None,
    state_space_reason: str | None = None,
    state_graph_available: bool | None = None,
    state_graph_reason: str | None = None,
) -> EngineReadinessState:
    """
    Compute readiness from stream counters and optional subsystem flags.

    ``transition_classification_ready`` requires:

    - structural windows full,
    - at least ``stabilization_margin_frames`` extra frames after windows first fill,
    - at least ``min_transition_history`` samples in the transition-pressure history.

    Subsystem flags are informational unless you extend callers to enforce them.
    """
    base_need = max(int(baseline_window), int(recent_window))
    stabilization_margin_frames = warmup_margin_frames
    min_history_required = base_need + max(0, int(stabilization_margin_frames))
    structural_ready = int(frame_count) >= base_need
    stabilization_progress = min(1.0, float(frame_count) / float(max(1, min_history_required)))

    unavailable: dict[str, str] = {}
    if geometry_available is False:
        unavailable["geometry"] = geometry_reason or "not_available"
    if state_space_available is False:
        unavailable["state_space_statistics"] = state_space_reason or "not_available"
    if state_graph_available is False:
        unavailable["state_graph"] = state_graph_reason or "not_available"

    transition_classification_ready = (
        structural_ready
        and int(frame_count) >= min_history_required
        and int(transition_pressure_history_len) >= int(min_transition_history)
    )

    # Trust Tier-1 structural analytics once baseline/recent windows are populated.
    ready = structural_ready

    if not structural_ready:
        reason = "awaiting_baseline_and_recent_windows"
    elif int(frame_count) < min_history_required:
        reason = "stabilization_period_incomplete"
    elif int(transition_pressure_history_len) < int(min_transition_history):
        reason = "insufficient_transition_pressure_history"
    elif unavailable:
        reason = "partial_subsystems_pending"
    else:
        reason = "ready"

    return EngineReadinessState(
        ready=bool(ready),
        structural_ready=structural_ready,
        transition_classification_ready=transition_classification_ready,
        history_points=int(frame_count),
        min_history_required=int(min_history_required),
        stabilization_progress=float(stabilization_progress),
        unavailable_components=unavailable,
        reason=reason,
    )


def readiness_dict_for_partial_stream(
    *,
    frame_count: int,
    baseline_window: int,
    recent_window: int,
    transition_pressure_history_len: int = 0,
    warmup_margin_frames: int = 8,
    min_transition_history: int = 6,
) -> dict[str, Any]:
    """Lightweight readiness when full subsystem metadata is not yet computed."""
    st = compute_engine_readiness(
        frame_count=frame_count,
        baseline_window=baseline_window,
        recent_window=recent_window,
        transition_pressure_history_len=transition_pressure_history_len,
        warmup_margin_frames=warmup_margin_frames,
        min_transition_history=min_transition_history,
    )
    return st.as_dict()
