"""UI replay endpoint serving frame sequences for the Next.js frontend."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter

from ui.app import load_builtin_demo_rows, create_app_state
from ui.config import UIConfig
from ui.core_integration import build_system_state, evaluate_gate, classify_transition
from ui.layouts.operations_view import build_operations_view

router = APIRouter(prefix="/api/ui", tags=["ui-replay"])


def _build_frame(row: dict[str, Any], previous_row: dict[str, Any] | None, index: int, total: int) -> dict[str, Any]:
    """Convert a data row into a complete UI frame."""
    system_state = build_system_state([row], config=UIConfig())
    gate_decision = evaluate_gate(row, previous_row, system_state)
    transition = classify_transition(previous_row, row)

    # Build the full operations view for this frame
    surface = build_operations_view(
        system_state,
        reasoning_context={},
        gate_decision=gate_decision,
        current_frame=index,
        total_frames=total,
    )

    gate_content = surface["zones"].get("gate", {}).get("content", {})
    system_content = surface["zones"].get("system_state", {}).get("content", {})
    reasoning_content = surface["zones"].get("reasoning", {}).get("content", {})

    return {
        "index": index,
        "total": total,
        "timestamp": row.get("timestamp", ""),
        "phase": row.get("regime_name", "baseline"),
        "system_health": row.get("system_health", "nominal"),
        "confidence": float(row.get("confidence_score", 0.0)),
        "structural_drift_score": float(row.get("structural_drift_score", 0.0)),
        "relational_stability_score": float(row.get("relational_stability_score", 0.0)),
        "coherence_score": float(row.get("coherence_score", 0.0)),
        "event_admitted": row.get("event_admitted", False),
        "transition_type": row.get("transition_type", "STABLE"),
        "persistence_minutes": row.get("persistence_minutes", 0),
        "gate_decision": gate_decision,
        "transition": transition,
        "system_geometry": system_content.get("nodes", []) if isinstance(system_content, dict) else [],
        "system_edges": system_content.get("edges", []) if isinstance(system_content, dict) else [],
        "reasoning_facts": reasoning_content.get("observed_facts", []) if isinstance(reasoning_content, dict) else [],
        "gates_zones": surface.get("zones", {}),
    }


@router.get("/frames")
async def get_frames(limit: int = 180, use_synthetic: bool = True) -> dict[str, Any]:
    """
    Get a complete frame sequence for replay.

    Returns all frames needed for the UI to play through the entire demo.
    Each frame contains all the data needed to render the UI state at that point.
    """
    rows = load_builtin_demo_rows(use_synthetic=use_synthetic)

    frames = []
    for idx, row in enumerate(rows):
        previous_row = rows[idx - 1] if idx > 0 else None
        frame = _build_frame(row, previous_row, idx + 1, len(rows))
        frames.append(frame)

    return {
        "frames": frames,
        "total_frames": len(rows),
        "demo_mode": "synthetic" if use_synthetic else "replay",
    }


@router.get("/frame/{index}")
async def get_frame(index: int, use_synthetic: bool = True) -> dict[str, Any]:
    """Get a single frame by index."""
    rows = load_builtin_demo_rows(use_synthetic=use_synthetic)

    if index < 0 or index >= len(rows):
        return {"error": f"Frame index {index} out of range [0, {len(rows) - 1})"}

    row = rows[index]
    previous_row = rows[index - 1] if index > 0 else None
    frame = _build_frame(row, previous_row, index + 1, len(rows))

    return frame
