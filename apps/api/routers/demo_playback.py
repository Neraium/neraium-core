from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ui.app import load_builtin_demo_rows
from ui.config import UIConfig
from ui.core_integration import build_system_state, evaluate_gate
from ui.reasoning import build_reasoning_context


@dataclass
class DemoReplay:
    mode: str
    frames: list[dict[str, Any]]
    initialized_at: float


class DemoReplayStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, DemoReplay] = {}

    def get_or_build(self, *, use_synthetic: bool) -> DemoReplay:
        mode = "synthetic" if use_synthetic else "historical"
        with self._lock:
            existing = self._cache.get(mode)
            if existing is not None:
                return existing
            frames = self._build_frames(use_synthetic=use_synthetic)
            replay = DemoReplay(mode=mode, frames=frames, initialized_at=time.time())
            self._cache[mode] = replay
            return replay

    def reset(self, *, use_synthetic: bool) -> DemoReplay:
        mode = "synthetic" if use_synthetic else "historical"
        with self._lock:
            replay = DemoReplay(mode=mode, frames=self._build_frames(use_synthetic=use_synthetic), initialized_at=time.time())
            self._cache[mode] = replay
            return replay

    @staticmethod
    def _build_frames(*, use_synthetic: bool) -> list[dict[str, Any]]:
        rows = load_builtin_demo_rows(use_synthetic=use_synthetic)
        if not rows:
            return []

        cfg = UIConfig()
        frames: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            history = rows[: idx + 1]
            previous = history[-2] if len(history) > 1 else None
            state = build_system_state(history, config=cfg)
            gate = evaluate_gate(row, previous, state)
            reasoning = build_reasoning_context(state, history, gate_decision=gate)
            evidence = [event for event in reasoning.get("recent_admitted_events", []) if isinstance(event, dict)]

            frames.append(
                {
                    "frame_index": idx,
                    "frame_count": len(rows),
                    "timestamp": row.get("timestamp"),
                    "current_phase": row.get("system_phase") or row.get("regime_name") or row.get("state") or "UNKNOWN",
                    "verdict": str(gate.get("decision") or "SUPPRESS").upper(),
                    "status": str(row.get("system_health") or row.get("risk_level") or "unknown").lower(),
                    "metrics": {
                        "structural_drift": float(row.get("structural_drift_score") or 0.0),
                        "relational_stability": float(row.get("relational_stability_score") or 0.0),
                        "coherence": float(row.get("coherence_score") or 0.0),
                        "confidence": float(row.get("confidence_score") or 0.0),
                    },
                    "tetrahedral_state": row.get("tetrahedral_state") or {},
                    "reasoning": {
                        "summary": row.get("evidence_summary") or row.get("explanation_text") or "",
                        "context": reasoning,
                    },
                    "evidence": evidence,
                    "gate_decision": gate,
                }
            )
        return frames


_store = DemoReplayStore()


def _build_summary(replay: DemoReplay) -> dict[str, Any]:
    frames = replay.frames
    if not frames:
        return {
            "mode": replay.mode,
            "frame_count": 0,
            "phase_counts": {},
            "lifecycle": [],
            "initialized_at": replay.initialized_at,
        }

    phase_counts: dict[str, int] = {}
    for frame in frames:
        phase = str(frame.get("current_phase") or "UNKNOWN")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    lifecycle = []
    for phase, count in phase_counts.items():
        lifecycle.append({"phase": phase, "frames": count})

    return {
        "mode": replay.mode,
        "frame_count": len(frames),
        "phase_counts": phase_counts,
        "lifecycle": lifecycle,
        "initialized_at": replay.initialized_at,
    }


def build_demo_playback_router() -> APIRouter:
    router = APIRouter(prefix="/api/demo", tags=["demo-playback"])

    @router.get("/init")
    def demo_init(use_synthetic: bool = Query(default=True)) -> dict[str, Any]:
        replay = _store.get_or_build(use_synthetic=use_synthetic)
        if not replay.frames:
            raise HTTPException(status_code=503, detail="Demo frames unavailable.")
        return {
            "mode": replay.mode,
            "summary": _build_summary(replay),
            "initial_frame": replay.frames[0],
            "frames": replay.frames,
        }

    @router.get("/frame/{index}")
    def demo_frame(index: int, use_synthetic: bool = Query(default=True)) -> dict[str, Any]:
        replay = _store.get_or_build(use_synthetic=use_synthetic)
        if index < 0 or index >= len(replay.frames):
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {index}")
        return replay.frames[index]

    @router.get("/summary")
    def demo_summary(use_synthetic: bool = Query(default=True)) -> dict[str, Any]:
        replay = _store.get_or_build(use_synthetic=use_synthetic)
        return _build_summary(replay)

    @router.post("/reset")
    def demo_reset(use_synthetic: bool = Query(default=True)) -> dict[str, Any]:
        replay = _store.reset(use_synthetic=use_synthetic)
        return {
            "status": "reset",
            "mode": replay.mode,
            "summary": _build_summary(replay),
            "initial_frame": replay.frames[0] if replay.frames else None,
        }

    @router.get("/stream")
    def demo_stream(
        use_synthetic: bool = Query(default=True),
        start_index: int = Query(default=0, ge=0),
        fps: float = Query(default=8.0, ge=0.5, le=60.0),
    ) -> StreamingResponse:
        replay = _store.get_or_build(use_synthetic=use_synthetic)

        def event_stream() -> Any:
            interval = 1.0 / fps
            for frame in replay.frames[start_index:]:
                yield f"data: {json.dumps(frame)}\n\n"
                time.sleep(interval)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router
