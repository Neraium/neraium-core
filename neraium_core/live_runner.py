from __future__ import annotations

from typing import Any, Dict


def process_live_frame(engine: Any, frame: Dict[str, Any]) -> Dict[str, Any]:
    """Thin helper that forwards a live frame into the structural engine."""
    return engine.process_frame(frame)
