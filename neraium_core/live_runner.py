from __future__ import annotations

from typing import Any, Dict

from neraium_core.alignment import StructuralEngine


def process_live_frame(engine: StructuralEngine, frame: Dict[str, Any]) -> Dict[str, Any]:
    """Thin helper that forwards a live frame into the structural engine."""
    return engine.process_frame(frame)
