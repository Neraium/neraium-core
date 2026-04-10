from __future__ import annotations

from typing import Any

from ui.core_integration import SystemState
from ui.layouts.responsive import build_navigation_surface


def build_operations_view(
    state: SystemState,
    *,
    width_px: int = 1440,
    reasoning_context: dict[str, Any] | None = None,
    operator_question: str = "What is the current system state?",
) -> dict[str, object]:
    return build_navigation_surface(
        state,
        mode="operations",
        width_px=width_px,
        reasoning_context=reasoning_context,
        operator_question=operator_question,
    )
