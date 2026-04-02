"""Interpretive gate that suppresses noisy signals."""

from __future__ import annotations


def should_emit_signal(confidence: float, drift: float, instability: float, min_confidence: float = 0.45) -> bool:
    if confidence < min_confidence:
        return False
    if drift < 0.25 and instability < 0.25:
        return False
    return True
