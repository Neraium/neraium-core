"""
Optimized configuration presets for SII-ML.

The presets are intentionally simple dictionaries so downstream users can
serialize/override them easily.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


_PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "thresholds": {
            "healthy": 0.30,
            "elevated": 0.40,
            "caution": 0.56,
            "critical": 0.76,
        },
        "persistence": {
            "evidence_decay": 0.92,
            "critical_evidence": 6.0,
            "elevated_min_cycles": 4,
            "caution_min_cycles": 7,
        },
        "ml": {
            "attention_heads": 4,
            "attention_head_dim": 16,
            "graph_hidden": 32,
            "neural_boost": 0.20,
            "online_lr": 0.0004,
        },
        "execution": {
            "l2_gate": 0.24,
            "l3_gate": 0.38,
            "l4_stride": 40,
            "l5_stride": 150,
        },
    },
    "balanced": {
        "thresholds": {
            "healthy": 0.28,
            "elevated": 0.38,
            "caution": 0.52,
            "critical": 0.72,
        },
        "persistence": {
            "evidence_decay": 0.90,
            "critical_evidence": 5.5,
            "elevated_min_cycles": 3,
            "caution_min_cycles": 6,
        },
        "ml": {
            "attention_heads": 4,
            "attention_head_dim": 16,
            "graph_hidden": 32,
            "neural_boost": 0.25,
            "online_lr": 0.0005,
        },
        "execution": {
            "l2_gate": 0.22,
            "l3_gate": 0.35,
            "l4_stride": 40,
            "l5_stride": 150,
        },
    },
    "aggressive": {
        "thresholds": {
            "healthy": 0.25,
            "elevated": 0.34,
            "caution": 0.48,
            "critical": 0.68,
        },
        "persistence": {
            "evidence_decay": 0.88,
            "critical_evidence": 5.0,
            "elevated_min_cycles": 2,
            "caution_min_cycles": 5,
        },
        "ml": {
            "attention_heads": 4,
            "attention_head_dim": 16,
            "graph_hidden": 32,
            "neural_boost": 0.30,
            "online_lr": 0.0007,
        },
        "execution": {
            "l2_gate": 0.20,
            "l3_gate": 0.32,
            "l4_stride": 30,
            "l5_stride": 120,
        },
    },
}


def get_optimal_config(preset: str = "balanced") -> Dict[str, Any]:
    """Return an isolated copy of one of the built-in SII-ML presets."""

    key = preset.lower().strip()
    if key not in _PRESETS:
        valid = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Unknown preset '{preset}'. Valid presets: {valid}.")
    return deepcopy(_PRESETS[key])


__all__ = ["get_optimal_config"]
