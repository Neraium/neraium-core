from __future__ import annotations

from neraium_core.scoring import composite_instability_score


def test_composite_instability_score() -> None:
    score = composite_instability_score(
        {
            "drift": 2.0,
            "spectral": 1.5,
            "directional": 1.2,
            "entropy": 0.8,
            "early_warning": 1.0,
            "subsystem_instability": 0.9,
        }
    )
    assert score > 0.0
