from __future__ import annotations

from neraium_core.scoring import composite_instability_score


def test_composite_instability_score() -> None:
    score = composite_instability_score(
        {
            "relational_drift": 2.0,
            "spectral": 1.5,
            "directional_divergence": 1.2,
            "entropy": 0.8,
            "early_warning": 1.0,
            "subsystem_instability": 0.9,
            "forecast": 1.1,
        }
    )
    assert score > 0.0


def test_composite_instability_score_supports_legacy_keys() -> None:
    legacy_score = composite_instability_score(
        {
            "drift": 1.0,
            "spectral": 1.0,
            "directional": 1.0,
            "entropy": 1.0,
            "early_warning": 1.0,
            "subsystem_instability": 1.0,
            "forecast": 1.0,
        }
    )
    assert legacy_score == 1.0
