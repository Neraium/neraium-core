from __future__ import annotations

from neraium_core.scoring import canonicalize_components, composite_instability_score


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


def test_component_canonicalization() -> None:
    canonical = canonicalize_components({" Drift ": 1.2, "SPECTRAL": 0.4, "unknown": 9.0})
    assert canonical["drift"] == 1.2
    assert canonical["spectral"] == 0.4
    assert "unknown" not in canonical
