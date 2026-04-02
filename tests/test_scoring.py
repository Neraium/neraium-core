<<<<<<< HEAD
from neraium_core.scoring import ScoringEngine


def test_score_empty():
    engine = ScoringEngine()

    result = engine.score([])

    assert result["status"] == "normal"


def test_score_normal():
    engine = ScoringEngine()

    result = engine.score([20, 40], {"cpu_usage": 20, "memory_usage": 40})

    assert "score" in result


def test_score_anomaly():
    engine = ScoringEngine()

    # build baseline history
    for _ in range(25):
        engine.score([20, 40], {"cpu_usage": 20, "memory_usage": 40})

    result = engine.score([95, 99], {"cpu_usage": 95, "memory_usage": 99})

    assert "status" in result
=======
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
>>>>>>> b5a6787d331053eaea92f461f7bbab489f4c495a
