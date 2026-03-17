from __future__ import annotations

from sii.scoring import composite_score


def test_composite_score() -> None:
    score = composite_score({"spectral_radius": 2.0, "entropy": 1.0})
    assert score > 0
