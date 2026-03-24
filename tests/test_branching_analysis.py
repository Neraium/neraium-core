from __future__ import annotations

from neraium_core.branching import derive_branching_analysis


def _trajectory(stabilizing: float, metastable: float, diverging: float) -> dict[str, object]:
    return {
        "dominant_path": "DIVERGING",
        "path_scores": {
            "stabilizing": stabilizing,
            "metastable": metastable,
            "diverging": diverging,
        },
    }


def test_strongly_dominant_diverging_path_is_high_commitment_not_branching() -> None:
    out = derive_branching_analysis(_trajectory(stabilizing=0.1, metastable=0.2, diverging=0.9))
    assert out is not None
    assert out["commitment"] == "HIGH"
    assert out["is_branching"] is False
    assert out["secondary_path"] == "METASTABLE"


def test_close_diverging_and_metastable_scores_are_branching() -> None:
    out = derive_branching_analysis(_trajectory(stabilizing=0.12, metastable=0.44, diverging=0.46))
    assert out is not None
    assert out["is_branching"] is True
    assert out["commitment"] == "LOW"
    assert out["secondary_path"] == "METASTABLE"


def test_balanced_scores_are_low_commitment() -> None:
    out = derive_branching_analysis(_trajectory(stabilizing=1.0, metastable=1.0, diverging=1.0))
    assert out is not None
    assert out["commitment"] == "LOW"
    assert out["path_entropy"] >= 0.99


def test_strongly_dominant_stabilizing_path_is_high_commitment_not_branching() -> None:
    out = derive_branching_analysis(_trajectory(stabilizing=0.95, metastable=0.2, diverging=0.1))
    assert out is not None
    assert out["commitment"] == "HIGH"
    assert out["is_branching"] is False
    assert out["secondary_path"] == "METASTABLE"


def test_missing_trajectory_analysis_returns_none() -> None:
    assert derive_branching_analysis(None) is None
    assert derive_branching_analysis({"dominant_path": "DIVERGING"}) is None
