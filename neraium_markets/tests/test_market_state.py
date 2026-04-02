from __future__ import annotations

import pandas as pd

from neraium.market_state import (
    compare_market_vs_asset_usefulness,
    generate_market_action_posture,
    generate_market_explanation,
    synthesize_market_state,
)


def _asset_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    rows = []
    for asset in ["spy", "qqq", "xlf"]:
        for i, ts in enumerate(idx):
            regime = "stable_trend" if i < 3 else "risk_off_transition"
            action = "lean_long" if i < 3 else "reduce_exposure"
            rows.append(
                {
                    "timestamp": ts,
                    "asset": asset,
                    "regime_label": regime,
                    "action_posture": action,
                    "adjusted_confidence_score": 0.7 if i < 3 else 0.45,
                    "action_useful_5d": 1.0 if i < 3 else -1.0,
                }
            )
    return pd.DataFrame(rows)


def test_market_state_columns_exist():
    market = synthesize_market_state(
        _asset_df(),
        pd.DataFrame(
            {
                "cluster_id": [1, 2],
                "dominant_regime": ["stable_trend", "risk_off_transition"],
                "average_confidence": [0.7, 0.45],
            }
        ),
        pd.DataFrame({"asset": ["spy", "qqq", "xlf"], "influence_score": [0.9, 0.7, 0.6]}),
    )
    assert {"market_regime_label", "market_confidence_score", "market_instability_score", "market_coherence_score"}.issubset(market.columns)


def test_market_action_exists():
    market = pd.DataFrame(
        {
            "market_regime_label": ["market_transition"],
            "market_confidence_score": [0.6],
            "market_instability_score": [0.7],
            "market_coherence_score": [0.4],
        }
    )
    out = generate_market_action_posture(market)
    assert "market_action_posture" in out.columns


def test_market_explanation_exists():
    market = pd.DataFrame(
        {
            "market_regime_label": ["market_transition"],
            "market_confidence_score": [0.6],
            "market_instability_score": [0.7],
            "market_coherence_score": [0.4],
            "market_action_posture": ["wait"],
        }
    )
    out = generate_market_explanation(market)
    assert "market_explanation" in out.columns


def test_market_vs_asset_comparison_not_empty():
    market = pd.DataFrame(
        {
            "market_regime_label": ["market_transition"],
            "market_confidence_score": [0.6],
            "market_instability_score": [0.7],
            "market_coherence_score": [0.4],
            "market_action_posture": ["wait"],
        }
    )
    cmp_df = compare_market_vs_asset_usefulness(_asset_df(), market)
    assert not cmp_df.empty
    assert set(cmp_df["level"]) == {"asset_level", "market_level"}


def test_sorted_output_preserved():
    market = synthesize_market_state(
        _asset_df(),
        pd.DataFrame({"cluster_id": [1], "dominant_regime": ["stable_trend"], "average_confidence": [0.7]}),
        pd.DataFrame({"asset": ["spy"], "influence_score": [0.9]}),
    )
    out = generate_market_action_posture(market)
    assert out.index.is_monotonic_increasing


def test_no_duplicate_columns():
    market = synthesize_market_state(
        _asset_df(),
        pd.DataFrame({"cluster_id": [1], "dominant_regime": ["stable_trend"], "average_confidence": [0.7]}),
        pd.DataFrame({"asset": ["spy"], "influence_score": [0.9]}),
    )
    out = generate_market_explanation(generate_market_action_posture(market))
    assert out.columns.duplicated().sum() == 0
