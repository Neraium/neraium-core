from __future__ import annotations

import pandas as pd

from neraium.propagation import (
    build_regime_propagation_table,
    compute_asset_influence_scores,
    compute_sector_influence_scores,
)


def _df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    seq = {
        "spy": ["stable_trend", "stable_trend", "risk_off_transition", "risk_off_transition", "high_volatility", "high_volatility", "mean_reversion", "mean_reversion", "stable_trend", "stable_trend"],
        "qqq": ["stable_trend", "stable_trend", "stable_trend", "risk_off_transition", "risk_off_transition", "high_volatility", "high_volatility", "mean_reversion", "mean_reversion", "stable_trend"],
        "xlf": ["stable_trend", "stable_trend", "stable_trend", "stable_trend", "risk_off_transition", "risk_off_transition", "high_volatility", "mean_reversion", "mean_reversion", "stable_trend"],
    }
    for asset, regimes in seq.items():
        for ts, regime in zip(idx, regimes):
            rows.append({"timestamp": ts, "asset": asset, "regime_label": regime})
    return pd.DataFrame(rows)


def test_propagation_table_not_empty():
    p = build_regime_propagation_table(_df(), ["spy", "qqq", "xlf"])
    assert not p.empty
    assert {"source_asset", "target_asset", "propagation_count", "avg_delay_steps"}.issubset(p.columns)


def test_asset_influence_scores_exist():
    p = build_regime_propagation_table(_df(), ["spy", "qqq", "xlf"])
    s = compute_asset_influence_scores(p)
    assert not s.empty
    assert {"asset", "influence_score", "outgoing_propagation_count", "avg_lead_time"}.issubset(s.columns)


def test_sector_influence_scores_exist():
    p = build_regime_propagation_table(_df(), ["spy", "qqq", "xlf"])
    out = compute_sector_influence_scores(p, {"spy": "broad", "qqq": "broad", "xlf": "financials"})
    assert not out.empty
    assert {"sector", "influence_score", "propagation_count"}.issubset(out.columns)
