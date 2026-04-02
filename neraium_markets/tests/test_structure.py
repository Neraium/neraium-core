from __future__ import annotations

import pandas as pd

from config import DATE_COLUMN, LEAD_LAG_PAIRS, SECTOR_ASSETS, STRUCTURE_ASSETS
from neraium.alignment import align_close_series
from neraium.data_loader import load_all_assets
from neraium.features import build_feature_table
from neraium.structure import (
    build_structural_snapshot,
    compute_coherence_score,
    compute_concentration_entropy,
    compute_correlation_matrices,
    compute_instability_score,
    compute_lead_lag_drift,
)


def _snapshot() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_all_assets()
    aligned = align_close_series(raw)
    features = build_feature_table(aligned)
    snapshot = build_structural_snapshot(features)
    return aligned, features, snapshot


def test_corr_drift_score_exists():
    _, features, _ = _snapshot()
    corr = compute_correlation_matrices(features, assets=STRUCTURE_ASSETS)
    assert "corr_drift_score" in corr.columns


def test_lag_drift_score_exists():
    _, features, _ = _snapshot()
    lag = compute_lead_lag_drift(features, asset_pairs=LEAD_LAG_PAIRS)
    assert "lag_drift_score" in lag.columns


def test_entropy_and_concentration_exist():
    _, features, _ = _snapshot()
    concentration = compute_concentration_entropy(features, sector_assets=SECTOR_ASSETS)
    assert "sector_entropy" in concentration.columns
    assert "sector_concentration_score" in concentration.columns


def test_instability_score_exists():
    _, _, snapshot = _snapshot()
    instability = compute_instability_score(snapshot)
    assert "instability_score" in instability.columns


def test_coherence_score_exists():
    _, _, snapshot = _snapshot()
    coherence = compute_coherence_score(snapshot)
    assert "coherence_score" in coherence.columns


def test_structural_snapshot_sorted():
    _, _, snapshot = _snapshot()
    assert snapshot[DATE_COLUMN].is_monotonic_increasing


def test_timestamp_preserved_in_snapshot():
    aligned, _, snapshot = _snapshot()
    assert DATE_COLUMN in snapshot.columns
    assert aligned[DATE_COLUMN].equals(snapshot[DATE_COLUMN])


def test_no_duplicate_columns_in_snapshot():
    _, _, snapshot = _snapshot()
    assert snapshot.columns.is_unique


def test_scores_are_numeric():
    _, _, snapshot = _snapshot()
    for col in [
        "corr_drift_score",
        "lag_drift_score",
        "sector_entropy",
        "sector_concentration_score",
        "instability_score",
        "coherence_score",
    ]:
        assert col in snapshot.columns
        assert pd.api.types.is_numeric_dtype(snapshot[col])
