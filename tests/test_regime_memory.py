from __future__ import annotations

import numpy as np

from neraium_core.regime import assign_regime, get_regime_entry, regime_library_stats, update_regime_library


def test_pending_regime_requires_persistence_before_assignment() -> None:
    regimes: list[dict[str, object]] = []
    base = np.array([0.0, 0.0, 0.0], dtype=float)
    update_regime_library(base, regimes, threshold=0.5, min_persistence=3)

    novel = np.array([5.0, 5.0, 5.0], dtype=float)
    update_regime_library(novel, regimes, threshold=0.5, min_persistence=3)
    # Pending regimes are intentionally excluded from normal assignment, so
    # nearest active regime can still be returned without a max-distance gate.
    assert assign_regime(novel, regimes) is not None

    update_regime_library(novel, regimes, threshold=0.5, min_persistence=3)
    assigned_pre = assign_regime(novel, regimes)
    assert assigned_pre is not None
    assert str(assigned_pre["name"]) == "regime_0"

    update_regime_library(novel, regimes, threshold=0.5, min_persistence=3)
    assigned = assign_regime(novel, regimes)
    assert assigned is not None
    assert str(assigned["name"]) == "regime_1"


def test_regime_prototype_count_is_bounded() -> None:
    regimes: list[dict[str, object]] = []
    update_regime_library(np.array([0.0, 0.0]), regimes, threshold=10.0, max_prototypes=2)
    update_regime_library(np.array([1.0, 1.0]), regimes, threshold=10.0, max_prototypes=2)
    update_regime_library(np.array([2.0, 2.0]), regimes, threshold=10.0, max_prototypes=2)

    regime0 = get_regime_entry(regimes, "regime_0")
    assert regime0 is not None
    prototypes = regime0.get("prototypes")
    assert isinstance(prototypes, list)
    assert len(prototypes) == 2


def test_regime_library_stats_reports_active_and_pending() -> None:
    regimes: list[dict[str, object]] = []
    update_regime_library(np.array([0.0, 0.0]), regimes, threshold=0.5, min_persistence=3)
    update_regime_library(np.array([3.0, 3.0]), regimes, threshold=0.5, min_persistence=3)

    stats = regime_library_stats(regimes)
    assert stats["total"] == 2
    assert stats["active"] == 1
    assert stats["pending"] == 1
