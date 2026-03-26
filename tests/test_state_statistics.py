from __future__ import annotations

import numpy as np

from neraium_core.stat_geometry.state_statistics import compute_state_statistics


def test_local_volume_stays_positive_for_nearly_collapsed_state_window() -> None:
    path = [np.array([1.0, 1.0, 1.0]) + (1e-10 * i) for i in range(20)]

    stats = compute_state_statistics(path, window=12)

    assert stats["local_volume"] > 0.0


def test_local_volume_varies_over_time_with_state_changes() -> None:
    base = [np.array([1.0 + 0.001 * i, 0.1 * i, 0.5]) for i in range(20)]
    shifted = base + [np.array([2.0 + 0.2 * i, -1.0 + 0.4 * i, 3.0]) for i in range(10)]

    before = compute_state_statistics(base, window=12)
    after = compute_state_statistics(shifted, window=12)

    assert before["local_volume"] > 0.0
    assert after["local_volume"] > 0.0
    assert before["local_volume"] != after["local_volume"]
