from __future__ import annotations

import pandas as pd

from sii.geometry import structural_drift
from sii.normalization import zscore_window


def test_geometry_drift_nonzero() -> None:
    w0 = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "c": [4, 3, 2, 1]})
    w1 = pd.DataFrame({"a": [1, 2, 3, 5], "b": [1, 2, 3, 6], "c": [4, 3, 2, 1]})
    r0 = zscore_window(w0).corr()
    r1 = zscore_window(w1).corr()
    fro, mad = structural_drift(r0, r1)
    assert fro >= 0
    assert mad >= 0
