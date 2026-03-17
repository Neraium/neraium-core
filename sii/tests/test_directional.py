from __future__ import annotations

import pandas as pd

from sii.directional import directional_metrics, lagged_directional_matrix


def test_directional_metrics() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 6]})
    c = lagged_directional_matrix(df)
    m = directional_metrics(c)
    assert m["causal_energy"] >= 0
    assert m["causal_divergence"] >= m["causal_energy"]
