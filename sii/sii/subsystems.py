from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def subsystem_instability(corr: pd.DataFrame, tau: float = 0.7) -> dict[str, float]:
    adj = (np.abs(corr.values) > tau).astype(int)
    np.fill_diagonal(adj, 0)
    g = nx.from_numpy_array(adj)
    components = [list(c) for c in nx.connected_components(g)]
    if not components:
        return {"subsystem_count": 0, "max_subsystem_instability": 0.0}
    max_instability = 0.0
    for comp in components:
        sub = corr.values[np.ix_(comp, comp)]
        vals = np.linalg.eigvalsh(sub)
        instability = float(np.max(np.abs(vals)))
        max_instability = max(max_instability, instability)
    return {"subsystem_count": len(components), "max_subsystem_instability": max_instability}
