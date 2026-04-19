from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from neraium_core.features.feature_pipeline import build_feature_bundle


def _sanitize_vector(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class StateSpaceFrame:
    mode: str
    state_vector: np.ndarray
    feature_groups: dict[str, np.ndarray]


class StateSpaceProjector:
    """Deterministic state-space frame builder from multivariate temporal windows."""

    MODES = {"raw", "residualized", "dynamic", "combined"}

    def __init__(self, mode: str = "combined") -> None:
        self.mode = mode if mode in self.MODES else "combined"

    def _mode_vector(self, groups: dict[str, np.ndarray]) -> np.ndarray:
        if self.mode == "raw":
            return groups["raw"]
        if self.mode == "residualized":
            return groups["residual"]
        if self.mode == "dynamic":
            return np.concatenate([groups["delta"], groups["slope"], groups["drift"], groups["directional"]])
        return np.concatenate(
            [
                groups["raw"],
                groups["residual"],
                groups["delta"],
                groups["slope"],
                groups["drift"],
                groups["directional"],
                groups["temporal"],
                groups["multi_scale"],
            ]
        )

    def project_window(self, matrix: np.ndarray) -> StateSpaceFrame:
        safe = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        bundle = build_feature_bundle(safe)
        groups = bundle.as_groups()
        state = _sanitize_vector(self._mode_vector(groups))
        return StateSpaceFrame(
            mode=self.mode,
            state_vector=state,
            feature_groups={k: _sanitize_vector(v) for k, v in groups.items()},
        )
