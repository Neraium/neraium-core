from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from neraium_core.features.directional_features import directional_features
from neraium_core.features.multiscale_features import multiscale_features
from neraium_core.features.residual_features import residual_features
from neraium_core.features.temporal_features import temporal_features


@dataclass(frozen=True)
class FeatureBundle:
    raw_features: np.ndarray
    residual_features: np.ndarray
    delta_features: np.ndarray
    slope_features: np.ndarray
    drift_features: np.ndarray
    directional_features: np.ndarray
    temporal_features: np.ndarray
    multi_scale_features: np.ndarray

    def as_groups(self) -> Dict[str, np.ndarray]:
        return {
            "raw": self.raw_features,
            "residual": self.residual_features,
            "delta": self.delta_features,
            "slope": self.slope_features,
            "drift": self.drift_features,
            "directional": self.directional_features,
            "temporal": self.temporal_features,
            "multi_scale": self.multi_scale_features,
        }


def _slope(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 3:
        return np.zeros(matrix.shape[1], dtype=float)
    x = np.arange(matrix.shape[0], dtype=float)
    xc = x - float(np.mean(x))
    denom = float(np.dot(xc, xc)) + 1e-9
    yc = matrix - np.mean(matrix, axis=0, keepdims=True)
    return np.dot(xc, yc) / denom


def build_feature_bundle(matrix: np.ndarray) -> FeatureBundle:
    safe = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if safe.ndim != 2:
        raise ValueError("matrix must be 2D")
    if safe.shape[0] < 2:
        zeros = np.zeros(safe.shape[1], dtype=float)
        return FeatureBundle(zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros)

    raw = np.mean(safe, axis=0)
    residual = residual_features(safe)
    delta = np.mean(np.diff(safe, axis=0), axis=0)
    slope = _slope(safe)
    drift = safe[-1] - safe[0]
    directional = directional_features(safe)
    temporal = temporal_features(safe)
    multi_scale = multiscale_features(safe)

    return FeatureBundle(
        raw_features=np.asarray(raw, dtype=float),
        residual_features=np.asarray(residual, dtype=float),
        delta_features=np.asarray(delta, dtype=float),
        slope_features=np.asarray(slope, dtype=float),
        drift_features=np.asarray(drift, dtype=float),
        directional_features=np.asarray(directional, dtype=float),
        temporal_features=np.asarray(temporal, dtype=float),
        multi_scale_features=np.asarray(multi_scale, dtype=float),
    )


def resolve_feature_mode(bundle: FeatureBundle, mode: str = "combined") -> np.ndarray:
    mode = (mode or "combined").strip().lower()
    groups = bundle.as_groups()
    if mode == "raw_only":
        return groups["raw"]
    if mode == "dynamic_only":
        return np.concatenate([groups["delta"], groups["slope"], groups["drift"], groups["residual"]])
    if mode == "directional_only":
        return groups["directional"]
    if mode == "temporal_only":
        return np.concatenate([groups["temporal"], groups["multi_scale"]])
    return np.concatenate(list(groups.values()))
