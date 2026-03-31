from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LatentStateSnapshot:
    embedding: list[float]
    summary_features: dict[str, float]
    trajectory: list[list[float]]
    velocity: float
    acceleration: float


class LatentStructuralStateEncoder:
    """Interpretable streaming latent state via normalized feature projection + SVD basis."""

    def __init__(self, latent_dim: int = 3, history: int = 240) -> None:
        self.latent_dim = max(2, int(latent_dim))
        self._feature_history: deque[np.ndarray] = deque(maxlen=max(40, int(history)))
        self._latent_history: deque[np.ndarray] = deque(maxlen=max(40, int(history)))
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _flatten_features(self, observation: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
        metrics = {
            "structural_drift": float(observation.get("structural_drift_score", 0.0)),
            "relational_instability": float(observation.get("relational_instability_score", 0.0)),
            "regime_distance": float(observation.get("regime_distance", 0.0)),
            "graph_deformation": float(observation.get("graph_deformation_score", 0.0)),
            "coherence_loss": float(max(0.0, 1.0 - float(observation.get("coherence_score", 0.0)))),
            "coupling_instability": float(observation.get("coupling_instability_score", 0.0)),
            "risk_pressure": float(observation.get("risk_pressure", 0.0)),
            "path_shift": float(observation.get("path_length_shift", 0.0)),
            "subspace_rotation": float(observation.get("subspace_rotation", 0.0)),
            "mean_shift": float(observation.get("mean_shift", 0.0)),
            "covariance_shift": float(observation.get("covariance_shift", 0.0)),
        }
        feature_names = list(metrics.keys())
        vec = np.asarray([metrics[k] for k in feature_names], dtype=float)
        return vec, feature_names

    def encode(self, observation: dict[str, Any]) -> LatentStateSnapshot:
        feature_vec, names = self._flatten_features(observation)
        self._feature_names = names
        self._feature_history.append(feature_vec)

        X = np.vstack(self._feature_history)
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        Xn = (X - mu) / sigma

        if Xn.shape[0] < 3:
            basis = np.eye(Xn.shape[1], self.latent_dim, dtype=float)
        else:
            _, _, vh = np.linalg.svd(Xn, full_matrices=False)
            basis = vh[: self.latent_dim].T
        latent = np.dot(Xn[-1], basis)
        self._latent_history.append(latent)

        traj = [x.astype(float).tolist() for x in list(self._latent_history)[-12:]]
        velocity = 0.0
        acceleration = 0.0
        if len(self._latent_history) >= 2:
            velocity = float(np.linalg.norm(self._latent_history[-1] - self._latent_history[-2]))
        if len(self._latent_history) >= 3:
            prev_v = float(np.linalg.norm(self._latent_history[-2] - self._latent_history[-3]))
            acceleration = float(velocity - prev_v)

        loading_energy = np.abs(basis[:, 0]) if basis.size else np.zeros((len(names),), dtype=float)
        top_idx = np.argsort(loading_energy)[::-1][:4]
        summary = {names[i]: float(feature_vec[i]) for i in top_idx}

        return LatentStateSnapshot(
            embedding=latent.astype(float).tolist(),
            summary_features=summary,
            trajectory=traj,
            velocity=velocity,
            acceleration=acceleration,
        )
