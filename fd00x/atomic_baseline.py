"""Atomic baseline fitting for FD00x detector v2.

This module learns the healthy reference used by the streaming AtomicMonitor.
The baseline captures micro-time, micro-dynamics, micro-structure, micro-state,
and micro-topology summaries from healthy history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class AtomicBaseline:
    mean: np.ndarray
    cov: np.ndarray
    precision: np.ndarray
    sde_mu: np.ndarray
    sde_sigma: np.ndarray
    dependence: np.ndarray
    directional_proxy: np.ndarray
    latent_centroids: np.ndarray
    latent_occupancy: np.ndarray
    mode_eigenvalues: np.ndarray
    mode_basis: np.ndarray
    event_rate: np.ndarray
    event_interval_cv: np.ndarray


class AtomicBaselineLearner:
    """Learns the healthy baseline required by AtomicMonitor."""

    def __init__(self, latent_states: int, dmd_rank: int, sde_dt: float, event_level_std: float = 1.0) -> None:
        self.latent_states = max(2, int(latent_states))
        self.dmd_rank = max(1, int(dmd_rank))
        self.sde_dt = max(float(sde_dt), 1e-6)
        self.event_level_std = max(float(event_level_std), 0.1)

    def fit(self, healthy: np.ndarray) -> AtomicBaseline:
        if healthy.ndim != 2 or healthy.shape[0] < 5:
            raise ValueError("healthy baseline data must be 2-D with >=5 rows")

        mean = healthy.mean(axis=0)
        cov = np.cov(healthy.T)
        cov = np.atleast_2d(cov)
        cov += 1e-6 * np.eye(cov.shape[0])
        precision = np.linalg.pinv(cov)

        dx = np.diff(healthy, axis=0)
        sde_mu = dx.mean(axis=0) / self.sde_dt
        sde_sigma = np.sqrt(np.maximum(dx.var(axis=0), 1e-12) / self.sde_dt)

        dependence = np.corrcoef(healthy.T)
        dependence = np.nan_to_num(dependence)
        directional_proxy = self._directional_proxy(healthy)

        latent_centroids, latent_occupancy = self._fit_latent_states(healthy)
        mode_eigenvalues, mode_basis = self._fit_modes(healthy)

        event_rate, event_interval_cv = self._event_stats(healthy, mean, healthy.std(axis=0))

        return AtomicBaseline(
            mean=mean,
            cov=cov,
            precision=precision,
            sde_mu=sde_mu,
            sde_sigma=sde_sigma,
            dependence=dependence,
            directional_proxy=directional_proxy,
            latent_centroids=latent_centroids,
            latent_occupancy=latent_occupancy,
            mode_eigenvalues=mode_eigenvalues,
            mode_basis=mode_basis,
            event_rate=event_rate,
            event_interval_cv=event_interval_cv,
        )

    def _directional_proxy(self, data: np.ndarray) -> np.ndarray:
        x = data[:-1]
        y = data[1:]
        n = x.shape[1]
        proxy = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a = x[:, i]
                b = y[:, j]
                c = np.corrcoef(a, b)[0, 1] if a.std() > 1e-12 and b.std() > 1e-12 else 0.0
                proxy[i, j] = float(np.nan_to_num(c))
        return proxy

    def _fit_latent_states(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.linspace(0, data.shape[0] - 1, self.latent_states, dtype=int)
        centroids = data[idx].copy()
        labels = np.zeros(data.shape[0], dtype=int)
        for _ in range(8):
            d2 = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(d2, axis=1)
            for k in range(self.latent_states):
                mask = labels == k
                if np.any(mask):
                    centroids[k] = data[mask].mean(axis=0)
        occupancy = np.bincount(labels, minlength=self.latent_states).astype(float)
        occupancy /= max(float(occupancy.sum()), 1.0)
        return centroids, occupancy

    def _fit_modes(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = data[:-1].T
        y = data[1:].T
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        r = min(self.dmd_rank, len(s))
        u_r = u[:, :r]
        s_r = s[:r]
        vt_r = vt[:r, :]
        s_inv = np.diag(1.0 / np.maximum(s_r, 1e-9))
        a_tilde = u_r.T @ y @ vt_r.T @ s_inv
        eigvals, eigvecs = np.linalg.eig(a_tilde)
        modes = u_r @ eigvecs
        return eigvals, modes

    def _event_stats(self, data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_t, n_s = data.shape
        std = np.where(std < 1e-6, 1.0, std)
        rates = np.zeros(n_s, dtype=float)
        cv = np.ones(n_s, dtype=float)
        for s in range(n_s):
            over = np.abs(data[:, s] - mean[s]) > (self.event_level_std * std[s])
            crossings = np.where(np.diff(over.astype(int)) != 0)[0]
            rates[s] = len(crossings) / max(n_t, 1)
            if len(crossings) > 2:
                gaps = np.diff(crossings)
                cv[s] = float(gaps.std() / max(gaps.mean(), 1e-6))
        return rates, cv
