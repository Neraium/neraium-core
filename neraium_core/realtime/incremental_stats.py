"""Incremental statistics computation using Welford's algorithm.

Provides O(1) memory streaming computation of mean, variance, and correlation
updates without storing the full history.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class WelfordStats:
    """Compute running mean and variance using Welford's algorithm (numerically stable).

    Supports incremental updates without storing all observations, using O(1) memory.
    """

    def __init__(self, n_features: int):
        """Initialize for n_features variables.

        Args:
            n_features: Number of variables to track
        """
        self.n_features = n_features
        self.n = 0  # Number of observations
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)  # For variance
        self.min_val = np.full(n_features, np.inf, dtype=np.float64)
        self.max_val = np.full(n_features, -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update with a single observation.

        Args:
            x: Feature vector of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[0]}")

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_val = np.minimum(self.min_val, x)
        self.max_val = np.maximum(self.max_val, x)

    def get_mean(self) -> np.ndarray:
        """Return current mean vector."""
        return self.mean.copy()

    def get_variance(self) -> np.ndarray:
        """Return current variance vector (unbiased)."""
        if self.n < 2:
            return np.zeros(self.n_features, dtype=np.float64)
        return self.M2 / (self.n - 1)

    def get_stddev(self) -> np.ndarray:
        """Return current standard deviation vector."""
        return np.sqrt(self.get_variance())

    def get_count(self) -> int:
        """Return number of observations processed."""
        return self.n

    def get_range(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_val, max_val) seen so far."""
        return self.min_val.copy(), self.max_val.copy()

    def reset(self) -> None:
        """Reset to initial state."""
        self.n = 0
        self.mean = np.zeros(self.n_features, dtype=np.float64)
        self.M2 = np.zeros(self.n_features, dtype=np.float64)
        self.min_val = np.full(self.n_features, np.inf, dtype=np.float64)
        self.max_val = np.full(self.n_features, -np.inf, dtype=np.float64)


class IncrementalCovarianceTracker:
    """Track covariance matrix updates incrementally.

    Uses update formulas to maintain covariance with O(n²) memory
    instead of storing full history.
    """

    def __init__(self, n_features: int):
        """Initialize for n_features variables.

        Args:
            n_features: Number of variables
        """
        self.n_features = n_features
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        # Store covariance directly (symmetric)
        self.cov = np.zeros((n_features, n_features), dtype=np.float64)
        self._prev_mean = np.zeros(n_features, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update with a single observation using incremental covariance formula.

        Args:
            x: Feature vector of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[0]}")

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n

        # Incremental update to covariance
        if self.n > 1:
            delta2 = x - self.mean
            self.cov += np.outer(delta, delta2)

    def get_covariance(self) -> np.ndarray:
        """Return current covariance matrix (unbiased)."""
        if self.n < 2:
            return np.zeros((self.n_features, self.n_features), dtype=np.float64)
        return self.cov / (self.n - 1)

    def get_correlation(self) -> np.ndarray:
        """Return current correlation matrix from covariance."""
        cov = self.get_covariance()
        std = np.sqrt(np.diag(cov))
        safe_std = np.where(std > 1e-12, std, 1.0)
        corr = cov / np.outer(safe_std, safe_std)
        np.fill_diagonal(corr, 1.0)
        return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def get_mean(self) -> np.ndarray:
        """Return current mean vector."""
        return self.mean.copy()

    def get_count(self) -> int:
        """Return number of observations processed."""
        return self.n

    def reset(self) -> None:
        """Reset to initial state."""
        self.n = 0
        self.mean = np.zeros(self.n_features, dtype=np.float64)
        self.cov = np.zeros((self.n_features, self.n_features), dtype=np.float64)
