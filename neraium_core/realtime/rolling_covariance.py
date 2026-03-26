"""Sliding-window matrices and covariance (matches batch np.cov on the same rows)."""

from __future__ import annotations

from collections import deque

import numpy as np

from neraium_core.realtime.numerics import sanitize_finite


class SlidingCovarianceWindow:
    """
    Fixed-length deque of observation vectors. Exposes the current window as a matrix
    and covariance consistent with numpy for that window (no full-history scans).
    """

    def __init__(self, maxlen: int, dim: int) -> None:
        self.maxlen = max(2, int(maxlen))
        self.dim = int(dim)
        self._dq: deque[np.ndarray] = deque(maxlen=self.maxlen)

    def __len__(self) -> int:
        return len(self._dq)

    def clear(self) -> None:
        self._dq.clear()

    def append(self, vec: np.ndarray) -> None:
        v = sanitize_finite(np.asarray(vec, dtype=np.float64).ravel())
        if v.size != self.dim:
            raise ValueError("vector dim mismatch")
        self._dq.append(v.copy())

    def as_matrix(self) -> np.ndarray | None:
        if len(self._dq) == 0:
            return None
        return np.stack(list(self._dq), axis=0)

    def covariance(self) -> np.ndarray:
        m = self.as_matrix()
        if m is None:
            return np.zeros((self.dim, self.dim), dtype=np.float64)
        if m.shape[0] < 2:
            return np.zeros((self.dim, self.dim), dtype=np.float64)
        c = np.cov(m.T.astype(np.float64))
        c = np.nan_to_num(np.asarray(c, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if c.ndim == 0:
            c = np.asarray([[float(c)]], dtype=np.float64)
        c = 0.5 * (c + c.T)
        return c
