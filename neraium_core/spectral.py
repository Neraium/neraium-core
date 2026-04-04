from __future__ import annotations

from typing import Any

import numpy as np

# Use ARPACK truncated eigensolver when the matrix is larger than this.
# Full numpy.linalg.eigh is O(n³); ARPACK for k eigenpairs is O(n² · k).
_ARPACK_MIN_N = 30

try:
    import scipy.sparse as _sp_sparse
    import scipy.sparse.linalg as _sp_linalg
    _SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    _SCIPY_SPARSE_AVAILABLE = False
    _sp_sparse = None  # type: ignore[assignment]
    _sp_linalg = None  # type: ignore[assignment]


ArrayLike = Any


def _top_k_eigh(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the top-k eigenvalues and eigenvectors (descending algebraic order).

    Uses ARPACK (scipy.sparse.linalg.eigsh) when the matrix is large and
    scipy is available; falls back to full numpy.linalg.eigh otherwise.
    ARPACK is significantly faster when k << n (e.g. k=1 or k=2 vs n=100+).

    k is capped at n so callers can pass k=2 safely for any matrix size.
    """
    n = matrix.shape[0]
    k = min(k, n)  # can't request more eigenpairs than the matrix dimension
    if _SCIPY_SPARSE_AVAILABLE and n > _ARPACK_MIN_N and k < n - 1:
        try:
            sparse_m = _sp_sparse.csr_matrix(matrix)
            evals, evecs = _sp_linalg.eigsh(sparse_m, k=k, which="LM")
            # eigsh returns in ascending order — reverse to match eigh convention
            order = np.argsort(evals)[::-1]
            return evals[order], evecs[:, order]
        except Exception:
            pass  # fall through to full decomposition
    evals, evecs = np.linalg.eigh(matrix)
    order = np.argsort(evals)[::-1]
    return evals[order][:k], evecs[:, order][:, :k]


def eigendecomposition(matrix: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Matrix must be square")
    if values.size == 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)
    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(safe_values)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def spectral_radius(matrix: ArrayLike) -> float:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
        return 0.0
    safe = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    # Spectral radius = max |eigenvalue| across the full spectrum.
    # ARPACK's "LM" finds algebraically largest, not largest-magnitude, so it can
    # miss a large-magnitude negative eigenvalue (e.g. diag(-2, 1) → reports 1, not 2).
    # eigvalsh computes all eigenvalues without eigenvectors and is fast enough here.
    evals = np.linalg.eigvalsh(safe)
    return float(np.max(np.abs(evals))) if evals.size else 0.0


def spectral_gap(matrix: ArrayLike) -> float:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
        return 0.0
    n = values.shape[0]
    if n < 2:
        return 0.0
    safe = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    # Request k=2; _top_k_eigh caps at n so this is safe for 2×2 matrices too.
    evals, _ = _top_k_eigh(safe, k=2)
    if evals.size < 2:
        return 0.0
    return float(evals[0] - evals[1])


def dominant_mode_loading(matrix: ArrayLike) -> dict[str, list[float] | float]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
        return {"dominant_eigenvalue": 0.0, "dominant_eigenvector": []}
    safe = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    evals, evecs = _top_k_eigh(safe, k=1)
    if evals.size == 0:
        return {"dominant_eigenvalue": 0.0, "dominant_eigenvector": []}
    return {
        "dominant_eigenvalue": float(evals[0]),
        "dominant_eigenvector": [float(v) for v in evecs[:, 0]],
    }
