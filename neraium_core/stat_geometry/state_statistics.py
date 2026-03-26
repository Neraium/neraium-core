from __future__ import annotations

import numpy as np


def _effective_rank(vals: np.ndarray) -> float:
    eig = np.clip(np.asarray(vals, dtype=float), 1e-12, None)
    p = eig / (np.sum(eig) + 1e-12)
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.exp(entropy))


def _regularized_covariance(samples: np.ndarray, epsilon: float) -> np.ndarray:
    if samples.shape[0] <= 1:
        dim = samples.shape[1] if samples.ndim == 2 else 1
        return np.eye(dim, dtype=float) * float(epsilon)
    cov = np.cov(samples.T)
    cov = np.atleast_2d(np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0))
    cov = 0.5 * (cov + cov.T)
    dim = cov.shape[0]
    return cov + (float(epsilon) * np.eye(dim, dtype=float))


def _safe_logdet(cov: np.ndarray, epsilon: float) -> tuple[float, np.ndarray]:
    """Return stable log-determinant and a PSD-regularized covariance."""
    dim = cov.shape[0]
    base = np.eye(dim, dtype=float)
    eps = max(float(epsilon), 1e-12)

    for _ in range(8):
        cov_reg = cov + eps * base
        sign, logdet = np.linalg.slogdet(cov_reg)
        if sign > 0 and np.isfinite(logdet):
            return float(logdet), cov_reg
        eps *= 10.0

    eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
    eigvals = np.clip(np.asarray(eigvals, dtype=float), eps, None)
    cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    logdet = float(np.sum(np.log(eigvals)))
    return logdet, cov_psd


def _empty_state_statistics() -> dict[str, float]:
    return {
        "local_volume": 0.0,
        "local_density": 0.0,
        "covariance_trace": 0.0,
        "principal_direction_strength": 0.0,
        "anisotropy": 0.0,
        "state_contraction_score": 0.0,
        "state_expansion_score": 0.0,
        "geometric_concentration": 1.0,
    }


def compute_state_statistics(path: list[np.ndarray], window: int = 12, min_covariance_samples: int = 20) -> dict[str, float]:
    if not path:
        return _empty_state_statistics()

    required = max(2, int(min_covariance_samples))
    if len(path) < required:
        return _empty_state_statistics()

    tail_window = max(required, max(2, int(window)))
    tail = np.vstack([np.asarray(v, dtype=float) for v in path[-tail_window:]])
    center = np.mean(tail, axis=0)
    centered = tail - center
    epsilon = 1e-6
    cov = _regularized_covariance(centered, epsilon=0.0)
    log_det, cov_reg = _safe_logdet(cov, epsilon=epsilon)
    eigvals, _ = np.linalg.eigh(cov_reg)
    eigvals = np.clip(np.asarray(eigvals, dtype=float), epsilon, None)

    trace = float(np.sum(eigvals))
    principal = float(np.max(eigvals)) if eigvals.size else 0.0
    principal_strength = float(principal / (trace + 1e-12))
    anisotropy = float((np.max(eigvals) - np.min(eigvals)) / (np.max(eigvals) + 1e-12))
    dimension = max(1, int(cov_reg.shape[0]))
    volume = float(np.exp(log_det / dimension))
    volume = max(volume, epsilon)

    dists = np.linalg.norm(centered, axis=1)
    scale = float(np.mean(dists) + 1e-9)
    density = float(tail.shape[0] / (1.0 + volume + scale))

    first = path[-min(len(path), max(4, window * 2)) : -min(len(path), max(2, window))]
    contraction = 0.0
    expansion = 0.0
    if first:
        prev = np.vstack([np.asarray(v, dtype=float) for v in first])
        prev_cov = _regularized_covariance(prev - np.mean(prev, axis=0), epsilon=epsilon)
        prev_trace = float(np.trace(prev_cov))
        delta = trace - prev_trace
        contraction = float(max(0.0, -delta) / (abs(prev_trace) + 1e-9))
        expansion = float(max(0.0, delta) / (abs(prev_trace) + 1e-9))

    concentration = float(1.0 / max(1.0, _effective_rank(eigvals)))

    return {
        "local_volume": round(volume, 9),
        "local_density": round(density, 6),
        "covariance_trace": round(trace, 6),
        "principal_direction_strength": round(principal_strength, 6),
        "anisotropy": round(anisotropy, 6),
        "state_contraction_score": round(contraction, 6),
        "state_expansion_score": round(expansion, 6),
        "geometric_concentration": round(concentration, 6),
    }
