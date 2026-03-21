from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_window(window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(window, axis=0)
    std = np.nanstd(window, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    z = (window - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z, mean, std


def correlation_matrix(z_window: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(z_window.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def flatten_upper(corr: np.ndarray) -> np.ndarray:
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        return np.array([], dtype=float)
    idx = np.triu_indices(corr.shape[0], k=1)
    return corr[idx]


def geometric_distance(current: np.ndarray, reference: np.ndarray) -> float:
    if current.shape != reference.shape:
        return 0.0
    return float(np.linalg.norm(current - reference, ord="fro"))


def relational_instability(current_corr: np.ndarray, reference_corr: np.ndarray) -> float:
    cur = flatten_upper(current_corr)
    ref = flatten_upper(reference_corr)
    if cur.shape != ref.shape or cur.size == 0:
        return 0.0
    return float(np.mean(np.abs(cur - ref)))


@dataclass(frozen=True)
class GeometryState:
    mean: np.ndarray
    cov: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class GeometryFeatures:
    baseline_mean: np.ndarray
    baseline_std: np.ndarray
    recent_mean: np.ndarray
    recent_std: np.ndarray
    baseline_corr: np.ndarray
    recent_corr: np.ndarray
    structural_drift: float
    relational_instability: float


def build_geometry_state(
    baseline_window: np.ndarray,
    recent_window: np.ndarray,
    reference_corr: np.ndarray | None = None,
) -> GeometryFeatures:
    z_base, mean_base, std_base = normalize_window(baseline_window)
    z_recent, mean_recent, std_recent = normalize_window(recent_window)

    baseline_corr = correlation_matrix(z_base)
    recent_corr = correlation_matrix(z_recent)
    effective_ref = reference_corr if reference_corr is not None and reference_corr.shape == recent_corr.shape else baseline_corr

    structural_drift = geometric_distance(recent_corr, effective_ref)
    rel_instability = relational_instability(recent_corr, effective_ref)

    return GeometryFeatures(
        baseline_mean=mean_base,
        baseline_std=std_base,
        recent_mean=mean_recent,
        recent_std=std_recent,
        baseline_corr=baseline_corr,
        recent_corr=recent_corr,
        structural_drift=structural_drift,
        relational_instability=rel_instability,
    )


@dataclass(frozen=True)
class GeometrySnapshot:
    mean: np.ndarray
    cov: np.ndarray
    principal_values: np.ndarray
    principal_vectors: np.ndarray


def _safe_cov(cov: np.ndarray) -> np.ndarray:
    c = np.asarray(cov, dtype=float)
    if c.ndim == 0:
        c = np.asarray([[float(c)]], dtype=float)
    if c.ndim == 1:
        c = np.diag(c)
    if c.shape[0] != c.shape[1]:
        n = int(min(c.shape[0], c.shape[1]))
        c = c[:n, :n]
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = 0.5 * (c + c.T)
    return c


def build_geometry_snapshot(state: GeometryState) -> GeometrySnapshot:
    cov = _safe_cov(state.cov)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, a_min=1e-12, a_max=None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return GeometrySnapshot(
        mean=np.asarray(state.mean, dtype=float),
        cov=cov,
        principal_values=vals,
        principal_vectors=vecs,
    )


def geometry_departure_score(current: GeometrySnapshot, baseline: GeometrySnapshot) -> float:
    if current.mean.shape != baseline.mean.shape:
        return 0.0
    delta_mean = float(np.linalg.norm(current.mean - baseline.mean))
    delta_cov = float(np.linalg.norm(current.cov - baseline.cov, ord="fro"))
    n = float(max(1, current.mean.size))
    return float((delta_mean / np.sqrt(n)) + 0.5 * (delta_cov / n))
