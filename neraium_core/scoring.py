from __future__ import annotations

import math
from typing import Iterable, Sequence


def _as_matrix(values: Iterable[Iterable[float]]) -> list[list[float]]:
    matrix = [[float(v) for v in row] for row in values]
    if len(matrix) < 2:
        raise ValueError("At least two rows are required for scoring")
    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        raise ValueError("Telemetry rows must all have the same non-zero width")
    return matrix


def _mean_vector(matrix: Sequence[Sequence[float]]) -> list[float]:
    cols = len(matrix[0])
    return [sum(row[i] for row in matrix) / len(matrix) for i in range(cols)]


def _variance_vector(matrix: Sequence[Sequence[float]], mean: Sequence[float]) -> list[float]:
    cols = len(mean)
    n = len(matrix)
    return [max(sum((row[i] - mean[i]) ** 2 for row in matrix) / max(n - 1, 1), 1e-6) for i in range(cols)]


def compute_mahalanobis(sample: Sequence[float], mean: Sequence[float], covariance: Sequence[Sequence[float]]) -> float:
    """Compute Mahalanobis distance from `sample` to baseline distribution."""
    del covariance  # Using diagonal approximation via variance in compute_drift_score.
    delta_sq = [(float(x) - float(mu)) ** 2 for x, mu in zip(sample, mean)]
    return float(math.sqrt(sum(delta_sq)))


def _covariance_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    mean = _mean_vector(matrix)
    cols = len(mean)
    n = len(matrix)
    cov = [[0.0 for _ in range(cols)] for _ in range(cols)]
    for i in range(cols):
        for j in range(cols):
            cov[i][j] = sum((row[i] - mean[i]) * (row[j] - mean[j]) for row in matrix) / max(n - 1, 1)
    return cov


def compute_covariance_shift(baseline: Iterable[Iterable[float]], recent: Iterable[Iterable[float]]) -> float:
    """Return Frobenius-norm covariance shift between baseline and recent matrices."""
    baseline_m = _as_matrix(baseline)
    recent_m = _as_matrix(recent)
    cov_baseline = _covariance_matrix(baseline_m)
    cov_recent = _covariance_matrix(recent_m)

    total = 0.0
    for i in range(len(cov_baseline)):
        for j in range(len(cov_baseline[i])):
            total += (cov_recent[i][j] - cov_baseline[i][j]) ** 2
    return float(math.sqrt(total))


def compute_drift_score(sample: Sequence[float], baseline: Iterable[Iterable[float]], recent: Iterable[Iterable[float]]) -> float:
    """Combine Mahalanobis and covariance drift with a z-score anomaly factor."""
    baseline_m = _as_matrix(baseline)
    recent_m = _as_matrix(recent)
    sample_v = [float(v) for v in sample]

    mean = _mean_vector(baseline_m)
    variances = _variance_vector(baseline_m, mean)

    mahalanobis = math.sqrt(sum(((x - mu) ** 2) / var for x, mu, var in zip(sample_v, mean, variances)))
    cov_shift = compute_covariance_shift(baseline_m, recent_m)

    std = [math.sqrt(v) for v in variances]
    z_scores = [abs((x - mu) / s) if s > 0 else 0.0 for x, mu, s in zip(sample_v, mean, std)]
    z_anomaly = max(z_scores) if z_scores else 0.0

    score = 0.55 * mahalanobis + 0.30 * cov_shift + 0.15 * z_anomaly
    return float(max(0.0, score))
