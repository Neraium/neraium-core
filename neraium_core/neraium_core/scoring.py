import numpy as np


def mahalanobis_distance(vector, mean_vector, covariance_matrix) -> float:
    x = np.array(vector, dtype=float)
    mu = np.array(mean_vector, dtype=float)
    cov = np.array(covariance_matrix, dtype=float)

    if x.shape[0] != mu.shape[0]:
        raise ValueError("vector and mean_vector dimension mismatch")

    if cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance_matrix must be square")

    if cov.shape[0] != x.shape[0]:
        raise ValueError("covariance_matrix dimension mismatch")

    delta = x - mu
    inv_cov = np.linalg.inv(cov)
    return float(np.sqrt(delta.T @ inv_cov @ delta))