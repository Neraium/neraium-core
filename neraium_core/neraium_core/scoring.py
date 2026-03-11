import numpy as np


def mahalanobis_distance(x, mean, cov):
    x = np.array(x)
    mean = np.array(mean)
    cov = np.array(cov)

    diff = x - mean
    inv_cov = np.linalg.inv(cov)

    return float(np.sqrt(diff.T @ inv_cov @ diff))