from neraium_core.scoring import compute_covariance_shift, compute_drift_score, compute_mahalanobis


def test_compute_mahalanobis_non_negative():
    sample = [1.1, 2.2]
    mean = [1.0, 2.0]
    cov = [[1.0, 0.2], [0.2, 1.1]]
    assert compute_mahalanobis(sample, mean, cov) >= 0.0


def test_compute_covariance_shift_non_negative():
    baseline = [[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]]
    recent = [[2.0, 3.0], [2.1, 3.2], [1.9, 2.8]]
    assert compute_covariance_shift(baseline, recent) >= 0.0


def test_compute_drift_score_detects_shift():
    baseline = [[1.0, 1.0], [1.1, 0.9], [0.9, 1.2], [1.0, 0.95]]
    recent = [[3.0, 3.1], [2.8, 3.2], [3.2, 2.9], [3.1, 3.0]]
    sample = recent[-1]

    score = compute_drift_score(sample=sample, baseline=baseline, recent=recent)
    assert score > 0.5
