"""Relational Instability & Trajectory Dynamics layers of the Intelligence Stack.

Detects:
- Relational Instability: breakdown of sensor correlations, covariance eigenvalue instability, dependency fracture
- Trajectory Dynamics: acceleration spikes with confidence, persistent curvature, change-point events

These components are part of the Intelligence Stack's evidence fusion layer.

Frontier improvements:
1. Acceleration confidence: sustained positive second derivative over rolling window
2. Rolling persistence measure: distinguish persistent curvature from single-cycle spikes
3. Covariance eigenvalue instability: detect mode activation changes
4. Correlation matrix drift confidence: relational instability score
5. Dependency fracture: detect sudden loss of multivariate structure
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class StructuralSignalDetector:
    """Detects genuine structural changes in system dynamics.

    Implements two Intelligence Stack layers:
    - Layer 2 (Relational Instability): correlation breakdown, eigenvalue instability, dependency fracture
    - Layer 3 (Trajectory Dynamics): acceleration confidence, persistent curvature, change-point detection

    Frontier improvements focus on early, high-confidence detection of structural changes.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def compute_acceleration_confidence(self, raw: np.ndarray, baseline_std: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Layer 3 (Trajectory Dynamics): acceleration confidence with persistence.

        Instead of just detecting acceleration spikes, measure:
        1. Sustained positive second derivative (persistent curvature)
        2. Confidence score based on rolling window persistence
        3. Distinguish genuine acceleration from random noise

        Returns:
            (accel_confidence, persistence_scores): per-cycle metrics
        """
        n_cycles = len(raw)
        accel_confidence = np.zeros(n_cycles)
        persistence_scores = np.zeros(n_cycles)

        if n_cycles < 3:
            return accel_confidence, persistence_scores

        # Compute first and second derivatives
        vel = np.diff(raw)
        accel = np.diff(vel)
        baseline_accel_std = np.std(accel[:min(20, len(accel))])
        baseline_accel_std = max(baseline_accel_std, 1e-6)

        # Threshold for significant acceleration
        accel_threshold = 0.003 * baseline_std

        # For each cycle in the acceleration series, measure persistence of positive acceleration
        window = 5
        for t in range(len(accel)):
            # Look back at acceleration window
            window_start = max(0, t - window)
            window_end = min(t + 1, len(accel))
            if window_start >= window_end:
                continue

            window_accel = accel[window_start:window_end]

            # Count sustained positive accelerations
            positive_count = np.sum(window_accel > accel_threshold)
            window_len = len(window_accel)

            if window_len > 0:
                persistence = positive_count / window_len

                # Update persistence scores (map to correct index in accel space)
                if t + 2 < n_cycles:
                    persistence_scores[t + 2] = min(1.0, persistence)

                # Confidence: high if sustained, low if sporadic
                if persistence > 0.5:  # More than half the window shows acceleration
                    current_accel = abs(accel[t])
                    if t + 2 < n_cycles:
                        accel_confidence[t + 2] = min(1.0, current_accel / (accel_threshold + 1e-6) * persistence)

        return accel_confidence, persistence_scores

    def compute_relational_instability_confidence(
        self, data: np.ndarray, ref_corr: np.ndarray, ref_cov: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute Layer 2 (Relational Instability): advanced multi-signal confidence.

        Combines three independent relational signals:
        1. Correlation breakdown: loss of pairwise sensor relationships
        2. Covariance eigenvalue instability: mode activation changes
        3. Dependency fracture: sudden loss of multivariate structure

        Returns dict with per-cycle metrics:
            - correlation_breakdown_confidence
            - eigenvalue_instability_score
            - dependency_fracture_score
            - relational_instability_confidence (combined)
        """
        arr = np.asarray(data, dtype=float)
        n_cycles = arr.shape[0]

        result = {
            "correlation_breakdown": np.zeros(n_cycles),
            "eigenvalue_instability": np.zeros(n_cycles),
            "dependency_fracture": np.zeros(n_cycles),
            "relational_instability": np.zeros(n_cycles),
        }

        if n_cycles < 10:
            return result

        # Signal 1: Correlation breakdown (refined)
        window_size = 15
        for t in range(window_size, n_cycles):
            window_data = arr[t - window_size : t]
            if window_data.shape[0] < 2:
                continue

            recent_corr = np.corrcoef(window_data.T)
            recent_corr = np.nan_to_num(recent_corr, nan=0.0)

            # Compare to reference (healthy) correlation
            corr_diff = recent_corr - ref_corr
            corr_change = np.linalg.norm(corr_diff, "fro")

            # Normalize by reference correlation scale
            ref_scale = np.linalg.norm(ref_corr, "fro") + 1e-6
            normalized_change = corr_change / ref_scale
            result["correlation_breakdown"][t] = min(1.0, normalized_change * 2.0)

        # Signal 2: Eigenvalue instability (mode activation)
        for t in range(window_size, n_cycles):
            window_data = arr[t - window_size : t]
            if window_data.shape[0] < 2:
                continue

            recent_cov = np.cov(window_data.T) + np.eye(window_data.shape[1]) * 1e-6

            # Compare eigenvalues to reference
            recent_eigs = np.linalg.eigvalsh(recent_cov)
            ref_eigs = np.linalg.eigvalsh(ref_cov)

            # Sort in descending order
            recent_eigs = np.sort(recent_eigs)[::-1]
            ref_eigs = np.sort(ref_eigs)[::-1]

            # Measure eigenvalue drift (relative)
            if len(recent_eigs) == len(ref_eigs):
                eig_diff = np.abs(recent_eigs - ref_eigs) / (np.abs(ref_eigs) + 1e-6)
                eigenvalue_drift = np.mean(eig_diff)
                result["eigenvalue_instability"][t] = min(1.0, eigenvalue_drift)

        # Signal 3: Dependency fracture (multivariate structure loss)
        for t in range(window_size, n_cycles):
            window_data = arr[t - window_size : t]
            if window_data.shape[0] < 2:
                continue

            # Measure condition number as proxy for multivariate structure health
            recent_cov = np.cov(window_data.T) + np.eye(window_data.shape[1]) * 1e-6

            try:
                cond_num = np.linalg.cond(recent_cov)
                ref_cov_stable = ref_cov + np.eye(ref_cov.shape[0]) * 1e-6
                ref_cond_num = np.linalg.cond(ref_cov_stable)

                # Detect sudden increase in condition number (structure breakdown)
                cond_ratio = min(cond_num / (ref_cond_num + 1e-6), 10.0)
                # Normalize: ratio of 2-3 is moderate, >5 is severe
                fracture_score = (cond_ratio - 1.0) / 4.0
                result["dependency_fracture"][t] = np.clip(fracture_score, 0.0, 1.0)
            except (np.linalg.LinAlgError, ValueError):
                result["dependency_fracture"][t] = 0.0

        # Combined relational instability: max of three signals
        for t in range(n_cycles):
            result["relational_instability"][t] = max(
                result["correlation_breakdown"][t],
                result["eigenvalue_instability"][t],
                result["dependency_fracture"][t],
            )

        return result

    def compute_correlation_breakdown(self, data: np.ndarray) -> np.ndarray:
        """Detect Layer 2 (Relational Instability): sensor correlation breakdown.

        Identifies cycles where pairwise sensor relationships change significantly.
        This indicates either dependency fracture or regime shift to new operating mode.

        Returns indices where significant correlation structure changes occur.
        """
        arr = np.asarray(data, dtype=float)
        if arr.shape[0] < 10:
            return np.array([], dtype=int)

        n_cycles = arr.shape[0]
        candidates = np.array([], dtype=int)

        # Use rolling window to detect correlation changes
        window_size = 15
        for t in range(window_size, n_cycles):
            window_data = arr[t - window_size : t]
            if window_data.shape[0] < 2:
                continue

            # Compute correlation matrix for current window
            recent_corr = np.corrcoef(window_data.T)
            recent_corr = np.nan_to_num(recent_corr, nan=0.0)

            # Compare to previous window
            if t >= 2 * window_size:
                prev_data = arr[t - 2 * window_size : t - window_size]
                prev_corr = np.corrcoef(prev_data.T)
                prev_corr = np.nan_to_num(prev_corr, nan=0.0)

                # Frobenius norm of correlation change
                corr_change = np.linalg.norm(recent_corr - prev_corr, "fro")

                # Detect significant changes (above 70th percentile threshold)
                if corr_change > 0.15:
                    candidates = np.append(candidates, t)

        return np.unique(candidates).astype(int)

    def detect_all_structural_changes(
        self, raw: np.ndarray, sensor_data: np.ndarray, baseline_std: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect Layer 3 (Trajectory Dynamics) and Layer 2 (Relational Instability) changes.

        Combines detection modes:
        - Acceleration spikes (2nd derivative): sudden curvature in drift trajectory
        - Correlation breakdown: sudden loss of pairwise sensor relationships

        Returns:
            (accel_candidates, corr_candidates): indices of detected changes
        """
        n_cycles = len(raw)

        # Acceleration candidates: 2nd derivative spikes with persistence check
        accel_candidates = np.array([], dtype=int)
        if n_cycles > 2:
            vel = np.diff(raw)
            accel = np.diff(vel)
            accel_threshold = 0.005 * baseline_std

            # Detect sustained acceleration (not just single spikes)
            for i in range(2, len(accel)):
                if accel[i] > accel_threshold:
                    # Check if preceded by elevated acceleration (persistence)
                    if i > 0 and accel[i - 1] > accel_threshold * 0.7:
                        accel_candidates = np.append(accel_candidates, i + 2)

        # Correlation candidates: correlation breakdown
        corr_candidates = self.compute_correlation_breakdown(sensor_data)

        return np.unique(accel_candidates).astype(int), corr_candidates
