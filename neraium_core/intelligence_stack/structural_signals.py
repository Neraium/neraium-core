"""Relational Instability & Trajectory Dynamics layers of the Intelligence Stack.

Detects:
- Relational Instability: breakdown of sensor correlations and dependencies
- Trajectory Dynamics: acceleration spikes and change-point events

These components are part of the Intelligence Stack's evidence fusion layer.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class StructuralSignalDetector:
    """Detects genuine structural changes in system dynamics.

    Implements two Intelligence Stack layers:
    - Layer 2 (Relational Instability): correlation breakdown detection
    - Layer 3 (Trajectory Dynamics): acceleration and change-point detection
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

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

        Combines two detection modes:
        - Acceleration spikes (2nd derivative): sudden curvature in drift trajectory
        - Correlation breakdown: sudden loss of pairwise sensor relationships

        Returns:
            (accel_candidates, corr_candidates): indices of detected changes
        """
        n_cycles = len(raw)

        # Acceleration candidates: 2nd derivative spikes
        accel_candidates = np.array([], dtype=int)
        if n_cycles > 2:
            vel = np.diff(raw)
            accel = np.diff(vel)
            accel_threshold = 0.005 * baseline_std
            accel_indices = np.where(accel > accel_threshold)[0] + 2
            accel_candidates = np.unique(accel_indices)

        # Correlation candidates: correlation breakdown
        corr_candidates = self.compute_correlation_breakdown(sensor_data)

        return accel_candidates, corr_candidates
