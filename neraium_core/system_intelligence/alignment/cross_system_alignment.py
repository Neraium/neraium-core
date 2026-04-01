from __future__ import annotations

import math


def _euclid(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return (sum((a[i] - b[i]) ** 2 for i in range(n))) ** 0.5


class CrossSystemTrajectoryAlignment:
    """Compares canonical trajectories with interpretable DTW-like distances."""

    def compare(
        self,
        *,
        path_a: list[list[float]],
        path_b: list[list[float]],
        descriptor_a: dict[str, float],
        descriptor_b: dict[str, float],
    ) -> dict[str, float]:
        dtw = self._dtw_distance(path_a, path_b)
        shape = self._shape_distance(descriptor_a, descriptor_b)
        invariant = self._invariant_distance(descriptor_a, descriptor_b)
        composite_distance = 0.45 * dtw + 0.35 * shape + 0.2 * invariant
        similarity = math.exp(-composite_distance)
        return {
            "dtw_distance": round(float(dtw), 6),
            "shape_distance": round(float(shape), 6),
            "invariant_distance": round(float(invariant), 6),
            "composite_distance": round(float(composite_distance), 6),
            "similarity_score": round(float(similarity), 6),
        }

    def _dtw_distance(self, a: list[list[float]], b: list[list[float]]) -> float:
        n, m = len(a), len(b)
        dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = _euclid(a[i - 1], b[j - 1])
                dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return float(dp[n][m] / max(1, n + m))

    def _shape_distance(self, da: dict[str, float], db: dict[str, float]) -> float:
        keys = ["curvature_mean", "oscillation_ratio", "speed_var", "acceleration_var", "drift_signed"]
        return sum(abs(float(da.get(k, 0.0)) - float(db.get(k, 0.0))) for k in keys) / float(len(keys))

    def _invariant_distance(self, da: dict[str, float], db: dict[str, float]) -> float:
        keys = ["speed_mean", "acceleration_mean", "reversibility"]
        return sum((float(da.get(k, 0.0)) - float(db.get(k, 0.0))) ** 2 for k in keys) ** 0.5
