from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignmentResult:
    distance: float
    normalized_distance: float
    similarity: float
    path_length: int
    mean_step_cost: float
    max_step_cost: float
    diagonal_fraction: float


class DTWAligner:
    """Maintainable DTW aligner with optional Sakoe-Chiba style window constraint."""

    def __init__(self, *, window_ratio: float = 0.25, prefilter_weight: float = 0.25) -> None:
        self.window_ratio = max(0.0, float(window_ratio))
        self.prefilter_weight = max(0.0, min(1.0, float(prefilter_weight)))

    def compare(self, a: np.ndarray, b: np.ndarray) -> AlignmentResult:
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.size == 0 or b.size == 0:
            return AlignmentResult(
                distance=float("inf"),
                normalized_distance=float("inf"),
                similarity=0.0,
                path_length=0,
                mean_step_cost=float("inf"),
                max_step_cost=float("inf"),
                diagonal_fraction=0.0,
            )

        prefilter = self._prefilter_distance(a, b)
        dtw_distance, path = self._constrained_dtw(a, b)
        path_length = len(path)
        if path_length == 0:
            return AlignmentResult(
                distance=float("inf"),
                normalized_distance=float("inf"),
                similarity=0.0,
                path_length=0,
                mean_step_cost=float("inf"),
                max_step_cost=float("inf"),
                diagonal_fraction=0.0,
            )

        step_costs = [float(np.linalg.norm(a[i] - b[j])) for i, j in path]
        normalized = float(dtw_distance / max(1, path_length))
        combined = (1.0 - self.prefilter_weight) * normalized + self.prefilter_weight * prefilter
        diagonal_moves = 0
        for (pi, pj), (ci, cj) in zip(path[:-1], path[1:]):
            if ci - pi == 1 and cj - pj == 1:
                diagonal_moves += 1
        diagonal_fraction = float(diagonal_moves / max(1, path_length - 1))
        similarity = float(np.exp(-combined))
        return AlignmentResult(
            distance=float(dtw_distance),
            normalized_distance=combined,
            similarity=similarity,
            path_length=path_length,
            mean_step_cost=float(np.mean(step_costs)),
            max_step_cost=float(np.max(step_costs)),
            diagonal_fraction=diagonal_fraction,
        )

    def _prefilter_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        # low-cost summary distance to keep runtime reasonable before DTW dominates
        def summarize(x: np.ndarray) -> np.ndarray:
            radius = np.linalg.norm(x, axis=1)
            delta = np.diff(radius) if radius.size > 1 else np.asarray([0.0])
            return np.asarray(
                [
                    float(radius[-1] - radius[0]),
                    float(np.mean(delta)) if delta.size else 0.0,
                    float(np.std(delta)) if delta.size else 0.0,
                    float(np.max(radius) - np.min(radius)) if radius.size else 0.0,
                ],
                dtype=float,
            )

        sa = summarize(a)
        sb = summarize(b)
        return float(np.linalg.norm(sa - sb) / (np.sqrt(sa.size) + 1e-6))

    def _constrained_dtw(self, a: np.ndarray, b: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
        na, nb = a.shape[0], b.shape[0]
        band = max(abs(na - nb), int(np.ceil(max(na, nb) * self.window_ratio)))
        dp = np.full((na + 1, nb + 1), np.inf, dtype=float)
        back: dict[tuple[int, int], tuple[int, int]] = {}
        dp[0, 0] = 0.0

        for i in range(1, na + 1):
            j_lo = max(1, i - band)
            j_hi = min(nb, i + band)
            for j in range(j_lo, j_hi + 1):
                cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
                prevs = ((i - 1, j), (i, j - 1), (i - 1, j - 1))
                best_prev = min(prevs, key=lambda p: dp[p[0], p[1]])
                best = dp[best_prev[0], best_prev[1]]
                if np.isfinite(best):
                    dp[i, j] = cost + best
                    back[(i, j)] = best_prev

        if not np.isfinite(dp[na, nb]):
            return float("inf"), []

        path: list[tuple[int, int]] = []
        i, j = na, nb
        while (i, j) != (0, 0):
            path.append((i - 1, j - 1))
            i, j = back[(i, j)]
        path.reverse()
        return float(dp[na, nb]), path
