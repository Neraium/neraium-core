from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neraium_core.system_intelligence.trajectory_alignment import DTWAligner


@dataclass
class TrajectorySegment:
    points: np.ndarray
    path_family: str


@dataclass
class TrajectoryArchetype:
    archetype_id: str
    prototype: np.ndarray
    support: int = 0
    escalation_events: int = 0
    phase_shift_events: int = 0
    steps_to_critical: list[int] = field(default_factory=list)
    next_path_counts: dict[str, int] = field(default_factory=dict)


class TrajectorySegmentEncoder:
    """Interpretable trajectory encoder: shape summary + normalized path signature."""

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = max(4, int(window_size))

    def build_segment(self, trajectory: list[list[float]]) -> TrajectorySegment | None:
        if len(trajectory) < 4:
            return None
        z = np.asarray(trajectory[-self.window_size :], dtype=float)
        if z.ndim != 2 or z.shape[0] < 4:
            return None
        if z.shape[0] != self.window_size:
            z = self._resample(z, self.window_size)
        family = self._infer_path_family(z)
        return TrajectorySegment(points=z, path_family=family)

    @staticmethod
    def _resample(points: np.ndarray, target_len: int) -> np.ndarray:
        if points.shape[0] == target_len:
            return points
        src = np.linspace(0.0, 1.0, points.shape[0])
        dst = np.linspace(0.0, 1.0, target_len)
        out = np.zeros((target_len, points.shape[1]), dtype=float)
        for i in range(points.shape[1]):
            out[:, i] = np.interp(dst, src, points[:, i])
        return out

    @staticmethod
    def _infer_path_family(points: np.ndarray) -> str:
        radius = np.linalg.norm(points, axis=1)
        drift = float(radius[-1] - radius[0])
        slope = float(np.mean(np.diff(radius)))
        signs = np.sign(np.diff(radius))
        flips = int(np.sum(signs[1:] * signs[:-1] < 0)) if signs.size > 1 else 0
        recovery = float(np.max(radius[:-1]) - radius[-1]) if radius.size > 2 else 0.0
        volatility = float(np.std(np.diff(radius))) if radius.size > 2 else 0.0
        recent_drop = float(np.max(radius[-3:]) - radius[-1]) if radius.size >= 3 else 0.0

        if drift > 1.4 and slope > 0.12:
            return "rapid-collapse"
        if drift > 0.55 and slope > 0.04:
            return "escalating"
        if recovery > 0.25 and drift < -0.12:
            return "reversible"
        if flips >= max(2, int(len(radius) * 0.22)) and volatility > 0.05:
            return "oscillatory"
        if abs(drift) < 0.18 and volatility < 0.06:
            return "stable"
        if recent_drop > 0.3 and drift < 0.05:
            return "reversible"
        return "drift"


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    aligner = DTWAligner(window_ratio=0.25, prefilter_weight=0.0)
    return float(aligner.compare(a, b).normalized_distance)
