from __future__ import annotations

import math
import numpy as np


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)))


def _safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    na = _safe_norm(a)
    nb = _safe_norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 1.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def compute_trajectory_geometry(path: list[np.ndarray], window: int = 8) -> dict[str, float]:
    if len(path) < 2:
        return {
            "path_length": 0.0,
            "local_velocity_norm": 0.0,
            "local_acceleration_norm": 0.0,
            "curvature": 0.0,
            "directional_consistency": 1.0,
            "angular_change": 0.0,
            "path_smoothness": 1.0,
            "directional_deviation": 0.0,
        }

    tail = [np.asarray(v, dtype=float) for v in path[-max(2, window):]]
    velocity = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    acceleration = [velocity[i] - velocity[i - 1] for i in range(1, len(velocity))]

    step_norms = [_safe_norm(v) for v in velocity]
    path_length = float(np.sum(step_norms))
    local_velocity = float(step_norms[-1]) if step_norms else 0.0
    local_accel = float(_safe_norm(acceleration[-1])) if acceleration else 0.0

    cosines = [_safe_cos(velocity[i - 1], velocity[i]) for i in range(1, len(velocity))]
    directional_consistency = float(np.mean([(c + 1.0) * 0.5 for c in cosines])) if cosines else 1.0
    angular_change = float(np.mean([math.acos(c) for c in cosines])) if cosines else 0.0

    curvature = 0.0
    if acceleration and velocity:
        denom = max(1e-9, local_velocity**2)
        curvature = float(local_accel / denom)

    smoothness = 1.0
    if step_norms:
        smoothness = float(max(0.0, 1.0 - (np.std(step_norms) / (np.mean(step_norms) + 1e-9))))

    directional_deviation = float(max(0.0, 1.0 - directional_consistency))
    return {
        "path_length": round(path_length, 6),
        "local_velocity_norm": round(local_velocity, 6),
        "local_acceleration_norm": round(local_accel, 6),
        "curvature": round(curvature, 6),
        "directional_consistency": round(directional_consistency, 6),
        "angular_change": round(angular_change, 6),
        "path_smoothness": round(smoothness, 6),
        "directional_deviation": round(directional_deviation, 6),
    }
