from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return (_mean([(v - m) ** 2 for v in values])) ** 0.5


@dataclass
class CanonicalTrajectory:
    system_type: str
    domain: str
    normalized_path: list[list[float]]
    descriptor_vector: list[float]
    descriptor_map: dict[str, float]


class CanonicalStructuralRepresentation:
    """Experimental domain-agnostic trajectory representation for cross-domain comparison."""

    def __init__(self, *, target_length: int = 16) -> None:
        self.target_length = int(max(8, target_length))

    def encode(
        self,
        *,
        system_type: str,
        domain: str,
        latent_trajectory: list[list[float]],
    ) -> CanonicalTrajectory:
        path = [list(map(float, row)) for row in latent_trajectory if row]
        if len(path) < 2:
            base = [0.0, 0.0, 0.0]
            path = [base, [0.0, 0.0, 0.01]]
        normalized = self._normalize(path)
        resampled = self._resample(normalized)
        descriptor = self._describe(resampled)
        return CanonicalTrajectory(
            system_type=str(system_type or "unknown"),
            domain=str(domain or "unknown"),
            normalized_path=resampled,
            descriptor_vector=[descriptor[k] for k in sorted(descriptor.keys())],
            descriptor_map=descriptor,
        )

    def _normalize(self, path: list[list[float]]) -> list[list[float]]:
        dims = len(path[0])
        cols = [[row[d] for row in path] for d in range(dims)]
        means = [_mean(c) for c in cols]
        scales = [max(_std(c), 1e-6) for c in cols]
        centered = [[(row[d] - means[d]) / scales[d] for d in range(dims)] for row in path]
        max_norm = max((sum(v * v for v in row) ** 0.5 for row in centered), default=1.0)
        scale = max(1.0, max_norm)
        return [[v / scale for v in row] for row in centered]

    def _resample(self, path: list[list[float]]) -> list[list[float]]:
        if len(path) == self.target_length:
            return path
        dims = len(path[0])
        out: list[list[float]] = []
        for i in range(self.target_length):
            t = i * (len(path) - 1) / max(1, self.target_length - 1)
            lo = int(t)
            hi = min(len(path) - 1, lo + 1)
            alpha = float(t - lo)
            row = [(1.0 - alpha) * path[lo][d] + alpha * path[hi][d] for d in range(dims)]
            out.append(row)
        return out

    def _describe(self, path: list[list[float]]) -> dict[str, float]:
        velocities = [
            (sum((path[i + 1][d] - path[i][d]) ** 2 for d in range(len(path[i]))) ** 0.5)
            for i in range(len(path) - 1)
        ]
        accelerations = [velocities[i + 1] - velocities[i] for i in range(len(velocities) - 1)]
        direction_changes: list[float] = []
        for i in range(1, len(path) - 1):
            v1 = [path[i][d] - path[i - 1][d] for d in range(len(path[i]))]
            v2 = [path[i + 1][d] - path[i][d] for d in range(len(path[i]))]
            n1 = max(1e-8, sum(x * x for x in v1) ** 0.5)
            n2 = max(1e-8, sum(x * x for x in v2) ** 0.5)
            cosang = max(-1.0, min(1.0, sum(v1[d] * v2[d] for d in range(len(v1))) / (n1 * n2)))
            direction_changes.append((1.0 - cosang) / 2.0)

        start_norm = sum(v * v for v in path[0]) ** 0.5
        end_norm = sum(v * v for v in path[-1]) ** 0.5
        signed_drift = end_norm - start_norm
        pos_steps = sum(1 for a in accelerations if a > 0.0)
        neg_steps = sum(1 for a in accelerations if a < 0.0)
        reversibility = 1.0 - abs(signed_drift)
        descriptor = {
            "acceleration_mean": _mean(accelerations),
            "acceleration_var": _std(accelerations),
            "curvature_mean": _mean(direction_changes),
            "drift_signed": signed_drift,
            "end_start_ratio": end_norm / max(1e-6, start_norm + 1e-3),
            "oscillation_ratio": min(pos_steps, neg_steps) / max(1.0, float(len(accelerations))),
            "reversibility": _clamp(reversibility),
            "speed_mean": _mean(velocities),
            "speed_var": _std(velocities),
        }
        return {k: round(float(v), 6) for k, v in descriptor.items()}


def canonical_to_dict(canonical: CanonicalTrajectory) -> dict[str, Any]:
    return {
        "system_type": canonical.system_type,
        "domain": canonical.domain,
        "normalized_path": [[round(float(v), 6) for v in row] for row in canonical.normalized_path],
        "descriptor_map": canonical.descriptor_map,
        "descriptor_vector": [round(float(v), 6) for v in canonical.descriptor_vector],
    }
