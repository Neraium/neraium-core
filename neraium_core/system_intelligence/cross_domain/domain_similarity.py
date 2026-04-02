from __future__ import annotations

from collections import defaultdict


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = max(1e-8, sum(v * v for v in a) ** 0.5)
    nb = max(1e-8, sum(v * v for v in b) ** 0.5)
    return dot / (na * nb)


class DomainSimilarityMapper:
    """Tracks structural similarity between systems (behavior-first, label-agnostic)."""

    def __init__(self) -> None:
        self._profiles: dict[str, list[list[float]]] = defaultdict(list)

    def update(self, *, system_type: str, descriptor_vector: list[float]) -> dict[str, object]:
        system = str(system_type or "unknown")
        self._profiles[system].append(list(map(float, descriptor_vector)))
        avg = {k: self._average(v) for k, v in self._profiles.items()}
        anchor = avg[system]
        scores = []
        for other, vec in avg.items():
            if other == system:
                continue
            score = _cosine(anchor, vec)
            scores.append((other, score))
        scores.sort(key=lambda row: row[1], reverse=True)
        nearest = [{"system_type": k, "structural_similarity_score": round(float(v), 6)} for k, v in scores[:3]]
        confidence = max(0.0, min(1.0, (nearest[0]["structural_similarity_score"] if nearest else 0.0)))
        return {
            "system_type": system,
            "nearest_systems": nearest,
            "transfer_confidence": round(float(confidence), 6),
        }

    def _average(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return [0.0]
        width = max(len(v) for v in vectors)
        out = [0.0] * width
        for vec in vectors:
            for i in range(width):
                out[i] += vec[i] if i < len(vec) else 0.0
        return [v / len(vectors) for v in out]
