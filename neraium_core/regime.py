from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from scipy.spatial import cKDTree
    _SCIPY_SPATIAL_AVAILABLE = True
except ImportError:
    _SCIPY_SPATIAL_AVAILABLE = False
    cKDTree = None  # type: ignore[assignment]

# Use KD-tree indexing when regime library exceeds this size for O(log k) lookups
_KDTREE_MIN_REGIMES = 20


@dataclass(frozen=True)
class StructuralOperator:
    """Derived operator over summary moments, subordinate to LawfulStructure."""

    center: np.ndarray
    spread: np.ndarray

    @classmethod
    def from_statistics(cls, mean: np.ndarray, std: np.ndarray) -> StructuralOperator:
        return cls(center=np.asarray(mean, dtype=float), spread=np.asarray(std, dtype=float))

    def apply(self) -> np.ndarray:
        return np.concatenate([self.center, self.spread]).astype(float, copy=False)


@dataclass
class LawfulStructure:
    """Coherence model for a regime; owns its StructuralOperator representation."""

    name: str
    operator: StructuralOperator
    prototypes: list[np.ndarray]
    coherence_threshold: float
    hits: int = 0
    pending: bool = False
    pending_hits: int = 0
    frozen_baseline_remaining: int = 0

    def nearest_distance(self, signature: np.ndarray) -> tuple[float, int]:
        sig = np.asarray(signature, dtype=float)
        best = float("inf")
        best_idx = 0
        for idx, proto in enumerate(self.prototypes):
            if proto.shape != sig.shape:
                continue
            dist = float(np.linalg.norm(sig - proto))
            if dist < best:
                best = dist
                best_idx = idx
        return best, best_idx

    def coherence_margin(self, signature: np.ndarray) -> float:
        dist, _ = self.nearest_distance(signature)
        if not np.isfinite(dist):
            return float("-inf")
        return float(self.coherence_threshold - dist)


def build_regime_signature(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Build a regime signature using the StructuralOperator induced by moments."""
    return StructuralOperator.from_statistics(mean, std).apply()


def regime_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _regime_prototypes(regime: dict[str, Any]) -> list[np.ndarray]:
    prototypes = regime.get("prototypes")
    if isinstance(prototypes, list) and prototypes:
        out: list[np.ndarray] = []
        for p in prototypes:
            try:
                out.append(np.asarray(p, dtype=float))
            except Exception:
                continue
        if out:
            return out
    return [np.asarray(regime.get("signature", []), dtype=float)]


def _lawful_structure_from_regime(regime: dict[str, Any], *, threshold: float) -> LawfulStructure:
    center = np.asarray(regime.get("mean", []), dtype=float)
    spread = np.asarray(regime.get("std", []), dtype=float)
    if center.size == 0 and spread.size == 0:
        signature = np.asarray(regime.get("signature", []), dtype=float)
        half = int(signature.size // 2)
        if half > 0:
            center = signature[:half]
            spread = signature[half:]
    operator = StructuralOperator(center=center, spread=spread)
    return LawfulStructure(
        name=str(regime.get("name", "unknown")),
        operator=operator,
        prototypes=_regime_prototypes(regime),
        coherence_threshold=float(regime.get("coherence_threshold", threshold)),
        hits=int(regime.get("hits", 0) or 0),
        pending=bool(regime.get("pending", False)),
        pending_hits=int(regime.get("pending_hits", 0) or 0),
        frozen_baseline_remaining=int(regime.get("frozen_baseline_remaining", 0) or 0),
    )


def assign_regime(
    signature: np.ndarray,
    regimes: list[dict[str, Any]],
    *,
    include_pending: bool = False,
    max_distance: float | None = None,
) -> dict[str, float | str] | None:
    if not regimes:
        return None

    sig = np.asarray(signature, dtype=float)
    threshold = float("inf") if max_distance is None else float(max_distance)

    # For large regime libraries, use KD-tree for faster nearest-neighbor search
    candidates = [r for r in regimes if not (r.get("pending", False) and not include_pending)]
    if not candidates:
        return None

    # Try KD-tree acceleration for large regime libraries
    if _SCIPY_SPATIAL_AVAILABLE and len(candidates) >= _KDTREE_MIN_REGIMES:
        try:
            # Build KD-tree from regime centroids
            centroids = []
            regime_indices = []
            for idx, regime in enumerate(candidates):
                center = np.asarray(regime.get("mean", []), dtype=float)
                if center.size > 0:
                    centroids.append(center)
                    regime_indices.append(idx)

            if centroids:
                tree = cKDTree(np.array(centroids))
                # Query for k nearest neighbors (use k=3 to account for closest matches)
                distances, indices = tree.query(sig[:len(centroids[0])], k=min(3, len(centroids)))

                if np.isscalar(distances):
                    # Single nearest neighbor
                    distances = np.array([distances])
                    indices = np.array([indices])

                # Check candidates in order of proximity
                nearest_name: str | None = None
                nearest_distance: float = float("inf")
                nearest_margin: float = float("-inf")
                nearest_index = 0

                for dist, idx in zip(distances, indices):
                    if idx >= len(regime_indices):
                        continue
                    regime_idx = regime_indices[idx]
                    regime = candidates[regime_idx]
                    structure = _lawful_structure_from_regime(regime, threshold=threshold)
                    full_dist, proto_idx = structure.nearest_distance(sig)

                    if not np.isfinite(full_dist):
                        continue
                    margin = structure.coherence_margin(sig)
                    if full_dist < nearest_distance:
                        nearest_distance = float(full_dist)
                        nearest_name = structure.name
                        nearest_margin = float(margin)
                        nearest_index = int(proto_idx)

                if nearest_name is not None and (max_distance is None or nearest_distance <= float(max_distance)):
                    return {
                        "name": nearest_name,
                        "distance": nearest_distance,
                        "prototype_index": nearest_index,
                        "coherence_margin": nearest_margin,
                    }
                return None
        except Exception:
            pass  # Fall through to linear search

    # Linear search fallback
    nearest_name: str | None = None
    nearest_distance: float = float("inf")
    nearest_margin: float = float("-inf")
    nearest_index = 0

    for regime in candidates:
        structure = _lawful_structure_from_regime(regime, threshold=threshold)
        dist, idx = structure.nearest_distance(sig)
        if not np.isfinite(dist):
            continue
        margin = structure.coherence_margin(sig)
        if dist < nearest_distance:
            nearest_distance = float(dist)
            nearest_name = structure.name
            nearest_margin = float(margin)
            nearest_index = int(idx)

    if nearest_name is None:
        return None
    if max_distance is not None and nearest_distance > float(max_distance):
        return None

    return {
        "name": nearest_name,
        "distance": nearest_distance,
        "prototype_index": nearest_index,
        "coherence_margin": nearest_margin,
    }


def _append_prototype(regime: dict[str, Any], signature: np.ndarray, *, max_prototypes: int) -> None:
    proto_list = regime.get("prototypes")
    if not isinstance(proto_list, list):
        proto_list = []
        regime["prototypes"] = proto_list
    proto_list.append(np.asarray(signature, dtype=float).tolist())
    if len(proto_list) > max(1, int(max_prototypes)):
        del proto_list[0 : len(proto_list) - int(max_prototypes)]

    sig_ref = np.asarray(signature, dtype=float)
    arrs = [np.asarray(p, dtype=float) for p in proto_list if isinstance(p, list)]
    arrs = [a for a in arrs if a.shape == sig_ref.shape]
    if arrs:
        stacked = np.stack(arrs, axis=0)
        mean_sig = np.mean(stacked, axis=0)
        regime["signature"] = mean_sig.tolist()
        half = int(mean_sig.size // 2)
        if half > 0:
            regime["mean"] = mean_sig[:half].tolist()
            regime["std"] = mean_sig[half:].tolist()


def update_regime_library(
    signature: np.ndarray,
    regimes: list[dict[str, Any]],
    threshold: float = 2.0,
    *,
    min_persistence: int = 3,
    freeze_baseline_frames: int = 12,
    max_prototypes: int = 5,
) -> list[dict[str, Any]]:
    sig = np.asarray(signature, dtype=float)

    if not regimes:
        half = int(sig.size // 2)
        regimes.append(
            {
                "name": "regime_0",
                "signature": sig.tolist(),
                "mean": sig[:half].tolist() if half > 0 else [],
                "std": sig[half:].tolist() if half > 0 else [],
                "prototypes": [sig.tolist()],
                "hits": 1,
                "pending": False,
                "pending_hits": 0,
                "coherence_threshold": float(threshold),
                "frozen_baseline_remaining": int(max(0, freeze_baseline_frames)),
            }
        )
        return regimes

    nearest_active = assign_regime(sig, regimes, include_pending=False)
    if nearest_active is not None and float(nearest_active.get("coherence_margin", -1.0)) >= 0.0:
        for regime in regimes:
            if str(regime.get("name")) == str(nearest_active["name"]):
                _append_prototype(regime, sig, max_prototypes=max_prototypes)
                regime["hits"] = int(regime.get("hits", 0) or 0) + 1
                regime["pending"] = False
                regime["pending_hits"] = 0
                regime["coherence_threshold"] = float(threshold)
                return regimes

    pending = [r for r in regimes if bool(r.get("pending", False))]
    nearest_pending = assign_regime(sig, pending, include_pending=True)
    if nearest_pending is not None and float(nearest_pending.get("coherence_margin", -1.0)) >= 0.0:
        for regime in pending:
            if str(regime.get("name")) == str(nearest_pending["name"]):
                _append_prototype(regime, sig, max_prototypes=max_prototypes)
                hits = int(regime.get("pending_hits", 0) or 0) + 1
                regime["pending_hits"] = hits
                regime["coherence_threshold"] = float(threshold)
                if hits >= int(max(1, min_persistence)):
                    regime["pending"] = False
                    regime["pending_hits"] = 0
                    regime["hits"] = int(regime.get("hits", 0) or 0) + 1
                    regime["frozen_baseline_remaining"] = int(max(0, freeze_baseline_frames))
                return regimes

    next_idx = len(regimes)
    half = int(sig.size // 2)
    candidate = {
        "name": f"regime_{next_idx}",
        "signature": sig.tolist(),
        "mean": sig[:half].tolist() if half > 0 else [],
        "std": sig[half:].tolist() if half > 0 else [],
        "prototypes": [sig.tolist()],
        "hits": 0,
        "pending": int(max(1, min_persistence)) > 1,
        "pending_hits": 1 if int(max(1, min_persistence)) > 1 else 0,
        "coherence_threshold": float(threshold),
        "frozen_baseline_remaining": int(max(0, freeze_baseline_frames)) if int(max(1, min_persistence)) <= 1 else 0,
    }

    if pending:
        replace = pending[0]
        replace.clear()
        replace.update(candidate)
    else:
        regimes.append(candidate)
    return regimes


def get_regime_entry(regimes: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for regime in regimes:
        if str(regime.get("name")) == str(name):
            return regime
    return None


def regime_library_stats(regimes: list[dict[str, Any]]) -> dict[str, int]:
    total = len(regimes)
    pending = sum(1 for r in regimes if bool(r.get("pending", False)))
    active = total - pending
    return {"total": int(total), "active": int(active), "pending": int(pending)}
