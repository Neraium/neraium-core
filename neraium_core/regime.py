from __future__ import annotations

from typing import Any

import numpy as np


def build_regime_signature(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Build a regime signature from per-signal mean and std."""
    return np.concatenate([np.asarray(mean, dtype=float), np.asarray(std, dtype=float)])


def regime_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two regime signatures."""
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


def assign_regime(
    signature: np.ndarray,
    regimes: list[dict[str, Any]],
    *,
    include_pending: bool = False,
    max_distance: float | None = None,
) -> dict[str, float | str] | None:
    """
    Find the nearest known regime (optionally including pending candidates).
    """
    if not regimes:
        return None

    distances: list[tuple[float, str, int]] = []
    sig = np.asarray(signature, dtype=float)
    for regime in regimes:
        if bool(regime.get("pending", False)) and not include_pending:
            continue
        for idx, proto in enumerate(_regime_prototypes(regime)):
            # Sensor sets/window configuration can change signature dimensionality over
            # time. Skip incompatible regimes instead of failing with broadcasting.
            if proto.shape != sig.shape:
                continue
            distances.append((regime_distance(sig, proto), str(regime["name"]), idx))

    distances.sort(key=lambda x: x[0])
    if not distances:
        return None
    nearest_distance, nearest_name, prototype_index = distances[0]
    if max_distance is not None and float(nearest_distance) > float(max_distance):
        return None
    return {
        "name": nearest_name,
        "distance": float(nearest_distance),
        "prototype_index": int(prototype_index),
    }


def _append_prototype(regime: dict[str, Any], signature: np.ndarray, *, max_prototypes: int) -> None:
    proto_list = regime.get("prototypes")
    if not isinstance(proto_list, list):
        proto_list = []
        regime["prototypes"] = proto_list
    proto_list.append(np.asarray(signature, dtype=float).tolist())
    if len(proto_list) > max(1, int(max_prototypes)):
        # Keep most recent prototypes to avoid overfitting to stale signatures.
        del proto_list[0 : len(proto_list) - int(max_prototypes)]
    arr = np.asarray(proto_list, dtype=float)
    if arr.ndim == 2 and arr.shape[0] >= 1:
        regime["signature"] = np.mean(arr, axis=0).tolist()


def update_regime_library(
    signature: np.ndarray,
    regimes: list[dict[str, Any]],
    threshold: float = 2.0,
    *,
    min_persistence: int = 3,
    freeze_baseline_frames: int = 12,
    max_prototypes: int = 5,
) -> list[dict[str, Any]]:
    """
    Update regime memory with persistence and multi-prototype support.
    """
    sig = np.asarray(signature, dtype=float)
    if not regimes:
        regimes.append(
            {
                "name": "regime_0",
                "signature": sig.tolist(),
                "prototypes": [sig.tolist()],
                "hits": 1,
                "pending": False,
                "pending_hits": 0,
                "frozen_baseline_remaining": int(max(0, freeze_baseline_frames)),
            }
        )
        return regimes

    nearest_active = assign_regime(sig, regimes, include_pending=False)
    if nearest_active is not None and float(nearest_active["distance"]) <= float(threshold):
        for regime in regimes:
            if str(regime.get("name")) == str(nearest_active["name"]):
                _append_prototype(regime, sig, max_prototypes=max_prototypes)
                regime["hits"] = int(regime.get("hits", 0) or 0) + 1
                regime["pending"] = False
                regime["pending_hits"] = 0
                return regimes

    pending = [r for r in regimes if bool(r.get("pending", False))]
    nearest_pending = assign_regime(sig, pending, include_pending=True)
    if nearest_pending is not None and float(nearest_pending["distance"]) <= float(threshold):
        for regime in pending:
            if str(regime.get("name")) == str(nearest_pending["name"]):
                _append_prototype(regime, sig, max_prototypes=max_prototypes)
                hits = int(regime.get("pending_hits", 0) or 0) + 1
                regime["pending_hits"] = hits
                if hits >= int(max(1, min_persistence)):
                    regime["pending"] = False
                    regime["pending_hits"] = 0
                    regime["hits"] = int(regime.get("hits", 0) or 0) + 1
                    regime["frozen_baseline_remaining"] = int(max(0, freeze_baseline_frames))
                return regimes

    # Keep at most one pending candidate at a time to avoid candidate explosion.
    next_idx = len(regimes)
    candidate = {
        "name": f"regime_{next_idx}",
        "signature": sig.tolist(),
        "prototypes": [sig.tolist()],
        "hits": 0,
        "pending": int(max(1, min_persistence)) > 1,
        "pending_hits": 1 if int(max(1, min_persistence)) > 1 else 0,
        "frozen_baseline_remaining": int(max(0, freeze_baseline_frames))
        if int(max(1, min_persistence)) <= 1
        else 0,
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