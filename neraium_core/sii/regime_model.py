from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neraium_core.sii.config import SIIConfig


@dataclass(frozen=True)
class RegimeObservation:
    signature: np.ndarray
    graph_signature: np.ndarray


@dataclass(frozen=True)
class RegimeResult:
    regime_name: str
    regime_distance: float
    regime_support: float
    regime_activated: bool


class RegimeModel:
    """
    Regime modeling for stable operating structure.
    """

    def __init__(self, *, config: SIIConfig) -> None:
        self.config = config
        self._regimes: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return 0.0
        return float(np.linalg.norm(a - b))

    def _nearest(self, signature: np.ndarray) -> tuple[dict[str, Any] | None, float]:
        best_reg: dict[str, Any] | None = None
        best_dist = float("inf")
        for reg in self._regimes:
            prototypes = reg.get("prototypes", [])
            for p in prototypes:
                proto = np.asarray(p, dtype=float)
                if proto.shape != signature.shape:
                    continue
                d = self._distance(signature, proto)
                if d < best_dist:
                    best_dist = d
                    best_reg = reg
        if best_reg is None:
            return None, 0.0
        return best_reg, float(best_dist)

    def _append_prototype(self, reg: dict[str, Any], signature: np.ndarray) -> None:
        protos = reg.setdefault("prototypes", [])
        protos.append(np.asarray(signature, dtype=float).tolist())
        max_p = int(self.config.regime_max_prototypes)
        if len(protos) > max_p:
            del protos[0 : len(protos) - max_p]
        arr = np.asarray(protos, dtype=float)
        reg["signature"] = np.mean(arr, axis=0).tolist()

    def observe(self, obs: RegimeObservation) -> RegimeResult:
        signature = np.concatenate([obs.signature, obs.graph_signature])
        if not self._regimes:
            reg = {
                "name": "regime_0",
                "prototypes": [signature.tolist()],
                "signature": signature.tolist(),
                "hits": 1,
            }
            self._regimes.append(reg)
            return RegimeResult(
                regime_name="regime_0",
                regime_distance=0.0,
                regime_support=1.0,
                regime_activated=True,
            )

        nearest, dist = self._nearest(signature)
        threshold = float(self.config.regime_distance_threshold)
        if nearest is not None and dist <= threshold:
            self._append_prototype(nearest, signature)
            nearest["hits"] = int(nearest.get("hits", 0)) + 1
            return RegimeResult(
                regime_name=str(nearest["name"]),
                regime_distance=float(dist),
                regime_support=max(0.0, min(1.0, float(nearest["hits"]) / 12.0)),
                regime_activated=False,
            )

        # Pending regime candidate requires repeated observations.
        if self._pending is None:
            self._pending = {
                "name": f"regime_{len(self._regimes)}",
                "signature": signature.tolist(),
                "prototypes": [signature.tolist()],
                "hits": 1,
            }
            return RegimeResult(
                regime_name=str(self._pending["name"]),
                regime_distance=float(dist),
                regime_support=0.1,
                regime_activated=False,
            )

        pending_sig = np.asarray(self._pending["signature"], dtype=float)
        pending_dist = self._distance(signature, pending_sig)
        if pending_dist <= threshold:
            self._pending["hits"] = int(self._pending.get("hits", 1)) + 1
            self._pending["prototypes"].append(signature.tolist())
            if int(self._pending["hits"]) >= int(self.config.regime_min_persistence):
                activated = dict(self._pending)
                activated["hits"] = int(activated.get("hits", 1))
                self._regimes.append(activated)
                self._pending = None
                return RegimeResult(
                    regime_name=str(activated["name"]),
                    regime_distance=float(pending_dist),
                    regime_support=max(0.0, min(1.0, float(activated["hits"]) / 12.0)),
                    regime_activated=True,
                )
            return RegimeResult(
                regime_name=str(self._pending["name"]),
                regime_distance=float(pending_dist),
                regime_support=max(0.0, min(1.0, float(self._pending["hits"]) / 12.0)),
                regime_activated=False,
            )

        # Replace pending if it diverges too far from repeated evidence.
        self._pending = {
            "name": f"regime_{len(self._regimes)}",
            "signature": signature.tolist(),
            "prototypes": [signature.tolist()],
            "hits": 1,
        }
        return RegimeResult(
            regime_name=str(self._pending["name"]),
            regime_distance=float(dist),
            regime_support=0.1,
            regime_activated=False,
        )
