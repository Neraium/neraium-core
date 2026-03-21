from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from neraium_core.sii.config import SIIConfig
from neraium_core.sii.errors import SIIIOError


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
        self._store_path = Path(self.config.regime_store_path)
        self.load()

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "regimes": list(self._regimes),
            "pending": None if self._pending is None else dict(self._pending),
        }

    def _sanitize_loaded(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        regimes = raw.get("regimes")
        pending = raw.get("pending")
        if isinstance(regimes, list):
            safe_regimes: list[dict[str, Any]] = []
            for item in regimes:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", f"regime_{len(safe_regimes)}"))
                signature = item.get("signature")
                prototypes = item.get("prototypes")
                hits = int(item.get("hits", 0))
                if not isinstance(signature, list) or not isinstance(prototypes, list):
                    continue
                safe_regimes.append(
                    {
                        "name": name,
                        "signature": signature,
                        "prototypes": prototypes,
                        "hits": hits,
                    }
                )
            self._regimes = safe_regimes
        if isinstance(pending, dict):
            name = str(pending.get("name", f"regime_{len(self._regimes)}"))
            signature = pending.get("signature")
            prototypes = pending.get("prototypes")
            hits = int(pending.get("hits", 1))
            if isinstance(signature, list) and isinstance(prototypes, list):
                self._pending = {
                    "name": name,
                    "signature": signature,
                    "prototypes": prototypes,
                    "hits": hits,
                }

    def save(self) -> None:
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._store_path.write_text(
                json.dumps(self.to_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            raise SIIIOError(f"Failed to write regime store: {self._store_path}") from exc

    def load(self) -> None:
        if not self._store_path.exists():
            return
        if not self._store_path.is_file():
            raise SIIIOError(f"Regime store path is not a file: {self._store_path}")
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SIIIOError(f"Failed to parse regime store: {self._store_path}") from exc
        self._sanitize_loaded(raw)
