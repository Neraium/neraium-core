"""Fixed-size rolling buffers for streaming feature vectors."""

from __future__ import annotations

from collections import deque
from typing import Iterator

import numpy as np


class VectorDequeBuffer:
    """Deque of fixed max length holding 1D float vectors (same shape each step)."""

    def __init__(self, maxlen: int) -> None:
        self._dq: deque[np.ndarray] = deque(maxlen=max(1, int(maxlen)))

    def __len__(self) -> int:
        return len(self._dq)

    def clear(self) -> None:
        self._dq.clear()

    def append(self, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float64).ravel()
        self._dq.append(v.copy())

    def to_matrix(self) -> np.ndarray | None:
        if len(self._dq) == 0:
            return None
        return np.stack(list(self._dq), axis=0).astype(np.float64, copy=False)

    def iter_vectors(self) -> Iterator[np.ndarray]:
        yield from self._dq


class TimestampDequeBuffer:
    """Parallel timestamps for VectorDequeBuffer."""

    def __init__(self, maxlen: int) -> None:
        self._dq: deque[float] = deque(maxlen=max(1, int(maxlen)))

    def __len__(self) -> int:
        return len(self._dq)

    def clear(self) -> None:
        self._dq.clear()

    def append(self, ts: float) -> None:
        self._dq.append(float(ts))

    def to_list(self) -> list[float]:
        return list(self._dq)
