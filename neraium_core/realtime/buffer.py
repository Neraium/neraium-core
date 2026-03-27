"""Fixed-size rolling buffers for streaming feature vectors."""

from __future__ import annotations

from collections import deque
from typing import Iterator

import numpy as np


class HistoryRingBuffer:
    """Circular storage for (feature_row, timestamp) aligned with ``StructuralEngine.frames``.

    Appends are O(d) with no reallocations. ``chronological_matrix`` copies rows into a
    reusable workspace in oldest→newest order (one O(count·d) pass per call).
    """

    def __init__(self, maxlen: int = 500) -> None:
        self._maxlen = max(1, int(maxlen))
        self._d = 0
        self._buf: np.ndarray | None = None
        self._ts: np.ndarray | None = None
        self._start = 0
        self._count = 0
        self._workspace: np.ndarray | None = None
        self._workspace_ts: np.ndarray | None = None
        self._tail_workspace: np.ndarray | None = None
        self._max_tail = 48

    def _ensure(self, d: int) -> None:
        if self._buf is not None and self._d == d:
            return
        if self._buf is not None and self._d != d:
            self.clear()
        self._d = int(d)
        self._buf = np.zeros((self._maxlen, self._d), dtype=np.float64)
        self._ts = np.zeros(self._maxlen, dtype=np.float64)
        self._workspace = np.zeros((self._maxlen, self._d), dtype=np.float64)
        self._workspace_ts = np.zeros(self._maxlen, dtype=np.float64)
        self._tail_workspace = np.zeros((self._max_tail, self._d), dtype=np.float64)

    def clear(self) -> None:
        self._start = 0
        self._count = 0

    def append(self, row: np.ndarray, ts_value: float) -> None:
        v = np.asarray(row, dtype=np.float64).ravel()
        self._ensure(v.size)
        assert self._buf is not None and self._ts is not None
        if self._count < self._maxlen:
            idx = (self._start + self._count) % self._maxlen
            self._buf[idx] = v
            self._ts[idx] = float(ts_value)
            self._count += 1
        else:
            self._buf[self._start] = v
            self._ts[self._start] = float(ts_value)
            self._start = (self._start + 1) % self._maxlen

    def __len__(self) -> int:
        return self._count

    def rebuild_from_frames(self, frames: list[dict]) -> None:
        self.clear()
        for i, f in enumerate(frames):
            v = f.get("_vector")
            if v is None:
                continue
            try:
                ts = float(f.get("timestamp"))
            except (TypeError, ValueError):
                ts = float(i)
            self.append(np.asarray(v, dtype=float), ts)

    def chronological_matrix(self) -> np.ndarray:
        if self._count == 0 or self._buf is None or self._workspace is None:
            return np.zeros((0, max(0, self._d)), dtype=np.float64)
        idx = (self._start + np.arange(self._count)) % self._maxlen
        out = self._workspace[: self._count]
        np.copyto(out, self._buf[idx])
        return out

    def chronological_timestamps(self) -> np.ndarray:
        if self._count == 0 or self._ts is None or self._workspace_ts is None:
            return np.zeros(0, dtype=np.float64)
        idx = (self._start + np.arange(self._count)) % self._maxlen
        wts = self._workspace_ts[: self._count]
        np.copyto(wts, self._ts[idx])
        return wts

    def chronological_tail_matrix(self, k: int) -> np.ndarray:
        """Last ``k`` rows in time order; reuses tail workspace (``k`` ≤ ``_max_tail``)."""
        if self._count == 0 or self._buf is None or self._tail_workspace is None:
            return np.zeros((0, max(0, self._d)), dtype=np.float64)
        kk = min(int(k), self._count)
        if kk > self._max_tail:
            m = self.chronological_matrix()
            return np.asarray(m[-kk:], dtype=np.float64, order="C")
        idx = (self._start + np.arange(self._count - kk, self._count)) % self._maxlen
        out = self._tail_workspace[:kk]
        np.copyto(out, self._buf[idx])
        return out


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
