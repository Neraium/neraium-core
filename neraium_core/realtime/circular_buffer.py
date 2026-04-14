"""High-performance circular buffers for streaming data."""
from __future__ import annotations

import numpy as np
from typing import Optional


class CircularNumpyBuffer:
    """Circular buffer using pre-allocated numpy arrays for O(1) append and fast iteration.

    Replaces deque for scenarios with large numbers of appends, providing better
    cache locality and avoiding Python list allocation overhead.
    """

    def __init__(self, maxlen: int, dtype: type = np.float64):
        """Initialize circular buffer.

        Args:
            maxlen: Maximum number of elements
            dtype: Numpy dtype for the buffer
        """
        self.maxlen = maxlen
        self.dtype = dtype
        self.buffer = np.empty(maxlen, dtype=dtype)
        self.pos = 0  # Write position (incremented mod maxlen)
        self.count = 0  # Total elements added (used to check if full)

    def append(self, value: float) -> None:
        """Append a value to the buffer (O(1))."""
        self.buffer[self.pos] = value
        self.pos = (self.pos + 1) % self.maxlen
        if self.count < self.maxlen:
            self.count += 1

    def to_array(self) -> np.ndarray:
        """Get contents as numpy array in chronological order (O(n))."""
        if self.count < self.maxlen:
            return self.buffer[: self.count]
        # Rotate to chronological order
        return np.concatenate([self.buffer[self.pos :], self.buffer[: self.pos]])

    def to_list(self) -> list:
        """Get contents as Python list in chronological order."""
        return self.to_array().tolist()

    def __len__(self) -> int:
        """Return number of elements in buffer."""
        return self.count

    def __getitem__(self, idx: int) -> float:
        """Get element by index (0-based, chronological). Supports negative indexing."""
        # Handle negative indexing
        if idx < 0:
            idx = self.count + idx
        if idx < 0 or idx >= self.count:
            raise IndexError("list index out of range")
        actual_idx = (self.pos + idx) % self.maxlen
        return float(self.buffer[actual_idx])


class CircularMatrix2DBuffer:
    """Circular buffer for 2D matrices (time x features).

    Stores row vectors in a pre-allocated array, providing fast iteration
    without the overhead of list/deque of arrays.
    """

    def __init__(self, maxlen: int, n_features: int, dtype: type = np.float64):
        """Initialize matrix buffer.

        Args:
            maxlen: Maximum number of rows
            n_features: Number of features per row
            dtype: Numpy dtype
        """
        self.maxlen = maxlen
        self.n_features = n_features
        self.dtype = dtype
        self.buffer = np.empty((maxlen, n_features), dtype=dtype)
        self.pos = 0
        self.count = 0

    def append(self, row: np.ndarray) -> None:
        """Append a row vector (O(n_features))."""
        if row.shape[0] != self.n_features:
            raise ValueError(f"Row shape {row.shape[0]} != {self.n_features}")
        self.buffer[self.pos] = row
        self.pos = (self.pos + 1) % self.maxlen
        if self.count < self.maxlen:
            self.count += 1

    def to_array(self) -> np.ndarray:
        """Get contents as 2D array in chronological order (O(n))."""
        if self.count < self.maxlen:
            return self.buffer[: self.count].copy()
        # Rotate to chronological order
        return np.vstack([self.buffer[self.pos :], self.buffer[: self.pos]])

    def __len__(self) -> int:
        """Return number of rows."""
        return self.count

    def recent(self, n: int) -> Optional[np.ndarray]:
        """Get last n rows in chronological order, or None if fewer available."""
        if self.count < n:
            return None
        if self.count < self.maxlen:
            return self.buffer[max(0, self.count - n) : self.count]
        # Circular case: might wrap
        if self.pos >= n:
            return self.buffer[self.pos - n : self.pos]
        else:
            return np.vstack([self.buffer[self.maxlen + self.pos - n :], self.buffer[: self.pos]])

    def baseline(self, n: int) -> Optional[np.ndarray]:
        """Get first n rows in chronological order, or None if fewer available."""
        if self.count < n:
            return None
        arr = self.to_array()
        return arr[:n]
