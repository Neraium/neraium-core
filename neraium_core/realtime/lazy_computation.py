"""Lazy computation utilities for deferring expensive operations until needed.

Supports conditional skipping of spectral and other heavy analytics based
on system state and thresholds.
"""
from __future__ import annotations

from typing import Callable, Optional, TypeVar, Any
import numpy as np

T = TypeVar("T")


class LazyComputation:
    """Wrapper for lazy evaluation of expensive computations.

    Supports conditional execution based on triggers (thresholds, anomaly scores, etc).
    Memoizes results to avoid recomputation.
    """

    def __init__(
        self,
        fn: Callable[[], T],
        *,
        trigger: Optional[Callable[[], bool]] = None,
        cache_ttl: Optional[int] = None,
    ):
        """Initialize lazy computation.

        Args:
            fn: Function that performs the expensive computation (no args)
            trigger: Optional callable that returns True if computation should run.
                    If None or returns False, returns None instead of computing.
            cache_ttl: Optional cache time-to-live in frames. If provided, cached
                      results are invalidated after N frames.
        """
        self.fn = fn
        self.trigger = trigger or (lambda: True)
        self.cache_ttl = cache_ttl
        self._cached_result: Optional[T] = None
        self._frame_cached = -1
        self._current_frame = 0

    def compute(self) -> Optional[T]:
        """Execute computation if trigger is satisfied, else return cached or None.

        Returns:
            Result of fn() if trigger is True and cache miss, else cached result or None.
        """
        self._current_frame += 1

        # Check if we have valid cached result
        if self._cached_result is not None:
            if self.cache_ttl is None or (self._current_frame - self._frame_cached) < self.cache_ttl:
                return self._cached_result

        # Check trigger condition
        if not self.trigger():
            return None

        # Compute and cache
        self._cached_result = self.fn()
        self._frame_cached = self._current_frame
        return self._cached_result

    def invalidate(self) -> None:
        """Force invalidation of cached result."""
        self._cached_result = None
        self._frame_cached = -1


class ThresholdBasedLazyComputation:
    """Lazy computation that only executes when a score exceeds a threshold.

    Useful for skipping expensive spectral and causal analytics until
    a system shows signs of anomaly.
    """

    def __init__(
        self,
        fn: Callable[[], T],
        threshold: float = 0.7,
        cooldown_frames: int = 5,
    ):
        """Initialize threshold-based lazy computation.

        Args:
            fn: Function that performs the computation (no args)
            threshold: Score threshold above which computation is triggered
            cooldown_frames: After computation, skip for N frames even if score drops
        """
        self.fn = fn
        self.threshold = threshold
        self.cooldown_frames = cooldown_frames
        self._cached_result: Optional[T] = None
        self._cooldown_until = 0
        self._current_frame = 0

    def compute(self, score: float) -> Optional[T]:
        """Execute computation if score exceeds threshold.

        Args:
            score: Current anomaly/instability score [0, 1+]

        Returns:
            Result if computation runs, else cached or None
        """
        self._current_frame += 1

        # Return cached result during cooldown
        if self._current_frame < self._cooldown_until:
            return self._cached_result

        # Execute if score exceeds threshold
        if score >= self.threshold:
            self._cached_result = self.fn()
            self._cooldown_until = self._current_frame + self.cooldown_frames
            return self._cached_result

        # Below threshold, return nothing but keep cache
        return None

    def force_compute(self) -> T:
        """Force immediate computation regardless of threshold."""
        self._cached_result = self.fn()
        self._cooldown_until = self._current_frame + self.cooldown_frames
        return self._cached_result

    def get_cached(self) -> Optional[T]:
        """Get last cached result without computing."""
        return self._cached_result


class BatchSkipComputation:
    """Skip expensive computations on a schedule (e.g., every Nth frame).

    Useful for subsetting expensive analytics to maintain latency SLO.
    """

    def __init__(self, fn: Callable[[], T], skip_interval: int = 3):
        """Initialize batch skip computation.

        Args:
            fn: Function to compute
            skip_interval: Skip computation for skip_interval-1 out of every skip_interval frames
        """
        self.fn = fn
        self.skip_interval = max(1, skip_interval)
        self._frame_count = 0
        self._cached_result: Optional[T] = None

    def should_compute(self) -> bool:
        """Check if computation should happen this frame."""
        result = self._frame_count % self.skip_interval == 0
        self._frame_count += 1
        return result

    def compute(self) -> Optional[T]:
        """Compute if scheduled, else return cached result."""
        if self.should_compute():
            self._cached_result = self.fn()
        return self._cached_result

    def get_cached(self) -> Optional[T]:
        """Get last cached result."""
        return self._cached_result
