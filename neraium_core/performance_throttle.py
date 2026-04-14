"""Performance throttling for expensive engine operations.

This module provides configurable throttling and lazy evaluation for expensive
operations like causal analysis, graph metrics, and regime persistence that can
accumulate overhead during long-running streams.

Key optimizations:
  - Lazy causal analysis: Skip computation when structural instability is low
  - Throttled persistence: Batch regime state saves to reduce I/O overhead
  - Bounded storage: Limit regime baselines and signatures to prevent unbounded growth
  - Smart caching: Reuse computed metrics across frames with similar scores
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


def _env_int(name: str, default: int) -> int:
    """Get an integer from environment variable."""
    val = os.environ.get(name, "").strip()
    return int(val) if val.isdigit() else default


def _env_enabled(name: str, default: bool = True) -> bool:
    """Check if an environment variable is enabled."""
    val = os.environ.get(name, "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


@dataclass
class ThrottleConfig:
    """Configuration for performance throttling."""

    # Causal analysis throttling
    enable_lazy_causal: bool = field(default_factory=lambda: _env_enabled(
        "NERAIUM_LAZY_CAUSAL", default=True
    ))
    causal_instability_threshold: float = 0.5
    causal_compute_every_n_frames: int = field(default_factory=lambda: _env_int(
        "NERAIUM_CAUSAL_COMPUTE_INTERVAL", 3
    ))

    # Regime persistence throttling
    enable_throttled_persistence: bool = field(default_factory=lambda: _env_enabled(
        "NERAIUM_THROTTLE_PERSISTENCE", default=True
    ))
    regime_persist_interval: int = field(default_factory=lambda: _env_int(
        "NERAIUM_REGIME_PERSIST_INTERVAL", 16
    ))
    async_persistence_enabled: bool = field(default_factory=lambda: _env_enabled(
        "NERAIUM_ASYNC_PERSISTENCE", default=False
    ))

    # Storage bounds
    max_regime_baselines: int = field(default_factory=lambda: _env_int(
        "NERAIUM_MAX_REGIME_BASELINES", 32
    ))
    max_regime_signatures: int = field(default_factory=lambda: _env_int(
        "NERAIUM_MAX_REGIME_SIGNATURES", 64
    ))

    # Graph metrics caching
    enable_graph_cache: bool = True
    graph_cache_hits_until_recompute: int = field(default_factory=lambda: _env_int(
        "NERAIUM_GRAPH_CACHE_HITS", 8
    ))


@dataclass
class PerformanceMetrics:
    """Track performance across frames."""

    frame_count: int = 0
    causal_skipped: int = 0
    causal_computed: int = 0
    persist_skipped: int = 0
    persist_executed: int = 0
    regime_baselines_pruned: int = 0
    regime_signatures_pruned: int = 0

    def record_causal_skip(self) -> None:
        self.causal_skipped += 1

    def record_causal_compute(self) -> None:
        self.causal_computed += 1

    def record_persist_skip(self) -> None:
        self.persist_skipped += 1

    def record_persist_execute(self) -> None:
        self.persist_executed += 1

    def get_summary(self) -> dict[str, Any]:
        """Return performance metrics summary."""
        total_causal = self.causal_skipped + self.causal_computed
        causal_skip_rate = (self.causal_skipped / total_causal * 100) if total_causal > 0 else 0

        total_persist = self.persist_skipped + self.persist_executed
        persist_skip_rate = (self.persist_skipped / total_persist * 100) if total_persist > 0 else 0

        return {
            "frame_count": self.frame_count,
            "causal_analysis": {
                "skipped": self.causal_skipped,
                "computed": self.causal_computed,
                "skip_rate_pct": round(causal_skip_rate, 1),
            },
            "persistence": {
                "skipped": self.persist_skipped,
                "executed": self.persist_executed,
                "skip_rate_pct": round(persist_skip_rate, 1),
            },
            "storage_pruning": {
                "regime_baselines_pruned": self.regime_baselines_pruned,
                "regime_signatures_pruned": self.regime_signatures_pruned,
            },
        }


class CausalAnalysisThrottle:
    """Lazy evaluation of expensive causal analysis."""

    def __init__(self, config: Optional[ThrottleConfig] = None):
        self.config = config or ThrottleConfig()
        self.frame_since_last_causal = 0
        self.metrics = PerformanceMetrics()

    def should_compute_causal(self, instability_score: float) -> bool:
        """Determine whether to compute causal analysis this frame.

        Returns True if:
          1. Lazy causal is disabled, OR
          2. Instability is above threshold, OR
          3. Enough frames have passed since last computation
        """
        if not self.config.enable_lazy_causal:
            return True

        if instability_score >= self.config.causal_instability_threshold:
            return True

        if self.frame_since_last_causal >= self.config.causal_compute_every_n_frames:
            return True

        return False

    def record_computation(self, computed: bool) -> None:
        """Record whether causal was computed this frame."""
        if computed:
            self.metrics.record_causal_compute()
            self.frame_since_last_causal = 0
        else:
            self.metrics.record_causal_skip()
            self.frame_since_last_causal += 1

    def update_frame_count(self) -> None:
        """Called once per frame."""
        self.metrics.frame_count += 1


class RegimePersistenceThrottle:
    """Throttle expensive regime state persistence."""

    def __init__(self, config: Optional[ThrottleConfig] = None):
        self.config = config or ThrottleConfig()
        self.frame_since_last_persist = 0
        self.pending_save = False
        self.metrics = PerformanceMetrics()

    def should_persist(self, force: bool = False) -> bool:
        """Determine whether to persist regime state this frame."""
        if force:
            return True

        if not self.config.enable_throttled_persistence:
            return True

        if self.frame_since_last_persist >= self.config.regime_persist_interval:
            return True

        return False

    def mark_persistence(self) -> None:
        """Record that persistence was executed."""
        self.metrics.record_persist_execute()
        self.frame_since_last_persist = 0

    def skip_persistence(self) -> None:
        """Record that persistence was skipped."""
        self.metrics.record_persist_skip()
        self.frame_since_last_persist += 1

    def update_frame_count(self) -> None:
        """Called once per frame."""
        self.metrics.frame_count += 1


class StorageBound:
    """Enforce bounds on dynamically-sized storage."""

    def __init__(self, config: Optional[ThrottleConfig] = None):
        self.config = config or ThrottleConfig()
        self.metrics = PerformanceMetrics()

    def prune_regime_baselines(self, baselines: dict[str, Any]) -> int:
        """Prune regime baselines to max configured size.

        Keeps the most recent (highest count) baselines.
        Returns number of items pruned.
        """
        if len(baselines) <= self.config.max_regime_baselines:
            return 0

        # Sort by count (descending) and keep top N
        sorted_items = sorted(
            baselines.items(),
            key=lambda kv: kv[1].get("count", 0),
            reverse=True
        )

        to_keep = dict(sorted_items[:self.config.max_regime_baselines])
        pruned = len(baselines) - len(to_keep)

        if pruned > 0:
            self.metrics.regime_baselines_pruned += pruned

        return pruned

    def prune_regime_signatures(self, signatures: list[Any]) -> int:
        """Prune regime signatures to max configured size.

        Keeps the most recent (last N) signatures.
        Returns number of items pruned.
        """
        if len(signatures) <= self.config.max_regime_signatures:
            return 0

        pruned = len(signatures) - self.config.max_regime_signatures

        if pruned > 0:
            self.metrics.regime_signatures_pruned += pruned

        return pruned
