"""Production entry point for the Neraium Structural Engine.

This module provides a clean, simple interface for production deployments.
It wraps the StructuralEngine with strict schemas, error handling, and logging.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from neraium_core.alignment import StructuralEngine
from neraium_core.engine.config import ProductionEngineConfig, ProductionLoggingConfig
from neraium_core.engine.logging_setup import get_logger
from neraium_core.engine.schemas import InputFrame, EngineResult, BatchResult


logger = logging.getLogger(__name__)


class ProductionEngine:
    """Production-ready wrapper around StructuralEngine.

    Provides:
    - Strict input/output schema validation
    - Comprehensive error handling
    - Structured logging
    - Safe defaults and recovery
    """

    def __init__(
        self,
        config: ProductionEngineConfig | None = None,
        logging_config: ProductionLoggingConfig | None = None,
    ):
        """Initialize production engine.

        Args:
            config: Engine configuration. If None, loads from environment.
            logging_config: Logging configuration. If None, loads from environment.
        """
        self.config = config or ProductionEngineConfig.from_env()
        self.logging_config = logging_config or ProductionLoggingConfig.from_env()

        # Validate configuration
        try:
            self.config.validate()
        except ValueError as e:
            raise ValueError(f"Invalid engine configuration: {e}") from e

        # Create the underlying structural engine with production-safe defaults
        self._engine = StructuralEngine(
            baseline_window=self.config.baseline_window,
            recent_window=self.config.recent_window,
            # Use safe defaults that won't cause memory issues
            frame_debug=False,  # Never enable debug in production
            drift_smoothing_window=25,  # FD004 standard
            watch_quantile=0.65,
            alert_quantile=0.85,
            watch_persistence=5,
            alert_persistence=3,
        )

        self._logger = get_logger(__name__, self.logging_config)
        self._frame_count_by_unit: dict[str, int] = {}

    def process_frame(self, frame: InputFrame) -> EngineResult:
        """Process a single frame and return result.

        Args:
            frame: Input frame with strict schema

        Returns:
            Engine result with strict schema

        Raises:
            ValueError: If frame is invalid
        """
        # Validate input
        try:
            frame.validate()
        except ValueError as e:
            self._logger.error(f"Invalid frame for unit {frame.unit_id}: {e}")
            raise

        # Track frame count
        if frame.unit_id not in self._frame_count_by_unit:
            self._frame_count_by_unit[frame.unit_id] = 0
        self._frame_count_by_unit[frame.unit_id] += 1
        frame_count = self._frame_count_by_unit[frame.unit_id]

        # Prepare frame for engine (add required legacy fields)
        engine_frame = {
            "timestamp": frame.timestamp,
            "sensor_values": frame.sensors,
            "unit_id": frame.unit_id,
            "site_id": "production",  # Legacy field, required by engine
            "asset_id": frame.unit_id,  # Map unit_id to asset_id
        }

        try:
            # Process through engine
            raw_result = self._engine.process_frame(engine_frame)

            # Extract key metrics from engine result
            state = self._extract_state(raw_result)
            drift_score = self._extract_drift_score(raw_result)
            stability_score = self._compute_stability_score(raw_result)
            health_percentage = self._compute_health(drift_score, stability_score)

            # Create result with strict schema
            result = EngineResult(
                timestamp=frame.timestamp,
                unit_id=frame.unit_id,
                frame_count=frame_count,
                state=state,
                drift_score=drift_score,
                stability_score=stability_score,
                health_percentage=health_percentage,
                baseline_ready=len(self._engine.frames) >= self._engine.baseline_window,
                sensor_count=len(frame.sensors),
                model_age_frames=len(self._engine.frames),
            )

            # Validate output
            result.validate()

            self._logger.info(
                f"Processed frame for unit {frame.unit_id}: "
                f"state={state}, drift={drift_score:.3f}, health={health_percentage}%"
            )

            return result

        except Exception as e:
            self._logger.error(f"Error processing frame for unit {frame.unit_id}: {e}", exc_info=True)
            # Return a safe degraded result instead of crashing
            return self._safe_degraded_result(frame, frame_count)

    def process_batch(self, frames: list[InputFrame]) -> BatchResult:
        """Process a batch of frames.

        Args:
            frames: List of input frames

        Returns:
            Batch result with all successful results and any errors
        """
        start_time = time.time()
        results: list[EngineResult] = []
        errors: list[str] = []

        for i, frame in enumerate(frames):
            try:
                result = self.process_frame(frame)
                results.append(result)
            except Exception as e:
                error_msg = f"Frame {i}: {str(e)}"
                errors.append(error_msg)
                self._logger.warning(f"Skipping frame in batch: {error_msg}")

        processing_time = time.time() - start_time

        batch_result = BatchResult(
            results=results,
            errors=errors,
            processing_time_seconds=processing_time,
        )

        self._logger.info(
            f"Processed batch: {len(results)} successful, {len(errors)} errors, "
            f"time={processing_time:.3f}s"
        )

        return batch_result

    def _extract_state(self, raw_result: dict) -> str:
        """Extract state from raw engine result.

        Safe extraction with fallbacks.
        """
        try:
            state = raw_result.get("state") or raw_result.get("policy_state")
            if state in {"STABLE", "WATCH", "ALERT"}:
                return state
        except Exception:
            pass
        return "STABLE"  # Safe default

    def _extract_drift_score(self, raw_result: dict) -> float:
        """Extract drift score from raw engine result.

        Normalized to [0, 1] range.
        """
        try:
            score = raw_result.get("score") or raw_result.get("drift_score") or 0.0
            score = float(score)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except (ValueError, TypeError):
            return 0.0

    def _compute_stability_score(self, raw_result: dict) -> float:
        """Compute stability score from raw result.

        Normalized to [0, 1] range where 1 is most stable.
        """
        try:
            # Try to get stability-related metrics
            stability = raw_result.get("stability_score")
            if stability is not None:
                return max(0.0, min(1.0, float(stability)))

            # Compute from other available metrics
            # Lower drift = higher stability
            drift = float(raw_result.get("score") or 0.0)
            stability = max(0.0, 1.0 - drift)
            return stability
        except (ValueError, TypeError):
            return 0.5  # Neutral default

    def _compute_health(self, drift_score: float, stability_score: float) -> int:
        """Compute health percentage from drift and stability.

        Returns:
            Health percentage [0, 100]
        """
        health = 100.0 - (drift_score * 50.0)  # Drift reduces health
        health += (stability_score * 20.0)  # Stability improves health
        return int(round(max(0.0, min(100.0, health))))

    def _safe_degraded_result(self, frame: InputFrame, frame_count: int) -> EngineResult:
        """Return a safe degraded result when processing fails.

        Prevents crashes and allows graceful degradation.
        """
        return EngineResult(
            timestamp=frame.timestamp,
            unit_id=frame.unit_id,
            frame_count=frame_count,
            state="STABLE",  # Conservative default
            drift_score=0.0,
            stability_score=0.5,  # Unknown
            health_percentage=50,  # Unknown
            baseline_ready=False,
            sensor_count=len(frame.sensors),
            model_age_frames=0,
        )

    def get_diagnostics(self) -> dict:
        """Get engine diagnostics for monitoring."""
        return {
            "config": self.config.to_dict(),
            "logging_config": self.logging_config.to_dict(),
            "units_tracked": len(self._frame_count_by_unit),
            "frame_counts_by_unit": dict(self._frame_count_by_unit),
            "engine_frames_stored": len(self._engine.frames),
        }


__all__ = [
    "ProductionEngine",
    "InputFrame",
    "EngineResult",
    "BatchResult",
    "ProductionEngineConfig",
    "ProductionLoggingConfig",
]
