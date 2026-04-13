"""Unified Engine: canonical entrypoint for all Neraium use cases.

This module provides the primary user-facing interface:
- Live ingestion with ProductionEngine-style API
- Replay/validation with deterministic results
- Shadow mode evidence generation
- Consistent output across all modes

All three modes (live, replay, evidence) flow through the same engine
with the same output schema.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

from neraium_core.engine.production import ProductionEngine, InputFrame, EngineResult, BatchResult
from neraium_core.engine.config import ProductionEngineConfig, ProductionLoggingConfig
from neraium_core.engine.dataset_loader import load_dataset


logger = logging.getLogger(__name__)


class Engine:
    """Unified Neraium engine: one interface for all use cases.

    Provides:
    - Live frame ingestion (ProductionEngine-style)
    - Batch replay (from datasets or frame lists)
    - Shadow mode evidence generation
    - Consistent EngineResult output across all modes

    Usage:

        # Live ingestion (recommended for production)
        engine = Engine()
        result = engine.ingest_frame(
            timestamp=1704067200.0,
            unit_id="pump_001",
            sensors={"temp": 65.3, "vibration": 0.12}
        )

        # Batch replay (validation, benchmarking)
        engine = Engine()
        engine.replay(dataset_path="FD004.csv", dataset_type="fd004")
        metrics = engine.get_summary()
        evidence = engine.get_evidence_report()  # If shadow mode enabled

        # Check operational readiness
        readiness = engine.get_readiness()
    """

    def __init__(
        self,
        config: ProductionEngineConfig | None = None,
        logging_config: ProductionLoggingConfig | None = None,
        baseline_window: int | None = None,
        recent_window: int | None = None,
        enable_shadow_mode: bool = False,
    ):
        """Initialize unified engine.

        Args:
            config: Engine configuration (uses env if None)
            logging_config: Logging configuration (uses env if None)
            baseline_window: Override baseline window size
            recent_window: Override recent window size
            enable_shadow_mode: Enable evidence collection for validation

        Note:
            For live production use, use default config from environment.
            For benchmarking/validation, explicitly set enable_shadow_mode=True.
        """
        self.config = config or ProductionEngineConfig.from_env()
        self.logging_config = logging_config or ProductionLoggingConfig.from_env()
        self.enable_shadow_mode = enable_shadow_mode

        # Override window sizes if provided
        if baseline_window is not None:
            self.config.baseline_window = baseline_window
        if recent_window is not None:
            self.config.recent_window = recent_window

        # Create the underlying ProductionEngine
        # (which wraps StructuralEngine with production safety)
        self._production_engine = ProductionEngine(self.config, self.logging_config)

        # Access to internal structural engine for advanced use cases
        self._structural_engine = self._production_engine._engine

        # Shadow mode state (for validation/evidence)
        self._shadow_mode_evidence: list[dict] = []
        self._replay_results: list[dict] = []

        self._logger = logging.getLogger(__name__)

    def _reset_runtime_state(self) -> None:
        """Reset runtime state to ensure replay isolation between datasets."""
        self._production_engine = ProductionEngine(self.config, self.logging_config)
        self._structural_engine = self._production_engine._engine
        self._replay_results = []
        self._shadow_mode_evidence = []

    def ingest_frame(
        self,
        timestamp: float,
        unit_id: str,
        sensors: dict[str, float],
    ) -> EngineResult:
        """Process a single live frame (production use case).

        Args:
            timestamp: Unix timestamp (seconds)
            unit_id: Equipment identifier
            sensors: Sensor readings {name: value}

        Returns:
            EngineResult with state, drift, health metrics

        Raises:
            ValueError: If frame is invalid

        Note:
            This is the recommended interface for live ingestion.
            Frames are normalized via ProductionEngine ingestion adapter.
        """
        frame = InputFrame(timestamp=timestamp, unit_id=unit_id, sensors=sensors)
        result = self._production_engine.process_frame(frame)

        # Collect evidence if shadow mode enabled
        if self.enable_shadow_mode:
            self._shadow_mode_evidence.append(result.to_dict())

        return result

    def ingest_batch(self, frames: list[InputFrame]) -> BatchResult:
        """Process a batch of frames.

        Args:
            frames: List of InputFrame objects

        Returns:
            BatchResult with results and any errors
        """
        return self._production_engine.process_batch(frames)

    def replay(
        self,
        dataset_path: Path | str,
        dataset_type: str,
        unit_filter: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Replay a dataset through the engine (validation/benchmarking use case).

        Args:
            dataset_path: Path to CSV dataset (FD004, IMS, etc.)
            dataset_type: Dataset type ('fd004', 'ims', 'generic')
            unit_filter: Optional unit ID(s) to process (default: all units)

        Returns:
            Replay summary with overall metrics

        Raises:
            ValueError: If dataset not found or invalid
            FileNotFoundError: If dataset_path doesn't exist
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        self._logger.info(f"Replaying dataset: {dataset_path} (type={dataset_type})")

        # Load dataset with canonical frame normalization
        df = load_dataset(dataset_path, dataset_type)
        if df is None:
            raise ValueError(f"Failed to load dataset: {dataset_path}")

        self._logger.info(f"Loaded {len(df)} records from {df['asset_id'].nunique()} units")

        # Filter units if requested
        if unit_filter is not None:
            if isinstance(unit_filter, str):
                unit_filter = [unit_filter]
            df = df[df["asset_id"].isin(unit_filter)]
            self._logger.info(f"Filtered to {len(df)} records for units: {unit_filter}")

        # Replay uses an isolated runtime state for deterministic dataset-level metrics
        self._reset_runtime_state()

        # Replay: iterate through frames in order
        for _, row in df.iterrows():
            try:
                # Convert row to engine frame format
                # (loaders now output canonical format with asset_id, site_id, timestamp)
                result = self._structural_engine.process_frame(
                    {
                        "timestamp": row["timestamp"],
                        "asset_id": row["asset_id"],
                        "site_id": row.get("site_id", "default-site"),
                        "sensor_values": self._extract_sensors_from_row(row),
                        # Include reference metrics for evidence
                        "structural_drift_score": row.get("structural_drift_score", 0.0),
                    }
                )
                self._replay_results.append(result)

                if self.enable_shadow_mode:
                    self._shadow_mode_evidence.append({
                        "timestamp": row["timestamp"],
                        "asset_id": row["asset_id"],
                        "result": result,
                    })

            except Exception as e:
                self._logger.warning(f"Error processing row: {e}")
                continue

        self._logger.info(f"Replay complete: {len(self._replay_results)} results")

        return {
            "dataset": str(dataset_path),
            "dataset_type": dataset_type,
            "records_loaded": len(df),
            "records_processed": len(self._replay_results),
            "units": df["asset_id"].nunique() if len(df) > 0 else 0,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get summary metrics from latest replay/batch.

        Returns:
            Summary stats (avg drift, state distribution, etc.)
        """
        if not self._replay_results:
            return {"error": "No replay data available"}

        results = self._replay_results
        states = [r.get("state", "STABLE") for r in results]
        drifts = [float(r.get("structural_drift_score", 0.0)) for r in results]

        return {
            "total_frames": len(results),
            "avg_drift": sum(drifts) / len(drifts) if drifts else 0.0,
            "max_drift": max(drifts) if drifts else 0.0,
            "min_drift": min(drifts) if drifts else 0.0,
            "state_distribution": {
                state: states.count(state)
                for state in set(states)
            },
            "alert_count": states.count("ALERT"),
            "watch_count": states.count("WATCH"),
            "stable_count": states.count("STABLE"),
        }

    def get_evidence_report(self) -> dict[str, Any]:
        """Get shadow mode evidence and replay comparison.

        Returns:
            Evidence report (if shadow_mode enabled)

        Note:
            Evidence only collected if enable_shadow_mode=True.
        """
        if not self.enable_shadow_mode:
            return {"error": "Shadow mode not enabled"}

        if not self._shadow_mode_evidence:
            return {"error": "No evidence data available"}

        return {
            "evidence_count": len(self._shadow_mode_evidence),
            "evidence": self._shadow_mode_evidence[:100],  # Return first 100 for summary
            "summary": self.get_summary(),
        }

    def get_readiness(self) -> dict[str, Any]:
        """Check operational readiness.

        Returns:
            Readiness status with baseline window and sensor info
        """
        frames_collected = len(self._structural_engine.frames)
        baseline_ready = frames_collected >= self._structural_engine.baseline_window

        return {
            "baseline_ready": baseline_ready,
            "frames_collected": frames_collected,
            "baseline_window": self._structural_engine.baseline_window,
            "sensors_detected": len(self._structural_engine.sensor_order),
            "sensor_names": list(self._structural_engine.sensor_order),
            "units_tracked": len(self._production_engine._frame_count_by_unit),
        }

    def get_diagnostics(self) -> dict[str, Any]:
        """Get engine diagnostics for monitoring/debugging.

        Returns:
            Diagnostic information
        """
        return {
            "config": self._production_engine.config.to_dict(),
            "logging_config": self._production_engine.logging_config.to_dict(),
            "readiness": self.get_readiness(),
            "production_diagnostics": self._production_engine.get_diagnostics(),
        }

    @staticmethod
    def _extract_sensors_from_row(row: dict) -> dict[str, float]:
        """Extract numeric sensor values from a data row.

        Args:
            row: DataFrame row (dict-like)

        Returns:
            Dictionary of sensor_name: value
        """
        # Skip metadata/reference columns
        skip_cols = {
            "timestamp", "asset_id", "site_id", "unit", "cycle",
            "structural_drift_score", "true_rul", "failure_cycle", "lead_time"
        }

        sensors = {}
        for key, value in row.items():
            if key not in skip_cols and isinstance(value, (int, float)):
                try:
                    sensors[key] = float(value)
                except (TypeError, ValueError):
                    pass

        return sensors


__all__ = ["Engine"]
