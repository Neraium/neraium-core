"""Core agnostic runner orchestration."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Optional

from neraium_core.system_intelligence.platform import StructuralSystemIntelligencePlatform

from .adapters.base import DataSourceAdapter, OutputSink, RunContext
from .config import EngineConfig

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """Metrics from a run."""
    run_id: str
    status: str  # 'success', 'partial', 'failed'
    frame_count: int
    error_count: int
    elapsed_seconds: float
    frames_per_second: float
    source_type: str
    sink_type: str
    engine_mode: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AgnosticRunner:
    """Orchestrate data flow through Neraium engine with pluggable adapters."""

    def __init__(
        self,
        data_source: DataSourceAdapter,
        output_sink: OutputSink,
        engine_config: EngineConfig | None = None,
        run_id: str | None = None,
        error_handler: str = "log",  # 'log', 'raise', 'skip'
    ) -> None:
        """Initialize runner.

        Args:
            data_source: Source for input frames
            output_sink: Destination for results
            engine_config: Engine configuration (default: production mode)
            run_id: Unique run identifier (auto-generated if not provided)
            error_handler: How to handle frame processing errors
        """
        self.data_source = data_source
        self.output_sink = output_sink
        self.engine_config = engine_config or EngineConfig.production()
        self.run_id = run_id or str(uuid.uuid4())
        self.error_handler = error_handler

        # Initialize engine
        self.platform = StructuralSystemIntelligencePlatform(operating_mode=self.engine_config.mode)
        self._configure_engine_components()

        # Metrics
        self.frame_count = 0
        self.error_count = 0
        self.start_time: float | None = None
        self.end_time: float | None = None

    def _configure_engine_components(self) -> None:
        """Configure engine based on component flags."""
        # This is a simplified approach - in production, you might use
        # the NeraiumModel pattern from evaluation/runner.py
        # For now, we just run the full platform
        pass

    def run(self) -> RunMetrics:
        """Execute the runner end-to-end.

        Returns:
            RunMetrics with execution details

        Raises:
            RuntimeError: On unrecoverable errors (if error_handler='raise')
        """
        logger.info(f"Starting run {self.run_id} in {self.engine_config.mode} mode")
        self.start_time = time.time()

        try:
            # Validate inputs
            if not self.data_source.validate():
                raise RuntimeError("Data source validation failed")
            if not self.output_sink.validate():
                raise RuntimeError("Output sink validation failed")

            # Create run context
            run_context = RunContext(
                run_id=self.run_id,
                run_mode=self.engine_config.mode,
                start_time=self.start_time,
                data_source_type=type(self.data_source).__name__,
                output_sink_type=type(self.output_sink).__name__,
            )

            # Process frames
            for frame in self.data_source.read_frames():
                try:
                    # Process frame through engine
                    engine_input = frame.to_dict()
                    engine_output = self.platform.update(engine_input)

                    # Write result
                    from .adapters.base import Result
                    result = Result(frame=frame, output=engine_output, metadata=run_context.__dict__)
                    self.output_sink.write_result(result)

                    self.frame_count += 1

                except Exception as e:
                    self.error_count += 1
                    error_msg = f"Error processing frame {self.frame_count}: {e}"

                    if self.error_handler == "raise":
                        logger.error(error_msg)
                        raise
                    elif self.error_handler == "log":
                        logger.warning(error_msg)
                    elif self.error_handler == "skip":
                        logger.debug(error_msg)
                    # else: silent skip

            # Finalize
            self.end_time = time.time()
            sink_summary = self.output_sink.finalize()

            logger.info(
                f"Run {self.run_id} completed: {self.frame_count} frames, "
                f"{self.error_count} errors in {self.end_time - self.start_time:.2f}s"
            )

            # Build metrics
            elapsed = max(self.end_time - self.start_time, 1e-9)
            fps = self.frame_count / elapsed
            if self.error_count == 0:
                status = "success"
            elif self.frame_count == 0:
                status = "failed"
            else:
                status = "partial"

            return RunMetrics(
                run_id=self.run_id,
                status=status,
                frame_count=self.frame_count,
                error_count=self.error_count,
                elapsed_seconds=elapsed,
                frames_per_second=fps,
                source_type=type(self.data_source).__name__,
                sink_type=type(self.output_sink).__name__,
                engine_mode=self.engine_config.mode,
                summary=sink_summary,
            )

        except Exception as e:
            self.end_time = time.time()
            logger.error(f"Run {self.run_id} failed: {e}")

            # Return failure metrics
            elapsed = self.end_time - self.start_time
            return RunMetrics(
                run_id=self.run_id,
                status="failed",
                frame_count=self.frame_count,
                error_count=self.error_count + 1,
                elapsed_seconds=elapsed,
                frames_per_second=0,
                source_type=type(self.data_source).__name__,
                sink_type=type(self.output_sink).__name__,
                engine_mode=self.engine_config.mode,
                summary={"error": str(e)},
            )

        finally:
            # Clean up
            self.data_source.close()
            self.output_sink.close()
