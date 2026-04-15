"""JSON output sink adapters."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .base import OutputSink, Result

logger = logging.getLogger(__name__)


class JsonFileSink(OutputSink):
    """Write results to a single JSON file (array of results)."""

    def __init__(self, sink_config: dict[str, Any] | None = None) -> None:
        super().__init__(sink_config)
        self.path = Path(self.sink_config.get("path", "results.json"))
        self.results: list[dict[str, Any]] = []
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """Create parent directory if it doesn't exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_result(self, result: Result) -> None:
        """Buffer result in memory."""
        result_dict = {
            "timestamp": result.frame.timestamp,
            "unit_id": result.frame.unit_id,
            "sensors": result.frame.sensors,
            "output": result.output,
        }
        if result.metadata:
            result_dict["metadata"] = result.metadata
        self.results.append(result_dict)
        self.result_count += 1

    def finalize(self) -> dict[str, Any]:
        """Write all buffered results to JSON file."""
        try:
            with open(self.path, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Wrote {self.result_count} results to {self.path}")
            return {
                "output_path": str(self.path),
                "record_count": self.result_count,
                "format": "json_array",
            }
        except Exception as e:
            logger.error(f"Failed to write JSON file {self.path}: {e}")
            raise RuntimeError(f"JSON write failed: {e}")


class JsonLineSink(OutputSink):
    """Write results to a JSONL file (one JSON object per line)."""

    def __init__(self, sink_config: dict[str, Any] | None = None) -> None:
        super().__init__(sink_config)
        self.path = Path(self.sink_config.get("path", "results.jsonl"))
        self._file_handle = None
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """Create parent directory if it doesn't exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _open_file(self) -> None:
        """Open file for writing (lazy)."""
        if self._file_handle is None:
            self._file_handle = open(self.path, "w")

    def write_result(self, result: Result) -> None:
        """Write result as a single JSON line."""
        self._open_file()
        result_dict = {
            "timestamp": result.frame.timestamp,
            "unit_id": result.frame.unit_id,
            "sensors": result.frame.sensors,
            "output": result.output,
        }
        if result.metadata:
            result_dict["metadata"] = result.metadata

        try:
            line = json.dumps(result_dict, default=str)
            self._file_handle.write(line + "\n")
            self._file_handle.flush()
            self.result_count += 1
        except Exception as e:
            logger.error(f"Failed to write JSONL line: {e}")
            raise RuntimeError(f"JSONL write failed: {e}")

    def finalize(self) -> dict[str, Any]:
        """Close file and return summary."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        logger.info(f"Wrote {self.result_count} results to {self.path}")
        return {
            "output_path": str(self.path),
            "record_count": self.result_count,
            "format": "jsonl",
        }

    def close(self) -> None:
        """Ensure file is closed."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
