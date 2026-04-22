"""CSV data source adapter."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from .base import DataSourceAdapter, Frame

logger = logging.getLogger(__name__)


class CsvDataSource(DataSourceAdapter):
    """Read frames from CSV file.

    Expected CSV columns:
    - timestamp (float, required)
    - unit_id or site_id (string, required)
    - sensor_* columns (float, required - at least one)

    Optional columns:
    - Any other columns become metadata

    Config parameters:
    - path (str, required): Path to CSV file
    - timestamp_col (str, default="timestamp"): Column name for timestamp
    - unit_id_col (str, default="unit_id"): Column name for unit/site ID
    - sensor_prefix (str, default="sensor_"): Prefix for sensor columns
    - skip_rows (int, default=0): Rows to skip before header
    """

    def __init__(self, source_config: dict[str, Any] | None = None) -> None:
        super().__init__(source_config)
        self.path = Path(self.source_config.get("path", ""))
        self.timestamp_col = self.source_config.get("timestamp_col", "timestamp")
        self.unit_id_col = self.source_config.get("unit_id_col", "unit_id")
        if not self.unit_id_col and "site_id" in self.source_config:
            self.unit_id_col = self.source_config.get("site_id_col", "site_id")
        self.sensor_prefix = self.source_config.get("sensor_prefix", "sensor_")
        self.skip_rows = self.source_config.get("skip_rows", 0)

    def validate(self) -> bool:
        """Check that CSV file exists and is readable."""
        if not self.path.exists():
            logger.error(f"CSV file not found: {self.path}")
            return False
        if not self.path.is_file():
            logger.error(f"Path is not a file: {self.path}")
            return False
        return True

    def read_frames(self):
        """Yield frames from CSV."""
        if not self.validate():
            raise RuntimeError(f"Invalid CSV source: {self.path}")

        with open(self.path, "r") as f:
            # Skip rows if specified
            for _ in range(self.skip_rows):
                next(f)

            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RuntimeError(f"CSV file {self.path} is empty or has no header")

            for row_num, row in enumerate(reader, start=self.skip_rows + 1):
                try:
                    # Extract required fields
                    timestamp = float(row[self.timestamp_col])
                    unit_id = str(row[self.unit_id_col]).strip()

                    # Extract sensor data
                    sensors = {}
                    metadata = {}

                    for col_name, col_value in row.items():
                        if col_name.startswith(self.sensor_prefix):
                            sensor_name = col_name[len(self.sensor_prefix):]
                            try:
                                sensors[sensor_name] = float(col_value)
                            except (ValueError, TypeError):
                                logger.warning(f"Row {row_num}: Could not parse sensor {col_name}={col_value}")
                        elif col_name not in {self.timestamp_col, self.unit_id_col}:
                            # Non-sensor columns go to metadata
                            try:
                                metadata[col_name] = float(col_value)
                            except (ValueError, TypeError):
                                metadata[col_name] = col_value

                    if not sensors:
                        logger.warning(f"Row {row_num}: No sensor data found (looking for prefix '{self.sensor_prefix}')")
                        continue

                    yield Frame(
                        timestamp=timestamp,
                        unit_id=unit_id,
                        sensors=sensors,
                        metadata=metadata if metadata else None,
                    )

                except (KeyError, ValueError, TypeError) as e:
                    logger.error(f"Row {row_num}: Failed to parse frame: {e}")
                    raise RuntimeError(f"CSV parsing error at row {row_num}: {e}")
