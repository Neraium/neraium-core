"""Database output sink adapter."""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import OutputSink, Result

logger = logging.getLogger(__name__)


class DatabaseSink(OutputSink):
    """Write results to a SQL database (requires SQLAlchemy).

    Config parameters:
    - connection_string (str, required): SQLAlchemy connection string
      Examples:
        - "sqlite:///neraium_results.db"
        - "postgresql://user:pass@localhost/neraium"
        - "mysql+pymysql://user:pass@localhost/neraium"
    - table_name (str, default="neraium_results"): Table name
    - batch_size (int, default=100): Batch insert size
    - json_columns (bool, default=True): Store output as JSON (if False, tries to flatten)

    Table will be auto-created if it doesn't exist with schema:
    - id (auto-increment primary key)
    - timestamp (float)
    - unit_id (string)
    - sensors (JSON)
    - output (JSON)
    - created_at (timestamp)
    """

    def __init__(self, sink_config: dict[str, Any] | None = None) -> None:
        super().__init__(sink_config)
        try:
            from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, func, text
            from sqlalchemy.orm import Session, declarative_base
            from sqlalchemy.types import JSON
        except ImportError:
            raise RuntimeError("Database sink requires 'sqlalchemy' package: pip install sqlalchemy")

        self.connection_string = self.sink_config.get("connection_string")
        self.table_name = self.sink_config.get("table_name", "neraium_results")
        self.batch_size = self.sink_config.get("batch_size", 100)
        self.json_columns = self.sink_config.get("json_columns", True)

        if not self.connection_string:
            raise ValueError("Database connection_string is required")

        self.engine = create_engine(self.connection_string)
        self.Session = Session
        self.batch_buffer: list[dict[str, Any]] = []

        # Define table schema
        Base = declarative_base()

        class NeraiumResult(Base):
            __tablename__ = self.table_name

            id = Column(Integer, primary_key=True)
            timestamp = Column(Float, nullable=False, index=True)
            unit_id = Column(String(255), nullable=False, index=True)
            sensors = Column(JSON if self.json_columns else String(10000), nullable=True)
            output = Column(JSON if self.json_columns else String(10000), nullable=True)
            created_at = Column(DateTime, nullable=False, default=func.now())

        self.NeraiumResult = NeraiumResult
        Base.metadata.create_all(self.engine)
        logger.info(f"Database table '{self.table_name}' ready")

    def write_result(self, result: Result) -> None:
        """Buffer result for batch insert."""
        result_dict = {
            "timestamp": result.frame.timestamp,
            "unit_id": result.frame.unit_id,
            "sensors": json.dumps(result.frame.sensors) if not self.json_columns else result.frame.sensors,
            "output": json.dumps(result.output, default=str) if not self.json_columns else result.output,
        }
        self.batch_buffer.append(result_dict)
        self.result_count += 1

        # Flush batch if full
        if len(self.batch_buffer) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Write buffered results to database."""
        if not self.batch_buffer:
            return

        try:
            with self.Session(self.engine) as session:
                for row in self.batch_buffer:
                    obj = self.NeraiumResult(**row)
                    session.add(obj)
                session.commit()
                logger.debug(f"Flushed {len(self.batch_buffer)} results to database")
        except Exception as e:
            logger.error(f"Database batch insert failed: {e}")
            raise RuntimeError(f"Database write failed: {e}")
        finally:
            self.batch_buffer = []

    def finalize(self) -> dict[str, Any]:
        """Flush remaining results and close connection."""
        try:
            self._flush_batch()

            # Get final count from database
            with self.Session(self.engine) as session:
                count = session.query(self.NeraiumResult).count()

            self.engine.dispose()
            logger.info(f"Wrote {count} results to database table '{self.table_name}'")
            return {
                "connection_string": self.connection_string,
                "table_name": self.table_name,
                "record_count": count,
                "format": "database",
            }
        except Exception as e:
            logger.error(f"Database finalization failed: {e}")
            raise RuntimeError(f"Database finalize failed: {e}")

    def close(self) -> None:
        """Close database connection."""
        try:
            self._flush_batch()
            self.engine.dispose()
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")
