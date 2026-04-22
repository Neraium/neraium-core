"""Kafka data source adapter."""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import DataSourceAdapter, Frame

logger = logging.getLogger(__name__)


class KafkaDataSource(DataSourceAdapter):
    """Read frames from Kafka topic.

    Config parameters:
    - brokers (list[str], required): List of broker addresses (e.g., ["localhost:9092"])
    - topic (str, required): Kafka topic to consume from
    - group_id (str, default="neraium-runner"): Consumer group ID
    - max_records (int, default=None): Max records to read (None = unlimited)
    - timeout_ms (int, default=10000): Timeout for poll operations
    - auto_offset_reset (str, default="earliest"): Where to start reading

    Message format (JSON):
    {
        "timestamp": float,
        "unit_id": str,
        "sensors": {"sensor_name": float, ...}
    }
    """

    def __init__(self, source_config: dict[str, Any] | None = None) -> None:
        super().__init__(source_config)
        try:
            from kafka import KafkaConsumer
        except ImportError:
            raise RuntimeError("Kafka support requires 'kafka-python' package: pip install kafka-python")

        self.brokers = self.source_config.get("brokers", ["localhost:9092"])
        self.topic = self.source_config.get("topic")
        self.group_id = self.source_config.get("group_id", "neraium-runner")
        self.max_records = self.source_config.get("max_records")
        self.timeout_ms = self.source_config.get("timeout_ms", 10000)
        self.auto_offset_reset = self.source_config.get("auto_offset_reset", "earliest")

        if not self.topic:
            raise ValueError("Kafka topic is required")

        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.brokers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=self.timeout_ms,
        )

    def validate(self) -> bool:
        """Validate Kafka connection."""
        try:
            self.consumer.topics()
            if self.topic not in self.consumer.topics():
                logger.error(f"Kafka topic '{self.topic}' not found")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to validate Kafka connection: {e}")
            return False

    def read_frames(self):
        """Yield frames from Kafka."""
        if not self.validate():
            raise RuntimeError(f"Invalid Kafka source: topic {self.topic}")

        count = 0
        try:
            for message in self.consumer:
                if self.max_records and count >= self.max_records:
                    break

                try:
                    data = message.value
                    timestamp = float(data["timestamp"])
                    unit_id = str(data["unit_id"]).strip()
                    sensors = {k: float(v) for k, v in data.get("sensors", {}).items()}

                    if not sensors:
                        logger.warning(f"Message has no sensors: {data}")
                        continue

                    metadata = {k: v for k, v in data.items() if k not in {"timestamp", "unit_id", "sensors"}}

                    yield Frame(
                        timestamp=timestamp,
                        unit_id=unit_id,
                        sensors=sensors,
                        metadata=metadata if metadata else None,
                    )
                    count += 1

                except (KeyError, ValueError, TypeError) as e:
                    logger.error(f"Failed to parse Kafka message: {e}")
                    raise RuntimeError(f"Kafka message parsing error: {e}")

        finally:
            self.consumer.close()

    def close(self) -> None:
        """Close Kafka consumer."""
        try:
            self.consumer.close()
        except Exception as e:
            logger.warning(f"Error closing Kafka consumer: {e}")
