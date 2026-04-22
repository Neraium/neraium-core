"""Kafka output sink adapter."""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import OutputSink, Result

logger = logging.getLogger(__name__)


class KafkaSink(OutputSink):
    """Write results to Kafka topic.

    Config parameters:
    - brokers (list[str], required): List of broker addresses (e.g., ["localhost:9092"])
    - topic (str, required): Topic to produce results to
    - key_field (str, default="unit_id"): Which field to use as message key

    Each result is serialized as JSON and sent to Kafka.
    """

    def __init__(self, sink_config: dict[str, Any] | None = None) -> None:
        super().__init__(sink_config)
        try:
            from kafka import KafkaProducer
        except ImportError:
            raise RuntimeError("Kafka support requires 'kafka-python' package: pip install kafka-python")

        self.brokers = self.sink_config.get("brokers", ["localhost:9092"])
        self.topic = self.sink_config.get("topic")
        self.key_field = self.sink_config.get("key_field", "unit_id")

        if not self.topic:
            raise ValueError("Kafka topic is required")

        self.producer = KafkaProducer(
            bootstrap_servers=self.brokers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )

    def write_result(self, result: Result) -> None:
        """Send result to Kafka."""
        result_dict = {
            "timestamp": result.frame.timestamp,
            "unit_id": result.frame.unit_id,
            "sensors": result.frame.sensors,
            "output": result.output,
        }
        if result.metadata:
            result_dict["metadata"] = result.metadata

        # Use unit_id or configured key_field as message key
        key = str(result.frame.unit_id).encode("utf-8")

        try:
            self.producer.send(self.topic, value=result_dict, key=key)
            self.result_count += 1
        except Exception as e:
            logger.error(f"Failed to send message to Kafka: {e}")
            raise RuntimeError(f"Kafka send failed: {e}")

    def finalize(self) -> dict[str, Any]:
        """Flush and close Kafka producer."""
        try:
            self.producer.flush()
            self.producer.close()
            logger.info(f"Sent {self.result_count} results to Kafka topic '{self.topic}'")
            return {
                "output_topic": self.topic,
                "record_count": self.result_count,
                "format": "kafka",
            }
        except Exception as e:
            logger.error(f"Failed to finalize Kafka sink: {e}")
            raise RuntimeError(f"Kafka finalize failed: {e}")

    def close(self) -> None:
        """Close Kafka producer."""
        try:
            self.producer.close()
        except Exception as e:
            logger.warning(f"Error closing Kafka producer: {e}")
