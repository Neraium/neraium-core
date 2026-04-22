"""Webhook output sink adapter."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .base import OutputSink, Result

logger = logging.getLogger(__name__)


class WebhookSink(OutputSink):
    """Send results to a webhook URL via HTTP POST.

    Config parameters:
    - url (str, required): Webhook endpoint URL
    - timeout_sec (int, default=10): Request timeout
    - verify_ssl (bool, default=True): Verify SSL certificates
    - headers (dict, default={}): Custom HTTP headers
    - batch_size (int, default=1): Number of results to batch in one request
    - retry_count (int, default=3): Number of retries on failure
    - retry_delay_ms (int, default=1000): Delay between retries

    Each POST body is JSON:
    {
        "timestamp": float,
        "unit_id": str,
        "sensors": {...},
        "output": {...}
    }

    For batch mode (batch_size > 1), the body is an array of objects.
    """

    def __init__(self, sink_config: dict[str, Any] | None = None) -> None:
        super().__init__(sink_config)
        try:
            import requests
        except ImportError:
            raise RuntimeError("Webhook sink requires 'requests' package: pip install requests")

        self.url = self.sink_config.get("url")
        self.timeout_sec = self.sink_config.get("timeout_sec", 10)
        self.verify_ssl = self.sink_config.get("verify_ssl", True)
        self.headers = self.sink_config.get("headers", {})
        self.batch_size = self.sink_config.get("batch_size", 1)
        self.retry_count = self.sink_config.get("retry_count", 3)
        self.retry_delay_ms = self.sink_config.get("retry_delay_ms", 1000)

        if not self.url:
            raise ValueError("Webhook URL is required")

        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

        self.requests = requests
        self.session = requests.Session()
        self.batch_buffer: list[dict[str, Any]] = []

    def write_result(self, result: Result) -> None:
        """Buffer result and potentially send."""
        result_dict = {
            "timestamp": result.frame.timestamp,
            "unit_id": result.frame.unit_id,
            "sensors": result.frame.sensors,
            "output": result.output,
        }
        if result.metadata:
            result_dict["metadata"] = result.metadata

        self.batch_buffer.append(result_dict)
        self.result_count += 1

        # Send if batch is full
        if len(self.batch_buffer) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Send buffered results to webhook."""
        if not self.batch_buffer:
            return

        # Prepare payload
        if self.batch_size == 1 and len(self.batch_buffer) == 1:
            payload = self.batch_buffer[0]
        else:
            payload = self.batch_buffer

        payload_json = json.dumps(payload, default=str)

        # Retry logic
        for attempt in range(self.retry_count):
            try:
                resp = self.session.post(
                    self.url,
                    data=payload_json,
                    headers=self.headers,
                    timeout=self.timeout_sec,
                    verify=self.verify_ssl,
                )
                resp.raise_for_status()
                logger.debug(f"Webhook POST successful ({len(self.batch_buffer)} items)")
                self.batch_buffer = []
                return

            except self.requests.RequestException as e:
                if attempt < self.retry_count - 1:
                    delay = (self.retry_delay_ms * (2 ** attempt)) / 1000.0
                    logger.warning(f"Webhook POST failed (attempt {attempt + 1}/{self.retry_count}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Webhook POST failed after {self.retry_count} attempts: {e}")
                    raise RuntimeError(f"Webhook send failed: {e}")

    def finalize(self) -> dict[str, Any]:
        """Flush remaining results and close session."""
        try:
            self._flush_batch()
            self.session.close()
            logger.info(f"Sent {self.result_count} results to webhook {self.url}")
            return {
                "webhook_url": self.url,
                "record_count": self.result_count,
                "format": "webhook",
            }
        except Exception as e:
            logger.error(f"Webhook finalization failed: {e}")
            raise RuntimeError(f"Webhook finalize failed: {e}")

    def close(self) -> None:
        """Close session."""
        try:
            self._flush_batch()
            self.session.close()
        except Exception as e:
            logger.warning(f"Error closing webhook session: {e}")
