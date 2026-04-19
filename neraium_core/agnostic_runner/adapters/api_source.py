"""API streaming data source adapter."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .base import DataSourceAdapter, Frame

logger = logging.getLogger(__name__)


class ApiStreamSource(DataSourceAdapter):
    """Poll frames from a REST API endpoint.

    Config parameters:
    - url (str, required): API endpoint URL
    - poll_interval_ms (int, default=1000): How often to poll (ms)
    - max_polls (int, default=None): Max number of polls (None = unlimited)
    - timeout_sec (int, default=10): Request timeout
    - verify_ssl (bool, default=True): Verify SSL certificates
    - headers (dict, default={}): Custom HTTP headers
    - response_key (str, default="data"): Key in response containing frames

    Response format expected:
    {
        "data": [
            {
                "timestamp": float,
                "unit_id": str,
                "sensors": {"sensor_name": float, ...}
            },
            ...
        ]
    }

    Or single frame:
    {
        "timestamp": float,
        "unit_id": str,
        "sensors": {"sensor_name": float, ...}
    }
    """

    def __init__(self, source_config: dict[str, Any] | None = None) -> None:
        super().__init__(source_config)
        try:
            import requests
        except ImportError:
            raise RuntimeError("API source requires 'requests' package: pip install requests")

        self.url = self.source_config.get("url")
        self.poll_interval_ms = self.source_config.get("poll_interval_ms", 1000)
        self.max_polls = self.source_config.get("max_polls")
        self.timeout_sec = self.source_config.get("timeout_sec", 10)
        self.verify_ssl = self.source_config.get("verify_ssl", True)
        self.headers = self.source_config.get("headers", {})
        self.response_key = self.source_config.get("response_key", "data")

        if not self.url:
            raise ValueError("API URL is required")

        self.requests = requests
        self.session = requests.Session()

    def validate(self) -> bool:
        """Test API connectivity."""
        try:
            resp = self.session.get(
                self.url,
                timeout=self.timeout_sec,
                verify=self.verify_ssl,
                headers=self.headers,
            )
            if resp.status_code >= 400:
                logger.error(f"API returned status {resp.status_code}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to connect to API: {e}")
            return False

    def read_frames(self):
        """Poll API and yield frames."""
        if not self.validate():
            raise RuntimeError(f"Invalid API source: {self.url}")

        poll_count = 0
        try:
            while True:
                if self.max_polls and poll_count >= self.max_polls:
                    break

                try:
                    resp = self.session.get(
                        self.url,
                        timeout=self.timeout_sec,
                        verify=self.verify_ssl,
                        headers=self.headers,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    # Handle both array and single-object responses
                    frames_data = data
                    if isinstance(data, dict):
                        if self.response_key in data:
                            frames_data = data[self.response_key]
                        elif isinstance(data, dict) and "timestamp" in data:
                            frames_data = [data]  # Single frame
                        else:
                            frames_data = []

                    if not isinstance(frames_data, list):
                        frames_data = [frames_data]

                    # Yield each frame
                    for frame_data in frames_data:
                        try:
                            timestamp = float(frame_data["timestamp"])
                            unit_id = str(frame_data["unit_id"]).strip()
                            sensors = {k: float(v) for k, v in frame_data.get("sensors", {}).items()}

                            if not sensors:
                                logger.warning(f"Frame has no sensors: {frame_data}")
                                continue

                            metadata = {
                                k: v
                                for k, v in frame_data.items()
                                if k not in {"timestamp", "unit_id", "sensors"}
                            }

                            yield Frame(
                                timestamp=timestamp,
                                unit_id=unit_id,
                                sensors=sensors,
                                metadata=metadata if metadata else None,
                            )

                        except (KeyError, ValueError, TypeError) as e:
                            logger.error(f"Failed to parse frame from API response: {e}")

                    poll_count += 1

                    # Sleep before next poll
                    if self.max_polls is None or poll_count < self.max_polls:
                        time.sleep(self.poll_interval_ms / 1000.0)

                except self.requests.RequestException as e:
                    logger.error(f"API request failed: {e}")
                    raise RuntimeError(f"API request error: {e}")

        finally:
            self.session.close()

    def close(self) -> None:
        """Close session."""
        try:
            self.session.close()
        except Exception as e:
            logger.warning(f"Error closing API session: {e}")
