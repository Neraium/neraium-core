"""Rate-limit aware REST client for Massive."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .config import MassiveConfig


class MassiveClientError(RuntimeError):
    pass


@dataclass(slots=True)
class MassiveRestClient:
    config: MassiveConfig
    timeout_seconds: float = 15.0
    max_retries: int = 4
    backoff_seconds: float = 0.5

    def _build_url(self, path: str, params: dict[str, str] | None = None) -> str:
        url = f"{self.config.rest_base_url.rstrip('/')}/{path.lstrip('/')}"
        query = dict(params or {})
        if self.config.api_key:
            query["apiKey"] = self.config.api_key
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        return url

    def get_json(self, path: str, params: dict[str, str] | None = None) -> dict:
        self.config.validate()
        url = self._build_url(path, params)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = response.read().decode("utf-8")
                return json.loads(payload)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    retry_after = exc.headers.get("Retry-After")
                    wait_for = float(retry_after) if retry_after else self.backoff_seconds * (2**attempt)
                    time.sleep(wait_for)
                    continue
                body = exc.read().decode("utf-8", errors="ignore")
                raise MassiveClientError(f"Massive HTTP {exc.code}: {body[:240]}") from exc
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                break
        raise MassiveClientError(f"Massive request failed after retries: {last_error}")
