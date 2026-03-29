from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error, request

AUX_TIME_TIME_URL = "https://api-v2.aux-time.com/time"


Gal2Payload = dict[str, Any]


def unavailable_payload(reason: str) -> Gal2Payload:
    return {
        "available": False,
        "reason": reason,
        "aux_time_time": None,
        "drift_ms": None,
        "wobble_ms": None,
        "live_ms": None,
        "fractal_factor": None,
    }


@dataclass
class AUX_TIMEClient:
    timeout_seconds: float = 3.0
    max_attempts: int = 3
    cache_ms: int = 0
    sleep_fn: Callable[[float], None] = time.sleep

    _cache_payload: Gal2Payload | None = None
    _cache_expiration_epoch_ms: float = 0.0

    def _headers(self) -> dict[str, str]:
        api_key = os.getenv("AUX_TIME_KEY")
        if not api_key:
            raise RuntimeError("AUX_TIME_KEY is not set")
        return {
            "x-api-key": api_key,
            "Accept": "application/json",
        }

    def _should_retry_http(self, status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code <= 599

    def _parse_payload(self, payload: dict[str, Any]) -> Gal2Payload:
        return {
            "available": True,
            "reason": None,
            "aux_time_time": payload.get("aux_time_time"),
            "drift_ms": payload.get("drift_ms"),
            "wobble_ms": payload.get("wobble_ms"),
            "live_ms": payload.get("live_ms"),
            "fractal_factor": payload.get("fractal_factor"),
            "utc_time": payload.get("utc_time"),
            "base_ms": payload.get("base_ms"),
        }

    def _get_from_api(self) -> Gal2Payload:
        req = request.Request(AUX_TIME_TIME_URL, headers=self._headers(), method="GET")

        for attempt in range(1, self.max_attempts + 1):
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    parsed = json.loads(body)
                    return self._parse_payload(parsed)
            except error.HTTPError as exc:
                if attempt < self.max_attempts and self._should_retry_http(exc.code):
                    self.sleep_fn(0.1 * (2 ** (attempt - 1)))
                    continue
                return unavailable_payload(f"http_error_{exc.code}")
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt < self.max_attempts:
                    self.sleep_fn(0.1 * (2 ** (attempt - 1)))
                    continue
                return unavailable_payload(str(exc))

        return unavailable_payload("unknown_error")

    def get_time(self) -> Gal2Payload:
        now_ms = time.time() * 1000.0
        if self.cache_ms > 0 and self._cache_payload is not None and now_ms < self._cache_expiration_epoch_ms:
            return self._cache_payload

        payload = self._get_from_api()
        if self.cache_ms > 0:
            self._cache_payload = payload
            self._cache_expiration_epoch_ms = now_ms + float(self.cache_ms)
        return payload
