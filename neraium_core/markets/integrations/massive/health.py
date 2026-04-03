"""Provider diagnostics and health modeling for Massive."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .client import MassiveClientError, MassiveRestClient
from .config import MassiveConfig


@dataclass(slots=True)
class MassiveHealthResult:
    config_present: bool
    api_key_present: bool
    api_key_valid: bool
    rest_reachable: bool
    websocket_configured: bool
    websocket_dependency_available: bool
    websocket_reachable: bool | None
    status: str
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "config_present": self.config_present,
            "api_key_present": self.api_key_present,
            "api_key_valid": self.api_key_valid,
            "rest_reachable": self.rest_reachable,
            "websocket_configured": self.websocket_configured,
            "websocket_dependency_available": self.websocket_dependency_available,
            "websocket_reachable": self.websocket_reachable,
            "status": self.status,
            "error": self.error,
        }


def check_massive_health(config: MassiveConfig, *, timeout_seconds: float = 5.0) -> MassiveHealthResult:
    cfg_present = bool(config.rest_base_url and config.ws_base_url)
    api_key_present = bool(config.api_key)
    websocket_dependency_available = True
    try:
        import websockets  # noqa: F401
    except Exception:  # noqa: BLE001
        websocket_dependency_available = False

    if not cfg_present:
        return MassiveHealthResult(
            config_present=False,
            api_key_present=api_key_present,
            api_key_valid=False,
            rest_reachable=False,
            websocket_configured=False,
            websocket_dependency_available=websocket_dependency_available,
            websocket_reachable=False,
            status="misconfigured",
            error="Massive base URLs are not configured",
        )

    if not api_key_present:
        return MassiveHealthResult(
            config_present=True,
            api_key_present=False,
            api_key_valid=False,
            rest_reachable=False,
            websocket_configured=bool(config.ws_base_url),
            websocket_dependency_available=websocket_dependency_available,
            websocket_reachable=False,
            status="missing_api_key",
            error="MASSIVE_API_KEY is missing",
        )

    client = MassiveRestClient(config=config, timeout_seconds=timeout_seconds, max_retries=0)
    today = date.today()
    start = (today - timedelta(days=5)).isoformat()
    end = today.isoformat()
    try:
        payload = client.get_json(
            "/v2/aggs/ticker/SPY/range/1/day/{start}/{end}".format(start=start, end=end),
            params={"adjusted": "true", "sort": "desc", "limit": "3"},
        )
    except MassiveClientError as exc:
        msg = str(exc)
        invalid = "HTTP 401" in msg or "HTTP 403" in msg
        status = "invalid_api_key" if invalid else "rest_unreachable"
        return MassiveHealthResult(
            config_present=True,
            api_key_present=True,
            api_key_valid=not invalid,
            rest_reachable=False,
            websocket_configured=bool(config.ws_base_url),
            websocket_dependency_available=websocket_dependency_available,
            websocket_reachable=None,
            status=status,
            error=msg,
        )

    no_data = not (payload.get("results") or [])
    return MassiveHealthResult(
        config_present=True,
        api_key_present=True,
        api_key_valid=True,
        rest_reachable=True,
        websocket_configured=bool(config.ws_base_url),
        websocket_dependency_available=websocket_dependency_available,
        websocket_reachable=None,
        status="ok_with_no_data" if no_data else "ok",
        error="No data returned from REST probe" if no_data else None,
    )
