"""Massive adapter configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


class MassiveConfigError(ValueError):
    pass


@dataclass(slots=True)
class MassiveConfig:
    api_key: str | None
    rest_base_url: str
    ws_base_url: str

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def validate(self, *, require_api_key: bool = True) -> None:
        if require_api_key and not self.api_key:
            raise MassiveConfigError("MASSIVE_API_KEY is required for Massive integration")
        if not self.rest_base_url.startswith(("http://", "https://")):
            raise MassiveConfigError("MASSIVE_REST_BASE_URL must start with http:// or https://")
        if not self.ws_base_url.startswith(("ws://", "wss://")):
            raise MassiveConfigError("MASSIVE_WS_BASE_URL must start with ws:// or wss://")



def load_massive_config() -> MassiveConfig:
    cfg = MassiveConfig(
        api_key=os.getenv("MASSIVE_API_KEY"),
        rest_base_url=os.getenv("MASSIVE_REST_BASE_URL", "https://api.massive.com"),
        ws_base_url=os.getenv("MASSIVE_WS_BASE_URL", "wss://socket.massive.com/stocks"),
    )
    return cfg
