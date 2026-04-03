from __future__ import annotations

from enum import Enum


class LiveSessionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED_WARMING_UP = "connected_warming_up"
    CONNECTED_LIVE = "connected_live"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    ERROR = "error"
