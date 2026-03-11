from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import floor
from typing import Optional

from pydantic import BaseModel


class AlignedWindow(BaseModel):
    system_id: str
    window_end: datetime
    vector_order: list[str]
    vector: list[float | None]
    missing_signals: list[str]
    filled_signals: list[str]
    missing_fraction: float
    accepted_for_scoring: bool


class AlignmentEngine:
    def __init__(self, definition) -> None:
        self._definition = definition
        self._known_signals = {signal.name for signal in definition.signals}
        self._last_emitted_window_index: Optional[int] = None
        self._last_ingested_timestamp: Optional[datetime] = None
        self._last_known_values: dict[str, float | None] = {
            name: None for name in self._known_signals
        }
        self._last_observed_window_index: dict[str, int | None] = {
            name: None for name in self._known_signals
        }

    def ingest(self, payload) -> list[AlignedWindow]:
        if payload.system_id != self._definition.system_id:
            raise ValueError(
                "Telemetry payload system_id must match AlignmentEngine system_id. "
                f"Got payload.system_id={payload.system_id!r} and "
                f"engine.system_id={self._definition.system_id!r}."
            )

        if (
            self._last_ingested_timestamp is not None
            and payload.timestamp < self._last_ingested_timestamp
        ):
            raise ValueError(
                f"Out-of-order payload timestamp for system_id={payload.system_id!r}. "
                f"Last timestamp={self._last_ingested_timestamp.isoformat()}, "
                f"received={payload.timestamp.isoformat()}."
            )

        unknown_signals = sorted(set(payload.signals) - self._known_signals)
        if unknown_signals:
            raise ValueError(
                "Telemetry payload contains unknown signals for this system definition. "
                f"Unknown signal names: {unknown_signals}."
            )

        current_window_index = self._window_index(
            payload.timestamp,
            self._definition.inference_window_seconds,
        )

        emitted_windows: list[AlignedWindow] = []
        if self._last_emitted_window_index is None:
            self._last_emitted_window_index = current_window_index - 1

        for window_index in range(self._last_emitted_window_index + 1, current_window_index):
            emitted_windows.append(self._build_window(window_index))

        self._last_emitted_window_index = current_window_index - 1

        for signal_name, value in payload.signals.items():
            self._last_known_values[signal_name] = None if value is None else float(value)
            self._last_observed_window_index[signal_name] = current_window_index

        self._last_ingested_timestamp = payload.timestamp
        return emitted_windows

    @staticmethod
    def _window_index(timestamp: datetime, window_seconds: int) -> int:
        ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        return floor(ts.timestamp() / window_seconds)

    def _build_window(self, window_index: int) -> AlignedWindow:
        vector: list[float | None] = []
        missing_signals: list[str] = []
        filled_signals: list[str] = []

        for signal_name in self._definition.vector_order:
            last_seen_index = self._last_observed_window_index[signal_name]
            value = self._last_known_values[signal_name]

            if last_seen_index is None or value is None:
                vector.append(None)
                missing_signals.append(signal_name)
                continue

            age_in_windows = window_index - last_seen_index
            if age_in_windows == 0:
                vector.append(value)
            elif 0 < age_in_windows <= self._definition.max_forward_fill_windows:
                vector.append(value)
                filled_signals.append(signal_name)
            else:
                vector.append(None)
                missing_signals.append(signal_name)

        missing_fraction = len(missing_signals) / len(self._definition.vector_order)
        accepted_for_scoring = (
            missing_fraction <= self._definition.max_missing_signal_fraction
        )

        return AlignedWindow(
            system_id=self._definition.system_id,
            window_end=self._window_end(
                window_index,
                self._definition.inference_window_seconds,
            ),
            vector_order=list(self._definition.vector_order),
            vector=vector,
            missing_signals=missing_signals,
            filled_signals=filled_signals,
            missing_fraction=missing_fraction,
            accepted_for_scoring=accepted_for_scoring,
        )

    @staticmethod
    def _window_end(window_index: int, window_seconds: int) -> datetime:
        total_seconds = (window_index + 1) * window_seconds
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=total_seconds)