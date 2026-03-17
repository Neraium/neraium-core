from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class SQLiteStore:
    def __init__(self, path: str = "sii_state.db") -> None:
        self.path = Path(path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kv_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def set(self, key: str, value: dict) -> None:
        payload = json.dumps(value)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO kv_state(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, payload),
            )

    def get(self, key: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT value FROM kv_state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
