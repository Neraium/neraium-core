from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResultStore:
    def __init__(self, db_path: str = "neraium.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    site_id TEXT,
                    asset_id TEXT,
                    payload_json TEXT NOT NULL,
                    result_timestamp TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )

    def reset(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM events")
            conn.execute("DELETE FROM results")

    def save_result(self, result: dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO results (created_at, result_json) VALUES (?, ?)",
                (_utc_now(), json.dumps(result)),
            )
        logger.info("persistence write complete for result timestamp=%s", result.get("timestamp"))

    def get_latest_result(self) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT result_json FROM results ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["result_json"])

    def save_event(self, payload: dict[str, Any], result: dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO events (timestamp, site_id, asset_id, payload_json, result_timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload.get("timestamp", _utc_now()),
                    payload.get("site_id"),
                    payload.get("asset_id"),
                    json.dumps(payload),
                    result.get("timestamp"),
                ),
            )
        logger.info("persistence write complete for event timestamp=%s", payload.get("timestamp"))

    def list_recent_results(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT created_at, result_json FROM results ORDER BY id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()

        out = []
        for row in rows:
            result = json.loads(row["result_json"])
            result["persisted_at"] = row["created_at"]
            out.append(result)
        return out
