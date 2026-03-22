from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


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
            # WAL + relaxed sync: fewer fsyncs, better read/write concurrency for API workloads.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    site_id TEXT,
                    asset_id TEXT,
                    run_id TEXT,
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
                    run_id TEXT,
                    result_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    config_json TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, "events", "run_id", "TEXT")
            self._ensure_column(conn, "results", "run_id", "TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_results_run_id_id ON results(run_id, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_run_id_id ON events(run_id, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_active ON runs(is_active, updated_at DESC)"
            )

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_sql: str) -> None:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(c["name"]) if isinstance(c, sqlite3.Row) else str(c[1]) for c in cols}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_sql}")

    def reset(self) -> None:
        logger.info("persistence reset db_path=%s", self.db_path)
        with self._conn() as conn:
            conn.execute("DELETE FROM events")
            conn.execute("DELETE FROM results")

    def save_result(self, result: dict[str, Any], *, run_id: str | None = None) -> None:
        logger.debug("persistence write result db_path=%s", self.db_path)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO results (created_at, run_id, result_json) VALUES (?, ?, ?)",
                (_utc_now(), run_id, json.dumps(result)),
            )

    def save_ingestion(
        self,
        payload: dict[str, Any],
        result: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> None:
        """Persist frame + result in a single transaction (half the connect/commit overhead)."""
        logger.debug("persistence write ingestion db_path=%s", self.db_path)
        now = _utc_now()
        payload_json = json.dumps(payload)
        result_json = json.dumps(result)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO results (created_at, run_id, result_json) VALUES (?, ?, ?)",
                (now, run_id, result_json),
            )
            conn.execute(
                """
                INSERT INTO events (timestamp, site_id, asset_id, run_id, payload_json, result_timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("timestamp", now),
                    payload.get("site_id"),
                    payload.get("asset_id"),
                    run_id,
                    payload_json,
                    result.get("timestamp"),
                ),
            )

    def save_ingestion_batch(
        self,
        pairs: list[tuple[dict[str, Any], dict[str, Any]]],
        *,
        run_id: str | None = None,
    ) -> None:
        """Batch CSV / multi-row ingest: one transaction for all rows."""
        if not pairs:
            return
        logger.debug(
            "persistence batch write count=%s db_path=%s", len(pairs), self.db_path
        )
        with self._conn() as conn:
            for payload, result in pairs:
                now = _utc_now()
                conn.execute(
                    "INSERT INTO results (created_at, run_id, result_json) VALUES (?, ?, ?)",
                    (now, run_id, json.dumps(result)),
                )
                conn.execute(
                    """
                    INSERT INTO events (timestamp, site_id, asset_id, run_id, payload_json, result_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload.get("timestamp", now),
                        payload.get("site_id"),
                        payload.get("asset_id"),
                        run_id,
                        json.dumps(payload),
                        result.get("timestamp"),
                    ),
                )

    @staticmethod
    def _decode_result_row(row: sqlite3.Row) -> dict[str, Any]:
        result = json.loads(row["result_json"])
        result["persisted_at"] = row["created_at"]
        result["result_id"] = int(row["id"])
        result["run_id"] = row["run_id"]
        return result

    def get_latest_result(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        with self._conn() as conn:
            if run_id is None:
                row = conn.execute(
                    "SELECT id, created_at, run_id, result_json FROM results ORDER BY id DESC LIMIT 1"
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT id, created_at, run_id, result_json
                    FROM results
                    WHERE run_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
        if row is None:
            return None
        return self._decode_result_row(row)

    def save_event(
        self,
        payload: dict[str, Any],
        result: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> None:
        logger.debug("persistence write event db_path=%s", self.db_path)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO events (timestamp, site_id, asset_id, run_id, payload_json, result_timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("timestamp", _utc_now()),
                    payload.get("site_id"),
                    payload.get("asset_id"),
                    run_id,
                    json.dumps(payload),
                    result.get("timestamp"),
                ),
            )

    def list_recent_results(
        self,
        limit: int = 100,
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self._conn() as conn:
            if run_id is None:
                rows = conn.execute(
                    """
                    SELECT id, created_at, run_id, result_json
                    FROM results
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, created_at, run_id, result_json
                    FROM results
                    WHERE run_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (run_id, safe_limit),
                ).fetchall()
        return [self._decode_result_row(row) for row in rows]

    def get_result_by_id(self, result_id: int, *, run_id: str | None = None) -> dict[str, Any] | None:
        rid = int(result_id)
        with self._conn() as conn:
            if run_id is None:
                row = conn.execute(
                    """
                    SELECT id, created_at, run_id, result_json
                    FROM results
                    WHERE id = ?
                    LIMIT 1
                    """,
                    (rid,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT id, created_at, run_id, result_json
                    FROM results
                    WHERE id = ? AND run_id = ?
                    LIMIT 1
                    """,
                    (rid, run_id),
                ).fetchone()
        if row is None:
            return None
        return self._decode_result_row(row)

    def create_run(
        self,
        *,
        name: str,
        config: dict[str, Any] | None = None,
        activate: bool = True,
    ) -> dict[str, Any]:
        text = str(name).strip()
        if not text:
            raise ValueError("Run name cannot be empty")
        now = _utc_now()
        run_id = f"run_{uuid4().hex[:12]}"
        config_json = json.dumps(config or {})
        with self._conn() as conn:
            if activate:
                conn.execute("UPDATE runs SET is_active = 0")
            conn.execute(
                """
                INSERT INTO runs (run_id, name, created_at, updated_at, status, is_active, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, text, now, now, "open", 1 if activate else 0, config_json),
            )
        return self.get_run(run_id) or {
            "run_id": run_id,
            "name": text,
            "created_at": now,
            "updated_at": now,
            "status": "open",
            "is_active": activate,
            "config": config or {},
        }

    @staticmethod
    def _decode_run_row(row: sqlite3.Row) -> dict[str, Any]:
        try:
            config = json.loads(row["config_json"])
        except Exception:
            config = {}
        if not isinstance(config, dict):
            config = {}
        return {
            "run_id": row["run_id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "is_active": bool(int(row["is_active"])),
            "config": config,
        }

    def list_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT run_id, name, created_at, updated_at, status, is_active, config_json
                FROM runs
                ORDER BY datetime(created_at) DESC, rowid DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [self._decode_run_row(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT run_id, name, created_at, updated_at, status, is_active, config_json
                FROM runs
                WHERE run_id = ?
                LIMIT 1
                """,
                (str(run_id),),
            ).fetchone()
        if row is None:
            return None
        return self._decode_run_row(row)

    def get_active_run(self) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT run_id, name, created_at, updated_at, status, is_active, config_json
                FROM runs
                WHERE is_active = 1
                ORDER BY datetime(updated_at) DESC, rowid DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return self._decode_run_row(row)

    def activate_run(self, run_id: str) -> dict[str, Any]:
        current = self.get_run(run_id)
        if current is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        now = _utc_now()
        with self._conn() as conn:
            conn.execute("UPDATE runs SET is_active = 0")
            conn.execute(
                "UPDATE runs SET is_active = 1, updated_at = ? WHERE run_id = ?",
                (now, str(run_id)),
            )
        out = self.get_run(run_id)
        if out is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        return out

    def update_run(
        self,
        run_id: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_run(run_id)
        if current is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        next_name = current["name"] if name is None else str(name).strip()
        if not next_name:
            raise ValueError("Run name cannot be empty")
        next_config = current["config"] if config is None else dict(config)
        next_status = current["status"] if status is None else str(status).strip() or current["status"]
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE runs
                SET name = ?, config_json = ?, status = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (next_name, json.dumps(next_config), next_status, now, str(run_id)),
            )
        out = self.get_run(run_id)
        if out is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        return out
