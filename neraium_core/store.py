from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


logger = logging.getLogger(__name__)


def _store_persistence_debug_logs() -> bool:
    """When True, emit ``logger.debug`` on each SQLite write (default: off).

    Avoids per-frame logging I/O when the root logger level is DEBUG (e.g. Jupyter).
    Set ``NERAIUM_DEBUG_STORE=1`` to restore verbose persistence diagnostics.
    """

    v = os.environ.get("NERAIUM_DEBUG_STORE", "0")
    return str(v).strip().lower() not in {"0", "false", "no", "off", ""}


def _json_safe(obj: Any) -> Any:
    """Recursively convert values so json.dumps succeeds (dataclasses, numpy, nested dicts)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return _json_safe(asdict(obj))
        except TypeError:
            pass
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            return _json_safe(vars(obj))
        except Exception:
            pass
    return str(obj)
DEFAULT_CUSTOMER_ID = "default-customer"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_customer_id(customer_id: str | None) -> str:
    text = str(customer_id or "").strip()
    return text or DEFAULT_CUSTOMER_ID


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
                    customer_id TEXT NOT NULL DEFAULT 'default-customer',
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
                    customer_id TEXT NOT NULL DEFAULT 'default-customer',
                    run_id TEXT,
                    result_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL DEFAULT 'default-customer',
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    config_json TEXT NOT NULL
                )
                """
            )
            self._ensure_column(
                conn,
                "events",
                "customer_id",
                "TEXT NOT NULL DEFAULT 'default-customer'",
            )
            self._ensure_column(
                conn,
                "results",
                "customer_id",
                "TEXT NOT NULL DEFAULT 'default-customer'",
            )
            self._ensure_column(
                conn,
                "runs",
                "customer_id",
                "TEXT NOT NULL DEFAULT 'default-customer'",
            )
            self._ensure_column(conn, "events", "run_id", "TEXT")
            self._ensure_column(conn, "results", "run_id", "TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_results_customer_run_id_id "
                "ON results(customer_id, run_id, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_customer_run_id_id "
                "ON events(customer_id, run_id, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_customer_active "
                "ON runs(customer_id, is_active, updated_at DESC)"
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

    def save_result(
        self,
        result: dict[str, Any],
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> None:
        if _store_persistence_debug_logs():
            logger.debug("persistence write result db_path=%s", self.db_path)
        resolved_customer = _normalize_customer_id(customer_id)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO results (created_at, customer_id, run_id, result_json) VALUES (?, ?, ?, ?)",
                (_utc_now(), resolved_customer, run_id, json.dumps(_json_safe(result))),
            )

    def save_ingestion(
        self,
        payload: dict[str, Any],
        result: dict[str, Any],
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist frame + result in a single transaction (half the connect/commit overhead).

        Returns metadata for the inserted results row so callers can avoid a follow-up
        ``SELECT`` to recover ``result_id`` / ``persisted_at``.
        """
        if _store_persistence_debug_logs():
            logger.debug("persistence write ingestion db_path=%s", self.db_path)
        now = _utc_now()
        resolved_customer = _normalize_customer_id(customer_id)
        payload_json = json.dumps(_json_safe(payload))
        result_json = json.dumps(_json_safe(result))
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO results (created_at, customer_id, run_id, result_json) VALUES (?, ?, ?, ?)",
                (now, resolved_customer, run_id, result_json),
            )
            result_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO events (
                    customer_id, timestamp, site_id, asset_id, run_id, payload_json, result_timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_customer,
                    payload.get("timestamp", now),
                    payload.get("site_id"),
                    payload.get("asset_id"),
                    run_id,
                    payload_json,
                    result.get("timestamp"),
                ),
            )
        return {
            "customer_id": resolved_customer,
            "run_id": run_id,
            "result_id": result_id,
            "persisted_at": now,
        }

    def save_ingestion_batch(
        self,
        pairs: list[tuple[dict[str, Any], dict[str, Any]]],
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> None:
        """Batch CSV / multi-row ingest: one transaction for all rows."""
        if not pairs:
            return
        resolved_customer = _normalize_customer_id(customer_id)
        if _store_persistence_debug_logs():
            logger.debug(
                "persistence batch write count=%s db_path=%s", len(pairs), self.db_path
            )
        with self._conn() as conn:
            for payload, result in pairs:
                now = _utc_now()
                conn.execute(
                    "INSERT INTO results (created_at, customer_id, run_id, result_json) VALUES (?, ?, ?, ?)",
                    (now, resolved_customer, run_id, json.dumps(_json_safe(result))),
                )
                conn.execute(
                    """
                    INSERT INTO events (
                        customer_id, timestamp, site_id, asset_id, run_id, payload_json, result_timestamp
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_customer,
                        payload.get("timestamp", now),
                        payload.get("site_id"),
                        payload.get("asset_id"),
                        run_id,
                        json.dumps(_json_safe(payload)),
                        result.get("timestamp"),
                    ),
                )

    @staticmethod
    def _decode_result_row(row: sqlite3.Row) -> dict[str, Any]:
        result = json.loads(row["result_json"])
        result["persisted_at"] = row["created_at"]
        result["result_id"] = int(row["id"])
        result["customer_id"] = row["customer_id"]
        result["run_id"] = row["run_id"]
        return result

    def get_latest_result(
        self,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any] | None:
        resolved_customer = _normalize_customer_id(customer_id)
        with self._conn() as conn:
            if run_id is None:
                row = conn.execute(
                    """
                    SELECT id, created_at, customer_id, run_id, result_json
                    FROM results
                    WHERE customer_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (resolved_customer,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT id, created_at, customer_id, run_id, result_json
                    FROM results
                    WHERE customer_id = ? AND run_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (resolved_customer, run_id),
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
        customer_id: str | None = None,
    ) -> None:
        if _store_persistence_debug_logs():
            logger.debug("persistence write event db_path=%s", self.db_path)
        resolved_customer = _normalize_customer_id(customer_id)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    customer_id, timestamp, site_id, asset_id, run_id, payload_json, result_timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_customer,
                    payload.get("timestamp", _utc_now()),
                    payload.get("site_id"),
                    payload.get("asset_id"),
                    run_id,
                    json.dumps(_json_safe(payload)),
                    result.get("timestamp"),
                ),
            )

    def list_recent_results(
        self,
        limit: int = 100,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        resolved_customer = _normalize_customer_id(customer_id)
        with self._conn() as conn:
            if run_id is None:
                rows = conn.execute(
                    """
                    SELECT id, created_at, customer_id, run_id, result_json
                    FROM results
                    WHERE customer_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (resolved_customer, safe_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, created_at, customer_id, run_id, result_json
                    FROM results
                    WHERE customer_id = ? AND run_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (resolved_customer, run_id, safe_limit),
                ).fetchall()
        return [self._decode_result_row(row) for row in rows]

    def get_result_by_id(
        self,
        result_id: int,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any] | None:
        rid = int(result_id)
        resolved_customer = _normalize_customer_id(customer_id)
        with self._conn() as conn:
            if run_id is None:
                row = conn.execute(
                    """
                    SELECT id, created_at, customer_id, run_id, result_json
                    FROM results
                    WHERE id = ? AND customer_id = ?
                    LIMIT 1
                    """,
                    (rid, resolved_customer),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT id, created_at, customer_id, run_id, result_json
                    FROM results
                    WHERE id = ? AND run_id = ? AND customer_id = ?
                    LIMIT 1
                    """,
                    (rid, run_id, resolved_customer),
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
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        text = str(name).strip()
        if not text:
            raise ValueError("Run name cannot be empty")
        resolved_customer = _normalize_customer_id(customer_id)
        now = _utc_now()
        run_id = f"run_{uuid4().hex[:12]}"
        config_json = json.dumps(config or {})
        with self._conn() as conn:
            if activate:
                conn.execute("UPDATE runs SET is_active = 0 WHERE customer_id = ?", (resolved_customer,))
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, customer_id, name, created_at, updated_at, status, is_active, config_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, resolved_customer, text, now, now, "open", 1 if activate else 0, config_json),
            )
        return self.get_run(run_id, customer_id=resolved_customer) or {
            "run_id": run_id,
            "customer_id": resolved_customer,
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
            "customer_id": row["customer_id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "is_active": bool(int(row["is_active"])),
            "config": config,
        }

    def list_runs(
        self,
        *,
        limit: int = 100,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        resolved_customer = _normalize_customer_id(customer_id)
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT run_id, customer_id, name, created_at, updated_at, status, is_active, config_json
                FROM runs
                WHERE customer_id = ?
                ORDER BY datetime(created_at) DESC, rowid DESC
                LIMIT ?
                """,
                (resolved_customer, safe_limit),
            ).fetchall()
        return [self._decode_run_row(row) for row in rows]

    def get_run(self, run_id: str, *, customer_id: str | None = None) -> dict[str, Any] | None:
        resolved_customer = _normalize_customer_id(customer_id)
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT run_id, customer_id, name, created_at, updated_at, status, is_active, config_json
                FROM runs
                WHERE run_id = ? AND customer_id = ?
                LIMIT 1
                """,
                (str(run_id), resolved_customer),
            ).fetchone()
        if row is None:
            return None
        return self._decode_run_row(row)

    def get_active_run(self, *, customer_id: str | None = None) -> dict[str, Any] | None:
        resolved_customer = _normalize_customer_id(customer_id)
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT run_id, customer_id, name, created_at, updated_at, status, is_active, config_json
                FROM runs
                WHERE customer_id = ? AND is_active = 1
                ORDER BY datetime(updated_at) DESC, rowid DESC
                LIMIT 1
                """,
                (resolved_customer,),
            ).fetchone()
        if row is None:
            return None
        return self._decode_run_row(row)

    def activate_run(self, run_id: str, *, customer_id: str | None = None) -> dict[str, Any]:
        resolved_customer = _normalize_customer_id(customer_id)
        current = self.get_run(run_id, customer_id=resolved_customer)
        if current is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        now = _utc_now()
        with self._conn() as conn:
            conn.execute("UPDATE runs SET is_active = 0 WHERE customer_id = ?", (resolved_customer,))
            conn.execute(
                "UPDATE runs SET is_active = 1, updated_at = ? WHERE run_id = ? AND customer_id = ?",
                (now, str(run_id), resolved_customer),
            )
        out = self.get_run(run_id, customer_id=resolved_customer)
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
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_customer = _normalize_customer_id(customer_id)
        current = self.get_run(run_id, customer_id=resolved_customer)
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
                WHERE run_id = ? AND customer_id = ?
                """,
                (next_name, json.dumps(next_config), next_status, now, str(run_id), resolved_customer),
            )
        out = self.get_run(run_id, customer_id=resolved_customer)
        if out is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        return out
