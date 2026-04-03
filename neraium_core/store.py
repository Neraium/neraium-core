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
<<<<<<< HEAD


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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS service_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    customer_id TEXT NOT NULL DEFAULT 'default-customer',
                    run_id TEXT,
                    timestamp TEXT,
                    cycle INTEGER,
                    risk_state TEXT,
                    decision TEXT,
                    confidence REAL,
                    top_drivers_json TEXT NOT NULL,
                    explanation_text TEXT,
                    event_flags_json TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS structural_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    customer_id TEXT NOT NULL DEFAULT 'default-customer',
                    run_id TEXT,
                    site_id TEXT,
                    asset_id TEXT,
                    timestamp TEXT,
                    cycle INTEGER,
                    memory_key TEXT NOT NULL,
                    family_label TEXT,
                    novelty_is_novel INTEGER NOT NULL DEFAULT 1,
                    novelty_score REAL NOT NULL DEFAULT 1.0,
                    signature_json TEXT NOT NULL,
                    summary TEXT,
                    decision TEXT,
                    risk_level TEXT,
                    event_flags_json TEXT NOT NULL,
                    source_history_id INTEGER,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_state (
                    state_key TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL DEFAULT 'default-customer',
                    run_id TEXT,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_service_history_customer_run_id_id "
                "ON service_history(customer_id, run_id, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_structural_memory_scope_cycle "
                "ON structural_memory(customer_id, site_id, asset_id, cycle DESC, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_structural_memory_customer_id "
                "ON structural_memory(customer_id, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_structural_memory_memory_key "
                "ON structural_memory(customer_id, memory_key, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_operational_state_customer_key "
                "ON operational_state(customer_id, state_key)"
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
            conn.execute("DELETE FROM service_history")
            conn.execute("DELETE FROM structural_memory")
            conn.execute("DELETE FROM operational_state")

    def upsert_operational_state(
        self,
        *,
        state_key: str,
        state: dict[str, Any],
        customer_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        key = str(state_key or "").strip()
        if not key:
            raise ValueError("state_key is required")
        resolved_customer = _normalize_customer_id(customer_id)
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO operational_state (state_key, customer_id, run_id, state_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    run_id=excluded.run_id,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (key, resolved_customer, run_id, json.dumps(_json_safe(state)), now),
            )
        return {"state_key": key, "customer_id": resolved_customer, "run_id": run_id, "updated_at": now}

    def delete_operational_state(self, *, state_key: str) -> None:
        key = str(state_key or "").strip()
        if not key:
            return
        with self._conn() as conn:
            conn.execute("DELETE FROM operational_state WHERE state_key = ?", (key,))

    def get_operational_state(self, *, state_key: str) -> dict[str, Any] | None:
        key = str(state_key or "").strip()
        if not key:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT state_key, customer_id, run_id, state_json, updated_at
                FROM operational_state
                WHERE state_key = ?
                LIMIT 1
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            decoded = json.loads(row["state_json"])
        except Exception:
            decoded = {}
        if not isinstance(decoded, dict):
            decoded = {}
=======


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


class EventStore:
    """
    Legacy event store retained for backward-compatible scripts/tests.
    """

    def __init__(self, db_path: str = "neraium_events.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                status TEXT,
                score REAL,
                signals TEXT,
                features TEXT,
                aligned TEXT,
                anomaly TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, event: dict[str, Any]) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO events (timestamp, status, score, signals, features, aligned, anomaly)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("timestamp"),
                event.get("status"),
                event.get("score"),
                json.dumps(event.get("signals", {})),
                json.dumps(event.get("features", {})),
                json.dumps(event.get("aligned", [])),
                json.dumps(event.get("anomaly", {})),
            ),
        )
        self.conn.commit()

    def all(self) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, status, score, signals, features, aligned, anomaly
            FROM events
            ORDER BY id ASC
            """
        )
        rows = cursor.fetchall()
        return [
            {
                "timestamp": row[0],
                "status": row[1],
                "score": row[2],
                "signals": json.loads(row[3]),
                "features": json.loads(row[4]),
                "aligned": json.loads(row[5]),
                "anomaly": json.loads(row[6]),
            }
            for row in rows
        ]

    def latest(self) -> dict[str, Any] | None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, status, score, signals, features, aligned, anomaly
            FROM events
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if not row:
            return None
>>>>>>> origin/main
        return {
            "state_key": str(row["state_key"]),
            "customer_id": str(row["customer_id"]),
            "run_id": row["run_id"],
            "state": decoded,
            "updated_at": str(row["updated_at"]),
        }

<<<<<<< HEAD
    def list_operational_state(self, *, key_prefix: str | None = None) -> list[dict[str, Any]]:
        prefix = str(key_prefix or "").strip()
        with self._conn() as conn:
            if prefix:
                rows = conn.execute(
                    """
                    SELECT state_key, customer_id, run_id, state_json, updated_at
                    FROM operational_state
                    WHERE state_key LIKE ?
                    ORDER BY state_key ASC
                    """,
                    (f"{prefix}%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT state_key, customer_id, run_id, state_json, updated_at
                    FROM operational_state
                    ORDER BY state_key ASC
                    """
                ).fetchall()
        decoded_rows: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["state_json"])
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            decoded_rows.append(
                {
                    "state_key": str(row["state_key"]),
                    "customer_id": str(row["customer_id"]),
                    "run_id": row["run_id"],
                    "state": payload,
                    "updated_at": str(row["updated_at"]),
                }
            )
        return decoded_rows

    def save_service_history(
        self,
        record: dict[str, Any],
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_customer = _normalize_customer_id(customer_id)
        now = _utc_now()
        risk_assessment = record.get("risk_assessment") if isinstance(record.get("risk_assessment"), dict) else {}
        risk_state = str(risk_assessment.get("risk_level", "UNKNOWN"))
        decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
        decision_label = str(
            decision.get("action")
            or decision.get("state")
            or decision.get("resolved_action")
            or "unknown"
        )
        attribution = record.get("attribution") if isinstance(record.get("attribution"), dict) else {}
        top_drivers = attribution.get("top_drivers")
        if not isinstance(top_drivers, list):
            top_drivers = []
        top_drivers = top_drivers[:5]
        events = record.get("events")
        if not isinstance(events, list):
            events = []
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO service_history (
                    created_at,
                    customer_id,
                    run_id,
                    timestamp,
                    cycle,
                    risk_state,
                    decision,
                    confidence,
                    top_drivers_json,
                    explanation_text,
                    event_flags_json,
                    record_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    resolved_customer,
                    run_id,
                    record.get("timestamp"),
                    int(record.get("cycle")) if record.get("cycle") is not None else None,
                    risk_state,
                    decision_label,
                    float(record.get("confidence", 0.0) or 0.0),
                    json.dumps(_json_safe(top_drivers)),
                    str(record.get("explanation_text", "")),
                    json.dumps(_json_safe(events)),
                    json.dumps(_json_safe(record)),
                ),
            )
            history_id = int(cur.lastrowid)
        return {"history_id": history_id, "persisted_at": now, "customer_id": resolved_customer, "run_id": run_id}

    @staticmethod
    def _decode_structural_memory_row(row: sqlite3.Row) -> dict[str, Any]:
        try:
            signature = json.loads(row["signature_json"])
        except Exception:
            signature = {}
        if not isinstance(signature, dict):
            signature = {}
        try:
            events = json.loads(row["event_flags_json"])
        except Exception:
            events = []
        if not isinstance(events, list):
            events = []
        try:
            metadata = json.loads(row["metadata_json"])
        except Exception:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
=======
    def anomalies(self) -> list[dict[str, Any]]:
        return [event for event in self.all() if event.get("status") == "anomaly"]

    def structural_summary(self) -> dict[str, Any]:
        events = self.all()
        if not events:
            return {
                "total_events": 0,
                "total_structural_anomalies": 0,
                "latest_status": "no data",
                "latest_drift_score": 0,
                "latest_vector": [],
                "relationship_points": [],
                "baseline_centroid": {"x": 0, "y": 0},
                "drift_threshold": 30,
            }

        anomalies = [e for e in events if e["status"] == "anomaly"]
        latest = events[-1]
        relationship_points = [
            {
                "x": e["signals"].get("cpu_usage", 0),
                "y": e["signals"].get("memory_usage", 0),
                "status": e["status"],
                "timestamp": e["timestamp"],
                "score": e["score"],
            }
            for e in events[-50:]
        ]
        cx = sum(p["x"] for p in relationship_points) / len(relationship_points)
        cy = sum(p["y"] for p in relationship_points) / len(relationship_points)
>>>>>>> origin/main
        return {
            "memory_id": int(row["id"]),
            "persisted_at": row["created_at"],
            "customer_id": row["customer_id"],
            "run_id": row["run_id"],
            "site_id": row["site_id"],
            "asset_id": row["asset_id"],
            "timestamp": row["timestamp"],
            "cycle": row["cycle"],
            "memory_key": row["memory_key"],
            "family_label": row["family_label"],
            "novelty": {
                "is_novel": bool(int(row["novelty_is_novel"])),
                "score": float(row["novelty_score"] or 0.0),
            },
            "signature": signature,
            "summary": row["summary"],
            "decision": row["decision"],
            "risk_level": row["risk_level"],
            "event_flags": events,
            "source_history_id": row["source_history_id"],
            "metadata": metadata,
        }

<<<<<<< HEAD
=======

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS service_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    customer_id TEXT NOT NULL DEFAULT 'default-customer',
                    run_id TEXT,
                    timestamp TEXT,
                    cycle INTEGER,
                    risk_state TEXT,
                    decision TEXT,
                    confidence REAL,
                    top_drivers_json TEXT NOT NULL,
                    explanation_text TEXT,
                    event_flags_json TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS structural_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    customer_id TEXT NOT NULL DEFAULT 'default-customer',
                    run_id TEXT,
                    site_id TEXT,
                    asset_id TEXT,
                    timestamp TEXT,
                    cycle INTEGER,
                    memory_key TEXT NOT NULL,
                    family_label TEXT,
                    novelty_is_novel INTEGER NOT NULL DEFAULT 1,
                    novelty_score REAL NOT NULL DEFAULT 1.0,
                    signature_json TEXT NOT NULL,
                    summary TEXT,
                    decision TEXT,
                    risk_level TEXT,
                    event_flags_json TEXT NOT NULL,
                    source_history_id INTEGER,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_state (
                    state_key TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL DEFAULT 'default-customer',
                    run_id TEXT,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_service_history_customer_run_id_id "
                "ON service_history(customer_id, run_id, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_structural_memory_scope_cycle "
                "ON structural_memory(customer_id, site_id, asset_id, cycle DESC, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_structural_memory_customer_id "
                "ON structural_memory(customer_id, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_structural_memory_memory_key "
                "ON structural_memory(customer_id, memory_key, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_operational_state_customer_key "
                "ON operational_state(customer_id, state_key)"
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
            conn.execute("DELETE FROM service_history")
            conn.execute("DELETE FROM structural_memory")
            conn.execute("DELETE FROM operational_state")

    def upsert_operational_state(
        self,
        *,
        state_key: str,
        state: dict[str, Any],
        customer_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        key = str(state_key or "").strip()
        if not key:
            raise ValueError("state_key is required")
        resolved_customer = _normalize_customer_id(customer_id)
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO operational_state (state_key, customer_id, run_id, state_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    run_id=excluded.run_id,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (key, resolved_customer, run_id, json.dumps(_json_safe(state)), now),
            )
        return {"state_key": key, "customer_id": resolved_customer, "run_id": run_id, "updated_at": now}

    def delete_operational_state(self, *, state_key: str) -> None:
        key = str(state_key or "").strip()
        if not key:
            return
        with self._conn() as conn:
            conn.execute("DELETE FROM operational_state WHERE state_key = ?", (key,))

    def get_operational_state(self, *, state_key: str) -> dict[str, Any] | None:
        key = str(state_key or "").strip()
        if not key:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT state_key, customer_id, run_id, state_json, updated_at
                FROM operational_state
                WHERE state_key = ?
                LIMIT 1
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            decoded = json.loads(row["state_json"])
        except Exception:
            decoded = {}
        if not isinstance(decoded, dict):
            decoded = {}
        return {
            "state_key": str(row["state_key"]),
            "customer_id": str(row["customer_id"]),
            "run_id": row["run_id"],
            "state": decoded,
            "updated_at": str(row["updated_at"]),
        }

    def list_operational_state(self, *, key_prefix: str | None = None) -> list[dict[str, Any]]:
        prefix = str(key_prefix or "").strip()
        with self._conn() as conn:
            if prefix:
                rows = conn.execute(
                    """
                    SELECT state_key, customer_id, run_id, state_json, updated_at
                    FROM operational_state
                    WHERE state_key LIKE ?
                    ORDER BY state_key ASC
                    """,
                    (f"{prefix}%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT state_key, customer_id, run_id, state_json, updated_at
                    FROM operational_state
                    ORDER BY state_key ASC
                    """
                ).fetchall()
        decoded_rows: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["state_json"])
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            decoded_rows.append(
                {
                    "state_key": str(row["state_key"]),
                    "customer_id": str(row["customer_id"]),
                    "run_id": row["run_id"],
                    "state": payload,
                    "updated_at": str(row["updated_at"]),
                }
            )
        return decoded_rows

    def save_service_history(
        self,
        record: dict[str, Any],
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_customer = _normalize_customer_id(customer_id)
        now = _utc_now()
        risk_assessment = record.get("risk_assessment") if isinstance(record.get("risk_assessment"), dict) else {}
        risk_state = str(risk_assessment.get("risk_level", "UNKNOWN"))
        decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
        decision_label = str(
            decision.get("action")
            or decision.get("state")
            or decision.get("resolved_action")
            or "unknown"
        )
        attribution = record.get("attribution") if isinstance(record.get("attribution"), dict) else {}
        top_drivers = attribution.get("top_drivers")
        if not isinstance(top_drivers, list):
            top_drivers = []
        top_drivers = top_drivers[:5]
        events = record.get("events")
        if not isinstance(events, list):
            events = []
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO service_history (
                    created_at,
                    customer_id,
                    run_id,
                    timestamp,
                    cycle,
                    risk_state,
                    decision,
                    confidence,
                    top_drivers_json,
                    explanation_text,
                    event_flags_json,
                    record_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    resolved_customer,
                    run_id,
                    record.get("timestamp"),
                    int(record.get("cycle")) if record.get("cycle") is not None else None,
                    risk_state,
                    decision_label,
                    float(record.get("confidence", 0.0) or 0.0),
                    json.dumps(_json_safe(top_drivers)),
                    str(record.get("explanation_text", "")),
                    json.dumps(_json_safe(events)),
                    json.dumps(_json_safe(record)),
                ),
            )
            history_id = int(cur.lastrowid)
        return {"history_id": history_id, "persisted_at": now, "customer_id": resolved_customer, "run_id": run_id}

    @staticmethod
    def _decode_structural_memory_row(row: sqlite3.Row) -> dict[str, Any]:
        try:
            signature = json.loads(row["signature_json"])
        except Exception:
            signature = {}
        if not isinstance(signature, dict):
            signature = {}
        try:
            events = json.loads(row["event_flags_json"])
        except Exception:
            events = []
        if not isinstance(events, list):
            events = []
        try:
            metadata = json.loads(row["metadata_json"])
        except Exception:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "memory_id": int(row["id"]),
            "persisted_at": row["created_at"],
            "customer_id": row["customer_id"],
            "run_id": row["run_id"],
            "site_id": row["site_id"],
            "asset_id": row["asset_id"],
            "timestamp": row["timestamp"],
            "cycle": row["cycle"],
            "memory_key": row["memory_key"],
            "family_label": row["family_label"],
            "novelty": {
                "is_novel": bool(int(row["novelty_is_novel"])),
                "score": float(row["novelty_score"] or 0.0),
            },
            "signature": signature,
            "summary": row["summary"],
            "decision": row["decision"],
            "risk_level": row["risk_level"],
            "event_flags": events,
            "source_history_id": row["source_history_id"],
            "metadata": metadata,
        }

>>>>>>> origin/main
    def save_structural_memory(
        self,
        record: dict[str, Any],
        *,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        resolved_customer = _normalize_customer_id(customer_id or record.get("customer_id"))
        novelty = record.get("novelty") if isinstance(record.get("novelty"), dict) else {}
        signature = record.get("signature") if isinstance(record.get("signature"), dict) else {}
        events = record.get("event_flags") if isinstance(record.get("event_flags"), list) else []
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO structural_memory (
                    created_at, customer_id, run_id, site_id, asset_id, timestamp, cycle, memory_key,
                    family_label, novelty_is_novel, novelty_score, signature_json, summary, decision,
                    risk_level, event_flags_json, source_history_id, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    resolved_customer,
                    record.get("run_id"),
                    record.get("site_id"),
                    record.get("asset_id"),
                    record.get("timestamp"),
                    int(record.get("cycle")) if record.get("cycle") is not None else None,
                    str(record.get("memory_key", "")),
                    record.get("family_label"),
                    1 if bool(novelty.get("is_novel", True)) else 0,
                    float(novelty.get("score", 1.0) or 1.0),
                    json.dumps(_json_safe(signature)),
                    record.get("summary"),
                    record.get("decision"),
                    record.get("risk_level"),
                    json.dumps(_json_safe(events)),
                    int(record.get("source_history_id")) if record.get("source_history_id") is not None else None,
                    json.dumps(_json_safe(metadata)),
                ),
            )
            memory_id = int(cur.lastrowid)
        return {"memory_id": memory_id, "persisted_at": now, "customer_id": resolved_customer}

    def list_structural_memory(
        self,
        *,
        customer_id: str | None = None,
        run_id: str | None = None,
        site_id: str | None = None,
        asset_id: str | None = None,
        limit: int = 100,
        exclude_cycle: int | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 2000))
        resolved_customer = _normalize_customer_id(customer_id)
        clauses = ["customer_id = ?"]
        args: list[Any] = [resolved_customer]
        if run_id is not None:
            clauses.append("run_id = ?")
            args.append(run_id)
        if site_id is not None:
            clauses.append("site_id = ?")
            args.append(site_id)
        if asset_id is not None:
            clauses.append("asset_id = ?")
            args.append(asset_id)
        if exclude_cycle is not None:
            clauses.append("(cycle IS NULL OR cycle <> ?)")
            args.append(int(exclude_cycle))
        where = " AND ".join(clauses)
        args.append(safe_limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM structural_memory
                WHERE {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(args),
            ).fetchall()
        return [self._decode_structural_memory_row(row) for row in rows]

    @staticmethod
    def _decode_service_history_row(row: sqlite3.Row) -> dict[str, Any]:
        try:
            record = json.loads(row["record_json"])
        except Exception:
            record = {}
        if not isinstance(record, dict):
            record = {}
        record["history_id"] = int(row["id"])
        record["persisted_at"] = row["created_at"]
        record["customer_id"] = row["customer_id"]
        record["run_id"] = row["run_id"]
        return record

    def get_latest_service_history(
        self,
        *,
        run_id: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any] | None:
        items = self.list_service_history(limit=1, run_id=run_id, customer_id=customer_id)
        return items[0] if items else None

    def list_service_history(
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
                    SELECT id, created_at, customer_id, run_id, record_json
                    FROM service_history
                    WHERE customer_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (resolved_customer, safe_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, created_at, customer_id, run_id, record_json
                    FROM service_history
                    WHERE customer_id = ? AND run_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (resolved_customer, run_id, safe_limit),
                ).fetchall()
        return [self._decode_service_history_row(row) for row in rows]

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
