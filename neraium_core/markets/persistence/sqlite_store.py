"""Local-first SQLite persistence for market ingestion/replay/live metadata."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class MarketsSQLiteStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path or os.getenv("NERAIUM_LIVE_DB_PATH", "artifacts/neraium_markets/live.sqlite3"))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fetch_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS replay_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS replay_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trader_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS suppressed_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS provider_health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                provider TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _insert_json(self, table: str, payload: dict, ticker: str | None = None) -> None:
        if ticker is None:
            self.conn.execute(f"INSERT INTO {table}(created_at, payload) VALUES (?, ?)", (self._now(), json.dumps(payload)))
        else:
            self.conn.execute(
                f"INSERT INTO {table}(created_at, ticker, payload) VALUES (?, ?, ?)",
                (self._now(), ticker, json.dumps(payload)),
            )
        self.conn.commit()

    def retain_latest(self, table: str, keep: int) -> None:
        self.conn.execute(
            f"DELETE FROM {table} WHERE id NOT IN (SELECT id FROM {table} ORDER BY id DESC LIMIT ?)",
            (keep,),
        )
        self.conn.commit()

    def record_fetch_job(self, payload: dict) -> None:
        self._insert_json("fetch_jobs", payload)

    def record_replay_run(self, run_id: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO replay_runs(run_id, created_at, payload) VALUES (?, ?, ?)",
            (run_id, self._now(), json.dumps(payload)),
        )
        self.conn.commit()

    def record_replay_output(self, run_id: str, row: dict) -> None:
        self.conn.execute(
            "INSERT INTO replay_outputs(run_id, timestamp, ticker, payload) VALUES (?, ?, ?, ?)",
            (run_id, str(row.get("timestamp")), str(row.get("ticker")), json.dumps(row)),
        )
        self.conn.commit()

    def list_replay_runs(self) -> list[dict]:
        rows = self.conn.execute("SELECT run_id, created_at, payload FROM replay_runs ORDER BY created_at DESC").fetchall()
        return [{"run_id": r["run_id"], "created_at": r["created_at"], **json.loads(r["payload"])} for r in rows]

    def get_replay_run(self, run_id: str) -> dict | None:
        row = self.conn.execute("SELECT run_id, created_at, payload FROM replay_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return {"run_id": row["run_id"], "created_at": row["created_at"], **json.loads(row["payload"])}

    def get_replay_signals(self, run_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT payload FROM replay_outputs WHERE run_id = ? ORDER BY id ASC", (run_id,)).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def record_live_session(self, payload: dict) -> None:
        self._insert_json("live_sessions", payload)

    def record_live_event(self, ticker: str, payload: dict, retention: int) -> None:
        self._insert_json("live_events", payload, ticker=ticker)
        self.retain_latest("live_events", retention)

    def record_live_bar(self, ticker: str, payload: dict, retention: int) -> None:
        self._insert_json("live_bars", payload, ticker=ticker)
        self.retain_latest("live_bars", retention)

    def record_trader_output(self, ticker: str, payload: dict) -> None:
        self._insert_json("trader_outputs", payload, ticker=ticker)

    def record_suppressed_output(self, ticker: str, payload: dict) -> None:
        self._insert_json("suppressed_outputs", payload, ticker=ticker)

    def record_provider_health(self, provider: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO provider_health_checks(created_at, provider, payload) VALUES (?, ?, ?)",
            (self._now(), provider, json.dumps(payload)),
        )
        self.conn.commit()

    def latest_provider_health(self, provider: str = "massive") -> dict | None:
        row = self.conn.execute(
            "SELECT created_at, payload FROM provider_health_checks WHERE provider = ? ORDER BY id DESC LIMIT 1",
            (provider,),
        ).fetchone()
        if row is None:
            return None
        return {"checked_at": row["created_at"], **json.loads(row["payload"])}

    def record_error(self, payload: dict) -> None:
        self._insert_json("live_errors", payload)

    def list_cached_datasets(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute("SELECT id, created_at, payload FROM fetch_jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": row["id"], "created_at": row["created_at"], **json.loads(row["payload"])} for row in rows]

    def list_live_errors(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute("SELECT created_at, payload FROM live_errors ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"created_at": row["created_at"], **json.loads(row["payload"])} for row in rows]

    def _query_signal_table(
        self,
        table: str,
        *,
        limit: int,
        ticker: str | None,
        action_permission: str | None,
        best_action: str | None,
        start_at: str | None,
        end_at: str | None,
    ) -> list[dict]:
        filters: list[str] = []
        params: list[str | int] = []
        if ticker:
            filters.append("ticker = ?")
            params.append(ticker.upper())
        if action_permission:
            filters.append("json_extract(payload, '$.action_permission') = ?")
            params.append(action_permission)
        if best_action:
            filters.append("json_extract(payload, '$.best_action') = ?")
            params.append(best_action)
        if start_at:
            filters.append("COALESCE(json_extract(payload, '$.timestamp'), created_at) >= ?")
            params.append(start_at)
        if end_at:
            filters.append("COALESCE(json_extract(payload, '$.timestamp'), created_at) <= ?")
            params.append(end_at)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        sql = f"SELECT created_at, payload FROM {table} {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [{"created_at": row["created_at"], **json.loads(row["payload"])} for row in rows]

    def list_trader_outputs(
        self,
        limit: int = 200,
        ticker: str | None = None,
        action_permission: str | None = None,
        best_action: str | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
    ) -> list[dict]:
        return self._query_signal_table(
            "trader_outputs",
            limit=limit,
            ticker=ticker,
            action_permission=action_permission,
            best_action=best_action,
            start_at=start_at,
            end_at=end_at,
        )

    def list_suppressed_outputs(
        self,
        limit: int = 200,
        ticker: str | None = None,
        action_permission: str | None = None,
        best_action: str | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
    ) -> list[dict]:
        return self._query_signal_table(
            "suppressed_outputs",
            limit=limit,
            ticker=ticker,
            action_permission=action_permission,
            best_action=best_action,
            start_at=start_at,
            end_at=end_at,
        )

    def list_replay_outputs(
        self,
        limit: int = 200,
        ticker: str | None = None,
        action_permission: str | None = None,
        best_action: str | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
    ) -> list[dict]:
        filters: list[str] = []
        params: list[str | int] = []
        if ticker:
            filters.append("ticker = ?")
            params.append(ticker.upper())
        if action_permission:
            filters.append("json_extract(payload, '$.action_permission') = ?")
            params.append(action_permission)
        if best_action:
            filters.append("json_extract(payload, '$.best_action') = ?")
            params.append(best_action)
        if start_at:
            filters.append("timestamp >= ?")
            params.append(start_at)
        if end_at:
            filters.append("timestamp <= ?")
            params.append(end_at)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        sql = f"SELECT run_id, timestamp, payload FROM replay_outputs {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [{"run_id": row["run_id"], "timestamp": row["timestamp"], **json.loads(row["payload"])} for row in rows]
