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

    def record_error(self, payload: dict) -> None:
        self._insert_json("live_errors", payload)
