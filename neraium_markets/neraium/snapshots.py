"""Day 14: CSV/JSON snapshots of inputs and outputs (inspectable, local)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_input_snapshot(
    normalized_data: dict[str, pd.DataFrame],
    snapshot_dir: str | Path,
    run_id: str,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Write one CSV per asset under ``{snapshot_dir}/{run_id}/inputs/``."""
    base = Path(snapshot_dir) / run_id / "inputs"
    base.mkdir(parents=True, exist_ok=True)
    meta: dict[str, str] = {}
    tf_suffix = f"_{timeframe}" if timeframe else ""
    for asset, df in normalized_data.items():
        path = base / f"{asset}{tf_suffix}.csv"
        df.to_csv(path, index=False)
        meta[asset] = str(path.resolve())
    return {
        "run_id": run_id,
        "kind": "input",
        "timeframe": timeframe,
        "files": meta,
        "snapshot_dir": str(Path(snapshot_dir).resolve()),
    }


def save_output_snapshot(
    outputs: dict[str, pd.DataFrame | dict],
    snapshot_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Write DataFrames as CSV and dicts as JSON under ``{snapshot_dir}/{run_id}/outputs/``."""
    base = Path(snapshot_dir) / run_id / "outputs"
    base.mkdir(parents=True, exist_ok=True)
    meta: dict[str, str] = {}
    for name, obj in outputs.items():
        safe = str(name).replace("/", "_")
        if isinstance(obj, pd.DataFrame):
            path = base / f"{safe}.csv"
            obj.to_csv(path, index=False)
            meta[name] = str(path.resolve())
        elif isinstance(obj, dict):
            path = base / f"{safe}.json"
            path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
            meta[name] = str(path.resolve())
        else:
            continue
    return {
        "run_id": run_id,
        "kind": "output",
        "files": meta,
        "snapshot_dir": str(Path(snapshot_dir).resolve()),
    }


def load_latest_snapshot(
    snapshot_dir: str | Path,
    artifact_name: str,
) -> pd.DataFrame | dict[str, Any] | None:
    """
    Load the newest ``{artifact_name}.csv`` or ``{artifact_name}.json`` under any ``run_id/outputs``.

    Searches ``snapshot_dir/*/outputs/{artifact_name}.csv`` (then ``.json``).
    """
    root = Path(snapshot_dir)
    if not root.is_dir():
        return None

    candidates: list[tuple[float, Path]] = []
    for out_dir in root.glob("*/outputs"):
        for ext, loader in ((".csv", "csv"), (".json", "json")):
            p = out_dir / f"{artifact_name}{ext}"
            if p.is_file():
                candidates.append((p.stat().st_mtime, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    if best.suffix == ".csv":
        return pd.read_csv(best)
    try:
        return json.loads(best.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
