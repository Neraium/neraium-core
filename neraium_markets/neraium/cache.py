"""Day 14: file-based cache for intermediate artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def make_cache_key(
    artifact_name: str,
    profile: str,
    timeframe: str | None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Stable deterministic key (hex digest) from inputs."""
    payload = {
        "artifact": artifact_name,
        "profile": profile,
        "timeframe": timeframe,
        "extra": extra or {},
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.cache.pkl"


def save_cache_artifact(
    artifact: Any,
    cache_dir: str | Path,
    cache_key: str,
) -> str:
    """Serialize ``artifact`` to pickle under ``cache_dir`` (inspectable via separate export)."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = _cache_path(d, cache_key)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return str(path.resolve())


def load_cache_artifact(cache_dir: str | Path, cache_key: str) -> Any:
    """Load pickled artifact, or raise ``FileNotFoundError``."""
    path = _cache_path(Path(cache_dir), cache_key)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as fh:
        return pickle.load(fh)


def cache_exists(cache_dir: str | Path, cache_key: str) -> bool:
    return _cache_path(Path(cache_dir), cache_key).is_file()
