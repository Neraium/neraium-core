"""Day 14: cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neraium.cache import cache_exists, load_cache_artifact, make_cache_key, save_cache_artifact


def test_cache_key_stable() -> None:
    k1 = make_cache_key("merged", "dev", "daily", {"x": 1})
    k2 = make_cache_key("merged", "dev", "daily", {"x": 1})
    assert k1 == k2


def test_cache_save_and_load(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1]})
    key = make_cache_key("t", "dev", None, {})
    path = save_cache_artifact(df, tmp_path, key)
    assert Path(path).is_file()
    assert cache_exists(tmp_path, key)
    out = load_cache_artifact(tmp_path, key)
    assert len(out) == 1
