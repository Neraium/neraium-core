from __future__ import annotations

import pytest

from neraium_core.runtime_config import RuntimeConfig


def test_runtime_config_from_env_defaults(monkeypatch) -> None:
    for key in (
        "NERAIUM_BASELINE_WINDOW",
        "NERAIUM_RECENT_WINDOW",
        "NERAIUM_RUNTIME_MODE",
        "NERAIUM_INPUT_PATH",
        "NERAIUM_DATA_PATH",
        "NERAIUM_REQUIRE_INPUT_PATH",
        "NERAIUM_REQUIRE_DATA_PATH",
    ):
        monkeypatch.delenv(key, raising=False)

    cfg = RuntimeConfig.from_env()
    assert cfg.baseline_window == 24
    assert cfg.recent_window == 8
    assert cfg.runtime_mode == "development"


def test_runtime_config_rejects_invalid_window(monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_BASELINE_WINDOW", "2")
    with pytest.raises(ValueError, match="NERAIUM_BASELINE_WINDOW"):
        RuntimeConfig.from_env()


def test_runtime_config_requires_existing_input_path_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_REQUIRE_INPUT_PATH", "1")
    monkeypatch.setenv("NERAIUM_INPUT_PATH", "/definitely/missing/path.csv")
    with pytest.raises(ValueError, match="NERAIUM_INPUT_PATH"):
        RuntimeConfig.from_env()


def test_runtime_config_rejects_nonexistent_input_path_when_provided(monkeypatch) -> None:
    monkeypatch.delenv("NERAIUM_REQUIRE_INPUT_PATH", raising=False)
    monkeypatch.setenv("NERAIUM_INPUT_PATH", "/definitely/missing/optional-input.csv")
    with pytest.raises(ValueError, match="NERAIUM_INPUT_PATH"):
        RuntimeConfig.from_env()
