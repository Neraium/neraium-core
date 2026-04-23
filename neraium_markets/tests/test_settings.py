"""Day 12: settings profiles and validation."""

from __future__ import annotations

import pytest

from neraium.settings import load_settings, validate_settings, with_overrides


def test_load_settings_profile_dev() -> None:
    s = load_settings("dev")
    assert s.profile == "dev"
    assert s.max_cycles >= 1
    assert "daily" in s.active_timeframes


def test_load_settings_profile_test() -> None:
    s = load_settings("test")
    assert s.profile == "test"
    assert s.polling_interval_seconds >= 1


def test_load_settings_profile_prod_local() -> None:
    s = load_settings("prod_local")
    assert s.profile == "prod_local"
    assert s.save_outputs is True


def test_settings_validation_ok() -> None:
    s = load_settings("dev")
    validate_settings(s)


def test_settings_validation_invalid_max_cycles() -> None:
    s = with_overrides(load_settings("dev"), max_cycles=0)
    with pytest.raises(ValueError, match="max_cycles"):
        validate_settings(s)


def test_settings_validation_bad_timeframe() -> None:
    s = with_overrides(load_settings("dev"), active_timeframes=("1m",))
    with pytest.raises(ValueError, match="timeframe"):
        validate_settings(s)


def test_unknown_profile_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NERAIUM_PROFILE", raising=False)
    with pytest.raises(ValueError, match="Unknown profile"):
        load_settings("staging")
