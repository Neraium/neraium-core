from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neraium_core.markets.app.api import create_app
from neraium_core.markets.integrations.massive.models import NormalizedMarketEvent
from neraium_core.markets.live.bar_builder import RollingBarBuilder


def test_bar_aggregation():
    builder = RollingBarBuilder(timeframe="1m")
    ts = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)
    builder.add_event(NormalizedMarketEvent(ts, "SPY", 500.0, None, None, 100, "massive", "aggregate", "1m"))
    builder.add_event(NormalizedMarketEvent(ts, "SPY", 501.0, None, None, 120, "massive", "aggregate", "1m"))
    bars = builder.get_bars("SPY")
    assert len(bars) == 1
    assert bars[0]["high"] == 501.0


def test_missing_config_returns_400(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    app = create_app()
    client = TestClient(app)
    res = client.post("/integrations/massive/historical/fetch", json={"symbols": ["SPY"], "timeframe": "1m", "start_date": "2026-01-01", "end_date": "2026-01-02"})
    assert res.status_code == 400


def test_live_status_endpoint():
    app = create_app()
    client = TestClient(app)
    res = client.get("/live/status")
    assert res.status_code == 200
    assert "session_state" in res.json()


def test_replay_over_cached_data(tmp_path: Path):
    ds = tmp_path / "15m"
    ds.mkdir(parents=True)
    for ticker in ["SPY", "QQQ", "IWM"]:
        lines = ["timestamp,open,high,low,close,volume"]
        for i in range(40):
            lines.append(f"2026-01-01T00:{i:02d}:00+00:00,100,101,99,{100+i*0.1},1000")
        (ds / f"{ticker}.csv").write_text("\n".join(lines), encoding="utf-8")
    app = create_app(data_dir=tmp_path)
    client = TestClient(app)
    res = client.post("/run-replay", params={"timeframe": "15m", "data_dir": str(tmp_path)})
    assert res.status_code == 200
    assert "run_id" in res.json()


def test_run_replay_uses_configured_data_dir_by_default(tmp_path: Path):
    data_root = tmp_path / "configured"
    bars_dir = data_root / "15m"
    bars_dir.mkdir(parents=True)
    for ticker in ["SPY", "QQQ", "IWM"]:
        rows = ["timestamp,open,high,low,close,volume"]
        for i in range(45):
            rows.append(f"2026-01-01T00:{i:02d}:00+00:00,100,101,99,{100 + i * 0.1},1000")
        (bars_dir / f"{ticker}.csv").write_text("\n".join(rows), encoding="utf-8")
    app = create_app(data_dir=data_root)
    client = TestClient(app)
    res = client.post("/run-replay", params={"timeframe": "15m"})
    assert res.status_code == 200
    assert res.json()["meta"]["data_dir"] == str(data_root)


def test_run_replay_explicit_data_dir_still_wins(tmp_path: Path):
    default_dir = tmp_path / "default"
    explicit_dir = tmp_path / "explicit"
    for root in [default_dir, explicit_dir]:
        bars_dir = root / "15m"
        bars_dir.mkdir(parents=True)
        for ticker in ["SPY", "QQQ", "IWM"]:
            rows = ["timestamp,open,high,low,close,volume"]
            for i in range(45):
                rows.append(f"2026-01-01T00:{i:02d}:00+00:00,100,101,99,{100 + i * 0.1},1000")
            (bars_dir / f"{ticker}.csv").write_text("\n".join(rows), encoding="utf-8")
    app = create_app(data_dir=default_dir)
    client = TestClient(app)
    res = client.post("/run-replay", params={"timeframe": "15m", "data_dir": str(explicit_dir)})
    assert res.status_code == 200
    assert res.json()["meta"]["data_dir"] == str(explicit_dir)


def test_run_replay_respects_use_massive_cached_data(tmp_path: Path):
    cache_root = tmp_path / "cache"
    ds_path = cache_root / "dataset_a"
    ds_bars = ds_path / "15m"
    ds_bars.mkdir(parents=True)
    for ticker in ["SPY", "QQQ", "IWM"]:
        rows = ["timestamp,open,high,low,close,volume,source,timeframe"]
        for i in range(45):
            rows.append(f"2026-01-01T00:{i:02d}:00+00:00,100,101,99,{100 + i * 0.1},1000,massive,15m")
        (ds_bars / f"{ticker}.csv").write_text("\n".join(rows), encoding="utf-8")
    (cache_root / "datasets.json").write_text(
        '[{\"dataset_id\":\"dataset_a\",\"provider\":\"massive\",\"symbols\":[\"SPY\",\"QQQ\",\"IWM\"],\"timeframe\":\"15m\",\"start_date\":\"2026-01-01\",\"end_date\":\"2026-01-02\",\"created_at\":\"2026-01-02T00:00:00+00:00\",\"dataset_path\":\"'
        + str(ds_path).replace("\\", "\\\\")
        + '\"}]',
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NERAIUM_MARKETS_CACHE_PATH", str(cache_root))
        app = create_app(data_dir=tmp_path / "unused")
        client = TestClient(app)
        res = client.post("/run-replay", params={"timeframe": "15m", "use_massive_cached_data": "true"})
    assert res.status_code == 200
    assert res.json()["meta"]["data_dir"] == str(ds_path)


def test_live_start_rejects_missing_required_core_symbols():
    app = create_app()
    client = TestClient(app)
    res = client.post("/live/start", json={"symbols": ["SPY", "QQQ"], "timeframe": "1m"})
    assert res.status_code == 400
    assert "Missing required core symbols" in res.json()["detail"]
    assert "IWM" in res.json()["detail"]


def test_live_start_accepts_valid_required_symbols(monkeypatch):
    from neraium_core.markets.live.live_runner import LiveSessionRunner

    async def _fake_run(self):
        return None

    monkeypatch.setattr(LiveSessionRunner, "_run", _fake_run)
    monkeypatch.setenv("MASSIVE_API_KEY", "demo")
    app = create_app()
    client = TestClient(app)
    res = client.post("/live/start", json={"symbols": ["SPY", "QQQ", "IWM"], "timeframe": "1m"})
    assert res.status_code == 200
    assert res.json()["status"] == "started"


def test_ui_route_serves_operator_console():
    app = create_app()
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "Command Center" in res.text


def test_operator_helper_endpoints_shape():
    app = create_app()
    client = TestClient(app)
    summary = client.get("/operator/summary")
    history = client.get("/signals/history")
    datasets = client.get("/integrations/massive/datasets")
    assert summary.status_code == 200
    assert "live_status" in summary.json()
    assert history.status_code == 200
    assert "signals" in history.json()
    assert datasets.status_code == 200
    assert "datasets" in datasets.json()
