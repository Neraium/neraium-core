from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neraium_core.markets.app.api import create_app
from neraium_core.markets.defaults import DEFAULT_LIVE_SYMBOLS, DEFAULT_LIVE_TIMEFRAME
from neraium_core.markets.integrations.massive.models import NormalizedMarketEvent
from neraium_core.markets.live.bar_builder import RollingBarBuilder
from neraium_core.markets.live.live_runner import LiveSessionRunner


def test_bar_aggregation():
    builder = RollingBarBuilder(timeframe="1m")
    ts = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)
    builder.add_event(NormalizedMarketEvent(ts, "SPY", 500.0, None, None, 100, "massive", "aggregate", "1m"))
    builder.add_event(NormalizedMarketEvent(ts, "SPY", 501.0, None, None, 120, "massive", "aggregate", "1m"))
    bars = builder.get_bars("SPY")
    assert len(bars) == 1
    assert bars[0]["high"] == 501.0


def test_timeframe_bucket_5m_correctness():
    builder = RollingBarBuilder(timeframe="5m")
    t1 = datetime(2026, 1, 1, 14, 31, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 14, 34, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 1, 14, 35, tzinfo=timezone.utc)
    builder.add_event(NormalizedMarketEvent(t1, "SPY", 500.0, None, None, 100, "massive", "aggregate", "5m"))
    builder.add_event(NormalizedMarketEvent(t2, "SPY", 501.0, None, None, 120, "massive", "aggregate", "5m"))
    builder.add_event(NormalizedMarketEvent(t3, "SPY", 502.0, None, None, 130, "massive", "aggregate", "5m"))
    bars = builder.get_bars("SPY")
    assert len(bars) == 2
    assert bars[0]["timestamp"].minute == 30
    assert bars[1]["timestamp"].minute == 35


def test_live_runner_warmup_progress_states():
    runner = LiveSessionRunner()
    runner.symbols = ["SPY", "QQQ", "IWM"]
    runner.state = runner.state.CONNECTING
    status = runner.status()
    assert status["readiness_state"] == "connecting"
    assert status["bars_collected"] == 0
    assert status["warmup_progress"] == 0.0


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
    payload = res.json()
    assert "session_state" in payload
    assert "bars_collected" in payload
    assert "bars_required" in payload
    assert "warmup_progress_pct" in payload


def test_massive_status_exposes_health_shape(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    app = create_app()
    client = TestClient(app)
    res = client.get("/integrations/massive/status")
    assert res.status_code == 200
    payload = res.json()
    assert "config_present" in payload
    assert "api_key_valid" in payload
    assert "rest_reachable" in payload
    assert "recent_live_event_at" in payload
    assert "recent_live_signal_at" in payload
    assert "session_state" in payload


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
        '[{"dataset_id":"dataset_a","provider":"massive","symbols":["SPY","QQQ","IWM"],"timeframe":"15m","start_date":"2026-01-01","end_date":"2026-01-02","created_at":"2026-01-02T00:00:00+00:00","dataset_path":"'
        + str(ds_path).replace("\\", "\\\\")
        + '"}]',
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NERAIUM_MARKETS_CACHE_PATH", str(cache_root))
        app = create_app(data_dir=tmp_path / "unused")
        client = TestClient(app)
        res = client.post("/run-replay", params={"timeframe": "15m", "use_massive_cached_data": "true"})
    assert res.status_code == 200
    assert res.json()["meta"]["data_dir"] == str(ds_path)


def test_run_replay_missing_configured_data_dir_returns_clear_error():
    app = create_app()
    app.state.data_dir = None
    client = TestClient(app)
    res = client.post("/run-replay", params={"timeframe": "15m"})
    assert res.status_code == 400
    assert "No replay data_dir configured" in res.json()["detail"]


def test_run_replay_invalid_data_dir_returns_clear_error(tmp_path: Path):
    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir(parents=True)
    app = create_app(data_dir=invalid_root)
    client = TestClient(app)
    res = client.post("/run-replay", params={"timeframe": "15m"})
    assert res.status_code == 400
    assert "Invalid replay data_dir" in res.json()["detail"]
    assert "missing CSV data for timeframe '15m'" in res.json()["detail"]


def test_live_start_rejects_missing_required_core_symbols():
    app = create_app()
    client = TestClient(app)
    res = client.post("/live/start", json={"symbols": ["SPY", "QQQ"], "timeframe": "1m"})
    assert res.status_code == 400
    assert "Live session requires core symbols" in res.json()["detail"]
    assert "Missing: IWM" in res.json()["detail"]


def test_live_start_default_body_uses_valid_defaults(monkeypatch):
    async def _fake_run(self):
        return None

    monkeypatch.setattr(LiveSessionRunner, "_run", _fake_run)
    monkeypatch.setenv("MASSIVE_API_KEY", "demo")
    app = create_app()
    client = TestClient(app)
    res = client.post("/live/start", json={})
    assert res.status_code == 200
    assert res.json()["status"] == "started"
    assert res.json()["symbols"] == DEFAULT_LIVE_SYMBOLS
    assert res.json()["timeframe"] == DEFAULT_LIVE_TIMEFRAME


def test_live_start_accepts_valid_required_symbols(monkeypatch):
    async def _fake_run(self):
        return None

    monkeypatch.setattr(LiveSessionRunner, "_run", _fake_run)
    monkeypatch.setenv("MASSIVE_API_KEY", "demo")
    app = create_app()
    client = TestClient(app)
    res = client.post("/live/start", json={"symbols": ["spy", "qqq", "iwm", "aapl"], "timeframe": "1m"})
    assert res.status_code == 200
    assert res.json()["status"] == "started"
    assert res.json()["symbols"] == ["SPY", "QQQ", "IWM", "AAPL"]


def test_live_status_defaults_to_first_run_values():
    app = create_app()
    client = TestClient(app)
    payload = client.get("/live/status").json()
    assert payload["timeframe"] == DEFAULT_LIVE_TIMEFRAME


def test_operator_summary_includes_launch_checklist():
    app = create_app()
    client = TestClient(app)
    payload = client.get("/operator/summary").json()
    assert "launch_checklist" in payload
    assert any(item["key"] == "core_symbols_present" for item in payload["launch_checklist"])


def test_history_filters_include_suppressed():
    app = create_app()
    client = TestClient(app)
    history = client.get("/signals/history", params={"include_suppressed": "true", "session_type": "live"})
    assert history.status_code == 200
    assert "summary" in history.json()


def test_ui_route_serves_operator_console():
    app = create_app()
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "Command Center" in res.text
    assert "SPY,QQQ,IWM,AAPL,NVDA" in res.text
    assert "5m" in res.text


def test_static_assets_served():
    app = create_app()
    client = TestClient(app)
    res = client.get("/static/operator.js")
    assert res.status_code == 200


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
