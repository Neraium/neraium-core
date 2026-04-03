from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
