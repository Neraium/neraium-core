from __future__ import annotations

from pathlib import Path

from run_live_stock_market import _append_csv_row, _build_connector, parse_args


def test_parse_args_supports_required_live_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_live_stock_market.py",
            "--tickers",
            "AAPL,MSFT",
            "--interval",
            "2",
            "--output",
            "logs/live.csv",
            "--mock",
            "--max-iterations",
            "1",
        ],
    )
    args = parse_args()
    assert args.tickers == "AAPL,MSFT"
    assert args.interval == 2.0
    assert args.output == "logs/live.csv"
    assert args.mock is True


def test_build_connector_uses_mock_shortcut(monkeypatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    args = type(
        "Args",
        (),
        {
            "mock": True,
            "provider": "polygon",
            "api_key": None,
            "provider_interval": "1min",
        },
    )()
    connector = _build_connector(args)
    assert connector.__class__.__name__ == "MockLiveConnector"


def test_append_csv_row_creates_header_once(tmp_path: Path) -> None:
    output_path = tmp_path / "signals.csv"
    row = {
        "timestamp": "2026-04-02T00:00:00+00:00",
        "ticker": "AAPL",
        "state": "NORMAL",
        "trading_signal": "HOLD",
        "structural_drift_score": 0.2,
        "latest_instability": 0.2,
        "system_health": 96.0,
        "evidence_confidence": 0.7,
    }
    _append_csv_row(str(output_path), row)
    _append_csv_row(str(output_path), row)

    text = output_path.read_text(encoding="utf-8")
    assert text.count("timestamp,ticker,state,trading_signal") == 1
    assert text.count("AAPL") == 2
