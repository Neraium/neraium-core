from __future__ import annotations

from datetime import datetime, timezone

from neraium_core.ingestion_normalization import normalize_external_payload
from neraium_core.pipeline import parse_csv_text
from neraium_core.stock_market_adapter import build_stock_frame


def _required_ingress_keys() -> set[str]:
    return {"timestamp", "customer_id", "site_id", "asset_id", "sensor_values", "sensor_quality", "aligned", "anomaly"}


def test_raw_csv_and_live_market_frames_share_engine_ingress_contract() -> None:
    epoch = 1712055000
    expected_ts = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    raw_frame = normalize_external_payload(
        {
            "timestamp": expected_ts,
            "customer_id": "cust-1",
            "site_id": "site-1",
            "asset_id": "AAPL",
            "signals": {"open": 123.0, "close": 124.0},
        }
    )
    csv_frame = parse_csv_text(
        "timestamp,site_id,asset_id,open,close\n"
        f"{expected_ts},site-1,AAPL,123.0,124.0\n",
        customer_id="cust-1",
    )[0]
    live_frame = build_stock_frame(
        timestamp=epoch,
        ticker="AAPL",
        row_dict={"open": 123.0, "close": 124.0, "note": "ignored"},
    )

    for frame in (raw_frame, csv_frame, live_frame):
        assert _required_ingress_keys().issubset(frame.keys())
        assert datetime.fromisoformat(frame["timestamp"]).tzinfo is not None
        assert isinstance(frame["sensor_values"], dict)
        assert isinstance(frame["sensor_quality"], dict)

    assert raw_frame["timestamp"] == expected_ts
    assert csv_frame["timestamp"] == expected_ts
    assert live_frame["timestamp"] == expected_ts
