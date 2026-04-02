from __future__ import annotations

import json
from pathlib import Path

import pytest

from neraium_core.sii import SIIApplication, SIIConfig
from neraium_core.sii.live_ingestion import (
    LiveIngestionConfig,
    LiveIngestionRunner,
    LiveIngestionBatch,
    USGSLiveProviderConfig,
    USGSLiveTelemetryProvider,
)


class _StubUSGSProvider(USGSLiveTelemetryProvider):
    def __init__(self, body: dict):
        cfg = USGSLiveProviderConfig(
            base_url="https://waterservices.usgs.gov/nwis/iv/",
            sites=("01646500",),
            parameter_codes=("00060",),
            period="P1D",
            start_dt=None,
            end_dt=None,
            extra_query_params={},
            request_timeout_seconds=2.0,
            request_retries=0,
            request_backoff_seconds=0.0,
            asset_prefix="usgs_site_",
            max_records=500,
        )
        super().__init__(cfg)
        self._body = body

    def fetch(self) -> LiveIngestionBatch:
        return self._normalize(
            self._body,
            request_url="https://waterservices.usgs.gov/nwis/iv/?format=json",
            request_params={"format": "json"},
        )


def _minimal_usgs_payload() -> dict:
    return {
        "name": "ns1:timeSeriesResponseType",
        "value": {
            "timeSeries": [
                {
                    "sourceInfo": {"siteCode": [{"value": "01646500"}]},
                    "variable": {
                        "variableCode": [{"value": "00060"}],
                        "variableName": "Discharge, cubic feet per second",
                    },
                    "values": [
                        {
                            "value": [
                                {"value": "123.4", "dateTime": "2026-03-21T00:00:00.000Z"},
                                {"value": "124.0", "dateTime": "2026-03-21T00:15:00.000Z"},
                            ]
                        }
                    ],
                }
            ]
        },
    }


def test_usgs_provider_normalizes_payload() -> None:
    provider = _StubUSGSProvider(_minimal_usgs_payload())
    batch = provider.fetch()
    assert batch.provider_name == "usgs_nwis_iv"
    assert batch.normalized_record_count == 2
    assert batch.raw_record_count == 2
    assert len(batch.canonical_records) == 2
    assert batch.detected_nodes == ["01646500"]
    assert "00060" in batch.detected_sensors
    assert batch.payloads[0]["site_id"] == "01646500"
    assert "00060" in batch.payloads[0]["sensor_values"]
    assert "00060" in batch.canonical_records[0].variables


def test_live_ingestion_runner_handles_empty_without_raising() -> None:
    provider = _StubUSGSProvider({"value": {"timeSeries": []}})
    runner = LiveIngestionRunner(provider=provider)
    batches = runner.fetch_batches(max_polls=1, poll_interval_seconds=1.0, raise_on_empty_response=False)
    assert len(batches) == 1
    assert batches[0].normalized_record_count == 0


def test_live_ingestion_runner_raises_on_empty_when_requested() -> None:
    provider = _StubUSGSProvider({"value": {"timeSeries": []}})
    runner = LiveIngestionRunner(provider=provider)
    with pytest.raises(Exception):
        runner.fetch_batches(max_polls=1, poll_interval_seconds=1.0, raise_on_empty_response=True)


def test_sii_app_live_poll_path_processes_frames(tmp_path: Path) -> None:
    config = SIIConfig(
        baseline_window=8,
        recent_window=4,
        max_history=64,
        live_ingestion_enabled=True,
        live_ingestion_provider="usgs_nwis_iv",
        live_ingestion_site_ids="01646500",
        live_ingestion_parameter_codes="00060",
        live_ingestion_poll_interval_seconds=1.0,
        live_ingestion_extra_query_json=json.dumps({"period": "P1D"}),
        regime_store_path=str(tmp_path / "sii_regimes.json"),
    )
    app = SIIApplication.from_config(config)

    # Seed warmup with regular payloads first.
    seed_payloads = []
    for idx in range(10):
        seed_payloads.append(
            {
                "timestamp": str(float(idx)),
                "site_id": "site-A",
                "asset_id": "asset-A",
                "sensor_values": {"s1": float(idx), "s2": float(idx) * 0.9, "s3": float(idx) * 1.1},
            }
        )
    _ = app.run_payloads(seed_payloads)

    # Patch app to run with local stub provider.
    provider = _StubUSGSProvider(_minimal_usgs_payload())
    runner = LiveIngestionRunner(provider=provider)
    batches = runner.fetch_batches(max_polls=1, poll_interval_seconds=1.0, raise_on_empty_response=False)
    outputs = app.run_payloads(batches[0].payloads)
    assert outputs
    assert outputs[-1]["state"] in {"STABLE", "WATCH", "ALERT"}
