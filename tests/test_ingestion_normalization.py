from __future__ import annotations

import pytest

from neraium_core.ingestion_normalization import (
    normalize_canonical_records_payload,
    normalize_external_batch_payload,
    normalize_external_payload,
)


def test_normalize_json_alias_payload() -> None:
    frame = normalize_external_payload(
        {
            "recorded_at": "2026-01-01T00:00:00Z",
            "entity_id": "pump-7",
            "location": "west-plant",
            "variables": {"temp_sensor_A": "72.1", "vib_x": 0.82},
        },
        customer_id="c1",
    )
    assert frame["asset_id"] == "pump-7"
    assert frame["site_id"] == "west-plant"
    assert frame["sensor_values"]["temp_sensor_A"] == pytest.approx(72.1)


def test_normalize_direct_canonical_payload() -> None:
    frame = normalize_external_payload(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "asset_id": "asset-1",
            "site_id": "site-1",
            "sensor_values": {"pressure": 12.5},
        }
    )
    assert frame["asset_id"] == "asset-1"
    assert frame["sensor_values"]["pressure"] == pytest.approx(12.5)


def test_batch_payload_with_mapping_and_stream_shape() -> None:
    rows, _review = normalize_external_batch_payload(
        {
            "mapping": {
                "timestamp": "recorded",
                "asset_id": "machine",
                "site_id": "plant",
                "signal_aliases": {"temp_sensor_A": "temperature"},
            },
            "stream": {
                "records": [
                    {
                        "recorded": "2026-01-01T00:00:00+00:00",
                        "machine": "m-1",
                        "plant": "p-1",
                        "measurements": {"temp_sensor_A": 21.0},
                    }
                ]
            },
        }
    )
    assert len(rows) == 1
    assert rows[0]["sensor_values"]["temperature"] == pytest.approx(21.0)


def test_missing_required_fields_raise_structured_errors() -> None:
    with pytest.raises(ValueError, match="missing_timestamp"):
        normalize_external_payload({"asset_id": "asset-1", "sensor_values": {"x": 1}})
    with pytest.raises(ValueError, match="no_usable_signals"):
        normalize_external_payload({"timestamp": "2026-01-01T00:00:00+00:00", "asset_id": "asset-1"})


def test_json_array_payload_normalizes() -> None:
    rows, _ = normalize_external_batch_payload(
        [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "asset_id": "a-1",
                "signals": {"p": 1.1},
            }
        ],
        customer_id="c1",
    )
    assert len(rows) == 1
    assert rows[0]["asset_id"] == "a-1"


def test_direct_canonical_records_payload() -> None:
    rows = normalize_canonical_records_payload(
        {
            "records": [
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "asset_id": "a-1",
                    "site_id": "s-1",
                    "signals": {"pressure": 10.2},
                }
            ]
        },
        customer_id="c1",
    )
    assert len(rows) == 1
    assert rows[0]["sensor_values"]["pressure"] == pytest.approx(10.2)


def test_ambiguous_aliases_require_mapping() -> None:
    with pytest.raises(ValueError, match="ambiguous_mapping"):
        normalize_external_payload(
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "time": "2026-01-01T00:00:01+00:00",
                "asset_id": "asset-1",
                "signals": {"x": 1},
            }
        )


def test_batch_mixed_valid_invalid_rows_reports_row_index() -> None:
    with pytest.raises(ValueError, match="batch_item_2"):
        normalize_external_batch_payload(
            {
                "items": [
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "asset_id": "a-1",
                        "signals": {"p": 1.0},
                    },
                    {
                        "timestamp": "2026-01-01T00:01:00+00:00",
                        "asset_id": "a-2",
                        "signals": {},
                    },
                ]
            }
        )


def test_malformed_batch_payload_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported_payload_shape"):
        normalize_external_batch_payload({"items": "bad-shape"})


def test_missing_asset_id_strict_error() -> None:
    with pytest.raises(ValueError, match="missing_asset_id"):
        normalize_external_payload({"timestamp": "2026-01-01T00:00:00+00:00", "signals": {"x": 1}})
