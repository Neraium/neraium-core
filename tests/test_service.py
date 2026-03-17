import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

from neraium_core.service import StructuralIntelligenceService
from scripts.run_fd004_test import run_fd004


def test_service_ingest_payload_and_latest() -> None:
    service = StructuralIntelligenceService()
    result = service.ingest_payload(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "site_id": "site",
            "asset_id": "asset",
            "sensor_values": {"s1": "2.5", "s2": "bad"},
        }
    )
    assert result["state"] == "STABLE"
    assert service.latest_result() == result


def test_service_ingest_csv_and_reset() -> None:
    service = StructuralIntelligenceService()
    rows = service.ingest_csv(
        "timestamp,site_id,asset_id,s1\n"
        "2026-01-01T00:00:00Z,site,asset,1\n"
        "2026-01-01T00:01:00Z,site,asset,2\n"
    )
    assert len(rows) == 2
    service.reset()
    assert service.latest_result() is None


def test_fd004_script_accepts_paths(tmp_path) -> None:
    input_file = tmp_path / "fd004.txt"
    output_file = tmp_path / "results.csv"
    input_file.write_text("1 1 0 0 0 " + " ".join(["1"] * 21) + "\n")

    run_fd004(str(input_file), str(output_file))

    assert output_file.exists()
