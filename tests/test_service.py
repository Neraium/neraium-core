import subprocess
import sys

from neraium_core.service import StructuralIntelligenceService


def test_service_ingest_payload_and_latest_result():
    service = StructuralIntelligenceService()
    payload = {
        "timestamp": "2026-01-01T00:00:00Z",
        "site_id": "site",
        "asset_id": "asset",
        "sensor_values": {"x": 1.0, "y": 2.0},
    }

    result = service.ingest_payload(payload)
    assert result == service.latest_result()


def test_service_ingest_csv_and_reset():
    service = StructuralIntelligenceService()
    csv_text = (
        "timestamp,site_id,asset_id,x,y\n"
        "2026-01-01T00:00:00Z,s,a,1.0,2.0\n"
        "2026-01-01T00:01:00Z,s,a,1.1,2.1\n"
    )

    results = service.ingest_csv(csv_text)
    assert len(results) == 2
    service.reset()
    assert service.latest_result() is None


def test_fd004_script_requires_input_arg_or_env():
    process = subprocess.run(
        [sys.executable, "scripts/run_fd004_test.py"],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "Provide --input or set FD004_PATH" in process.stderr
