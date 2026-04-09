from __future__ import annotations

import json

from neraium_core.sii import cli as compatibility_cli
from neraium_core import sii_cli as hardened_cli


class _DummyLogger:
    def info(self, *_args, **_kwargs) -> None:
        return None

    def error(self, *_args, **_kwargs) -> None:
        return None


class _DummyEngine:
    def close(self) -> None:
        return None


class _DummyApp:
    def __init__(self) -> None:
        self.engine = _DummyEngine()

    def run_payloads(self, payloads):
        return [{"frame_id": i} for i, _ in enumerate(payloads)]

    def write_output_file(self, output_path, outputs) -> None:
        output_path.write_text(json.dumps(outputs), encoding="utf-8")


def _patch_hardened_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(hardened_cli, "configure_structured_logging", lambda _level: _DummyLogger())
    monkeypatch.setattr(hardened_cli.SIIApplication, "from_config", lambda _config: _DummyApp())
    monkeypatch.setattr(
        hardened_cli,
        "run_structural_pipeline",
        lambda _payloads, config=None: {"validation_results": {}},
    )


def test_cli_entrypoints_share_success_contract(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "input.json"
    output_a = tmp_path / "out_a.json"
    output_b = tmp_path / "out_b.json"
    input_path.write_text(
        json.dumps([
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "site_id": "site-a",
                "asset_id": "asset-a",
                "sensor_values": {"temperature": 10.0},
            }
        ]),
        encoding="utf-8",
    )

    _patch_hardened_dependencies(monkeypatch)

    monkeypatch.setattr("sys.argv", ["sii_cli", "--input", str(input_path), "--output", str(output_a)])
    hardened_code = hardened_cli.main()
    hardened_summary = json.loads(capsys.readouterr().out)

    compatibility_code = compatibility_cli.main(["--input", str(input_path), "--output", str(output_b)])
    compatibility_summary = json.loads(capsys.readouterr().out)

    assert hardened_code == 0
    assert compatibility_code == hardened_code
    assert compatibility_summary == {
        **hardened_summary,
        "output_path": str(output_b),
    }


def test_cli_entrypoints_share_partial_failure_contract(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "bad.json"
    output_a = tmp_path / "out_a.json"
    output_b = tmp_path / "out_b.json"
    input_path.write_text(
        json.dumps([
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "site_id": "site-a",
                "asset_id": "asset-a",
                "sensor_values": {"temperature": 10.0},
            },
            "not-an-object",
        ]),
        encoding="utf-8",
    )

    _patch_hardened_dependencies(monkeypatch)

    monkeypatch.setattr("sys.argv", ["sii_cli", "--input", str(input_path), "--output", str(output_a)])
    hardened_code = hardened_cli.main()
    hardened_summary = json.loads(capsys.readouterr().out)

    compatibility_code = compatibility_cli.main(["--input", str(input_path), "--output", str(output_b)])
    compatibility_summary = json.loads(capsys.readouterr().out)

    assert hardened_code == 1
    assert compatibility_code == hardened_code
    assert compatibility_summary == {
        **hardened_summary,
        "output_path": str(output_b),
    }


def test_hardened_cli_uses_runtime_input_path_fallback(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "runtime_input.json"
    output_path = tmp_path / "out.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "site_id": "site-a",
                    "asset_id": "asset-a",
                    "sensor_values": {"temperature": 10.0},
                }
            ]
        ),
        encoding="utf-8",
    )

    _patch_hardened_dependencies(monkeypatch)
    monkeypatch.setenv("NERAIUM_INPUT_PATH", str(input_path))
    monkeypatch.setattr("sys.argv", ["sii_cli", "--output", str(output_path)])

    code = hardened_cli.main()
    summary = json.loads(capsys.readouterr().out)

    assert code == 0
    assert summary["frames_processed"] == 1
    assert summary["frames_succeeded"] == 1
    assert summary["output_path"] == str(output_path)


def test_hardened_cli_rejects_missing_output_directory(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "missing-dir" / "out.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "site_id": "site-a",
                    "asset_id": "asset-a",
                    "sensor_values": {"temperature": 10.0},
                }
            ]
        ),
        encoding="utf-8",
    )

    _patch_hardened_dependencies(monkeypatch)
    monkeypatch.setattr("sys.argv", ["sii_cli", "--input", str(input_path), "--output", str(output_path)])

    code = hardened_cli.main()
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert "Output directory does not exist" in str(payload.get("error"))
