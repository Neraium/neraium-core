from __future__ import annotations

import json
from pathlib import Path

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
    monkeypatch.setattr(hardened_cli, "run_structural_pipeline", lambda _payloads, config=None: {"validation_results": {}})


def test_hardened_cli_exit_code_all_invalid_records(tmp_path: Path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "all_invalid.json"
    output_path = tmp_path / "out.json"
    input_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    _patch_hardened_dependencies(monkeypatch)
    monkeypatch.setattr("sys.argv", ["sii_cli", "--input", str(input_path), "--output", str(output_path)])

    code = hardened_cli.main()
    summary = json.loads(capsys.readouterr().out)

    assert code == 1
    assert summary["frames_succeeded"] == 0
    assert summary["frames_failed"] == 3
    assert summary["all_failed"] is True
    assert summary["ingest_errors"]


def test_compatibility_entrypoint_delegates_configuration_errors(tmp_path: Path, monkeypatch, capsys) -> None:
    missing_path = tmp_path / "missing.json"
    output_path = tmp_path / "out.json"
    _patch_hardened_dependencies(monkeypatch)

    code = compatibility_cli.main(["--input", str(missing_path), "--output", str(output_path)])
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert "error" in payload
