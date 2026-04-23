"""Day 12: CLI argument parsing."""

from __future__ import annotations

import pytest

from neraium.cli import SUBCOMMANDS, _build_parser, main


def test_cli_commands_parse() -> None:
    p = _build_parser()
    ns = p.parse_args(["run-batch", "--profile", "test"])
    assert ns.command == "run-batch"
    assert ns.profile == "test"

    ns2 = p.parse_args(["run-realtime", "--profile", "prod_local", "--max-cycles", "3"])
    assert ns2.command == "run-realtime"
    assert ns2.max_cycles == 3

    ns3 = p.parse_args(["health-check", "--output-dir", "/tmp/o"])
    assert ns3.command == "health-check"

    ns4 = p.parse_args(["inspect-sources", "--profile", "dev"])
    assert ns4.command == "inspect-sources"


def test_subcommands_known() -> None:
    assert "run-batch" in SUBCOMMANDS
    assert "rebuild-outputs" in SUBCOMMANDS
    assert "ingest-preview" in SUBCOMMANDS


def test_main_health_check_missing_output(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("NERAIUM_OUTPUT_DIR", str(tmp_path))
    code = main(["health-check", "--profile", "dev"])
    assert code == 1
