"""Day 12: run manifests."""

from __future__ import annotations

from pathlib import Path

from neraium.manifests import build_run_manifest, save_run_manifest
from neraium.settings import load_settings


def test_manifest_created() -> None:
    s = load_settings("dev")
    m = build_run_manifest(
        s,
        mode="batch",
        output_paths={"evidence": s.evidence_dir / "evidence_log.jsonl"},
        run_id="test-run-1",
        success=True,
    )
    required = {
        "run_id",
        "timestamp",
        "profile",
        "mode",
        "timeframes_used",
        "assets_used",
        "output_files",
        "settings_snapshot",
        "success",
    }
    for k in required:
        assert k in m


def test_manifest_saved(tmp_path: Path) -> None:
    s = load_settings("dev")
    m = build_run_manifest(s, mode="batch", output_paths={}, run_id="x", success=True)
    p = tmp_path / "m.json"
    save_run_manifest(m, p)
    assert p.is_file()
    assert "run_id" in p.read_text(encoding="utf-8")
