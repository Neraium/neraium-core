"""Day 12: safe runtime and fallbacks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neraium.runtime import (
    atomic_write_json,
    build_run_status_payload,
    ensure_output_layout,
    fallback_to_last_good_output,
    save_run_status,
    safe_run_pipeline,
)
from neraium.settings import load_settings


def test_safe_run_pipeline_success() -> None:
    out = safe_run_pipeline(lambda: 42, stage="t")
    assert out["success"] is True
    assert out["error_message"] is None
    assert out["result"] == 42
    assert out["health_status"] == "ok"


def test_safe_run_pipeline_failure() -> None:
    def _boom() -> None:
        raise RuntimeError("planned")

    out = safe_run_pipeline(_boom, stage="t")
    assert out["success"] is False
    assert "planned" in (out["error_message"] or "")
    assert out["stage_failed"] == "t"
    assert out["health_status"] == "degraded"


def test_fallback_to_last_good_output(tmp_path: Path) -> None:
    latest = tmp_path / "latest"
    latest.mkdir(parents=True)
    status = {
        "outputs_written": {"a": str(tmp_path / "x.txt")},
        "manifest_path": str(tmp_path / "manifests" / "m.json"),
    }
    (latest / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
    fb = fallback_to_last_good_output(tmp_path)
    assert fb["best_path"] == str(tmp_path / "x.txt")


def test_run_status_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NERAIUM_OUTPUT_DIR", str(tmp_path / "out"))
    s = load_settings("test")
    ensure_output_layout(s)
    payload = build_run_status_payload(
        run_id="r1",
        mode="batch",
        success=True,
        settings=s,
        manifest_path=str(tmp_path / "m.json"),
        outputs_written={"k": "v"},
        error_message=None,
        health_status="ok",
    )
    p = save_run_status(s, payload)
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["run_id"] == "r1"
    assert data["manifest_path"] == str(tmp_path / "m.json")


def test_output_paths_consistent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NERAIUM_OUTPUT_DIR", str(tmp_path / "o"))
    s = load_settings("dev")
    ensure_output_layout(s)
    for name in ("latest", "runs", "manifests", "alerts", "evidence", "reviews", "health"):
        assert (s.output_dir / name).is_dir()


def test_atomic_write_json(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    atomic_write_json(p, {"x": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"x": 1}
