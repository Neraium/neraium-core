from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="Neraium UI")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _find_fd004_dir() -> Path | None:
    candidates = [
        BASE_DIR / "fd004_outputs_subset",
        BASE_DIR / "data" / "fd004_outputs_subset",
        BASE_DIR / "outputs" / "fd004_outputs_subset",
        BASE_DIR / "neraium_ui" / "fd004_outputs_subset",
    ]

    for path in candidates:
        if (path / "fd004_real_report.json").exists():
            return path

    return None


FD004_DIR = _find_fd004_dir()
REPORT_PATH = FD004_DIR / "fd004_real_report.json" if FD004_DIR else None
TIMESERIES_CSV_PATH = FD004_DIR / "hero_unit_timeseries.csv" if FD004_DIR else None


def _normalize_mode(mode: str | None) -> str:
    if mode and mode.lower() in {"stable", "unstable", "real"}:
        return mode.lower()
    return "stable"


def _mock_status(mode: str) -> dict[str, Any]:
    if mode == "unstable":
        return {
            "phase": "degrading",
            "risk_level": "MEDIUM",
            "signal_emitted": True,
            "signal_strength": "medium",
            "confidence": "medium",
            "operator_message": (
                "Observed structural patterns indicate deviation from a stable regime "
                "with emerging instability under current analysis."
            ),
            "latest_drift": 0.24,
            "latest_instability": 0.28,
            "regime_name": "regime_demo_unstable",
            "regime_distance": 1.42,
            "regime_drift": 0.91,
        }

    return {
        "phase": "stable",
        "risk_level": "LOW",
        "signal_emitted": False,
        "signal_strength": "low",
        "confidence": "low",
        "operator_message": (
            "Observed structural patterns are consistent with a stable regime under "
            "current analysis."
        ),
        "latest_drift": 0.10,
        "latest_instability": 0.09,
        "regime_name": "regime_demo_stable",
        "regime_distance": 0.18,
        "regime_drift": 0.07,
    }


def _mock_timeseries(mode: str) -> list[dict[str, Any]]:
    if mode == "unstable":
        return [
            {"timestamp": 1, "drift": 0.03, "instability": 0.05, "regime_drift": 0.02},
            {"timestamp": 2, "drift": 0.05, "instability": 0.08, "regime_drift": 0.05},
            {"timestamp": 3, "drift": 0.09, "instability": 0.12, "regime_drift": 0.12},
            {"timestamp": 4, "drift": 0.16, "instability": 0.19, "regime_drift": 0.24},
            {"timestamp": 5, "drift": 0.24, "instability": 0.28, "regime_drift": 0.36},
        ]

    return [
        {"timestamp": 1, "drift": 0.02, "instability": 0.03, "regime_drift": 0.01},
        {"timestamp": 2, "drift": 0.03, "instability": 0.04, "regime_drift": 0.01},
        {"timestamp": 3, "drift": 0.05, "instability": 0.05, "regime_drift": 0.02},
        {"timestamp": 4, "drift": 0.08, "instability": 0.07, "regime_drift": 0.03},
        {"timestamp": 5, "drift": 0.10, "instability": 0.09, "regime_drift": 0.04},
    ]


def _load_report() -> dict[str, Any] | None:
    if not REPORT_PATH or not REPORT_PATH.exists():
        return None

    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _data_source(mode: str) -> dict[str, Any]:
    is_real = (
        mode == "real"
        and FD004_DIR is not None
        and REPORT_PATH is not None
        and REPORT_PATH.exists()
    )

    return {
        "mode": mode if is_real else ("stable" if mode == "real" else mode),
        "source": "fd004" if is_real else "mock",
        "fd004_dir": str(FD004_DIR) if FD004_DIR else None,
        "using_mock": not is_real,
    }


def _get_latest_unit_summary(mode: str) -> dict[str, Any]:
    source_info = _data_source(mode)

    if source_info["source"] != "fd004":
        return _mock_status(source_info["mode"])

    report = _load_report()
    if not report:
        return _mock_status("stable")

    unit_summaries = report.get("unit_summaries", [])
    if not unit_summaries:
        return _mock_status("stable")

    summary = unit_summaries[0]
    return {
        "phase": summary.get("phase", "unknown"),
        "risk_level": summary.get("risk_level", "UNKNOWN"),
        "signal_emitted": bool(summary.get("signal_emitted", False)),
        "signal_strength": summary.get("signal_strength", "low"),
        "confidence": summary.get("confidence", "low"),
        "operator_message": summary.get(
            "operator_message",
            _mock_status("stable")["operator_message"],
        ),
        "latest_drift": float(summary.get("latest_drift", summary.get("structural_drift_score", 0.0))),
        "latest_instability": float(summary.get("latest_instability", 0.0)),
        "regime_name": summary.get("regime_name"),
        "regime_distance": summary.get("regime_distance"),
        "regime_drift": float(summary.get("regime_drift", 0.0)),
    }


def _load_timeseries(mode: str) -> list[dict[str, Any]]:
    source_info = _data_source(mode)

    if source_info["source"] == "fd004":
        if TIMESERIES_CSV_PATH and TIMESERIES_CSV_PATH.exists():
            try:
                rows: list[dict[str, Any]] = []
                with TIMESERIES_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(
                            {
                                "timestamp": int(float(row.get("timestamp", 0))),
                                "drift": float(row.get("drift", 0.0)),
                                "instability": float(row.get("instability", 0.0)),
                                "regime_drift": float(row.get("regime_drift", 0.0)),
                            }
                        )
                if rows:
                    return rows
            except (OSError, ValueError):
                pass

        report = _load_report()
        if report:
            timeseries = report.get("timeseries", [])
            if timeseries:
                unit_id = report.get("hero_unit")
                filtered = [
                    {
                        "timestamp": int(float(item.get("timestamp", 0))),
                        "drift": float(item.get("structural_drift_score", 0.0)),
                        "instability": float(item.get("experimental_analytics", {}).get("composite_instability", 0.0)),
                        "regime_drift": float(item.get("experimental_analytics", {}).get("regime_drift", 0.0)),
                    }
                    for item in timeseries
                    if unit_id is None or item.get("unit_id") == unit_id
                ]
                if filtered:
                    return filtered

    return _mock_timeseries(source_info["mode"])


@app.get("/", response_class=HTMLResponse)
def index(request: Request, mode: str = Query("stable")) -> HTMLResponse:
    normalized_mode = _normalize_mode(mode)
    source_info = _data_source(normalized_mode)
    status = _get_latest_unit_summary(normalized_mode)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "status": status,
            "mode": source_info["mode"],
            "source_info": source_info,
        },
    )


@app.get("/status")
def status(mode: str = Query("stable")) -> JSONResponse:
    normalized_mode = _normalize_mode(mode)
    return JSONResponse(_get_latest_unit_summary(normalized_mode))


@app.get("/timeseries")
def timeseries(mode: str = Query("stable")) -> JSONResponse:
    normalized_mode = _normalize_mode(mode)
    return JSONResponse(_load_timeseries(normalized_mode))
