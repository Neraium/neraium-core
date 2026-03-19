from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from neraium_core.decision_layer import decision_output
from neraium_core.scoring import canonicalize_weights, composite_instability_score_normalized

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
            "interpreted_state": "STRUCTURAL_INSTABILITY_OBSERVED",
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
        "interpreted_state": "NOMINAL_STRUCTURE",
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

def _coerce_status(summary: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize unit-summary shapes for the UI.

    Supports either:
    A) flat unit summary fields, or
    B) nested under `status` and/or `scores`.
    """

    default = _mock_status("stable")
    status_block = summary.get("status", {}) if isinstance(summary.get("status"), dict) else {}
    scores_block = summary.get("scores", {}) if isinstance(summary.get("scores"), dict) else {}

    def pick(key: str) -> Any:
        # Presence checks matter: `False` is meaningful for `signal_emitted`.
        if key in status_block:
            return status_block.get(key)
        if key in scores_block:
            return scores_block.get(key)
        if key in summary:
            return summary.get(key)
        return None

    def pick_float(key: str, default_value: float) -> float:
        v = pick(key)
        if v is None:
            return default_value
        try:
            return float(v)
        except (TypeError, ValueError):
            return default_value

    # FD004 "real" unit summaries often provide averages instead of "latest_*".
    fd004_avg_instability = pick("average_instability")
    fd004_avg_drift = pick("average_drift")
    has_fd004_avg_instability = fd004_avg_instability is not None
    has_fd004_avg_drift = fd004_avg_drift is not None

    phase_value = pick("phase")
    phase_missing = phase_value is None
    phase = phase_value if phase_value is not None else "unknown"

    risk_value = pick("risk_level")
    risk_missing = risk_value is None
    risk_level = risk_value if risk_value is not None else "UNKNOWN"

    signal_emitted_value = pick("signal_emitted")
    signal_emitted_missing = signal_emitted_value is None
    signal_emitted = bool(signal_emitted_value) if signal_emitted_value is not None else False

    signal_strength_value = pick("signal_strength")
    signal_strength_missing = signal_strength_value is None
    signal_strength = signal_strength_value if signal_strength_value is not None else "low"

    confidence_value = pick("confidence")
    confidence = confidence_value
    if confidence is None:
        # If the real FD004 unit summary only provides averages, use a neutral mid-confidence.
        confidence = "medium" if has_fd004_avg_instability else "low"

    interpreted_state_value = pick("interpreted_state")
    interpreted_state_missing = interpreted_state_value is None
    interpreted_state = interpreted_state_value

    operator_message_value = pick("operator_message")
    operator_message_missing = operator_message_value is None
    operator_message = operator_message_value if operator_message_value is not None else default["operator_message"]

    # If the report doesn't include decision-layer outputs (common for FD004 real
    # `unit_summaries`), compute them from the available averages via the decision layer.
    if interpreted_state_missing:
        try:
            relational_drift_for_decision = float(fd004_avg_drift or 0.0)
            regime_drift_for_decision = relational_drift_for_decision * 0.3

            # FD004 real reports often only provide a coarse confidence label.
            # Approximate it into an evidence-quality factor to down-weight
            # Tier-1 components when confidence is low.
            if isinstance(confidence, str):
                conf_lower = confidence.lower().strip()
                if conf_lower == "high":
                    evidence_conf = 1.0
                elif conf_lower == "medium":
                    evidence_conf = 0.6
                else:
                    evidence_conf = 0.25
            else:
                # If somehow numeric, clamp to [0, 1]
                try:
                    evidence_conf = float(confidence)
                except (TypeError, ValueError):
                    evidence_conf = 0.25
                evidence_conf = max(0.0, min(1.0, evidence_conf))

            component_confidence = {
                "relational_drift": evidence_conf,
                "regime_drift": evidence_conf,
                "spectral": evidence_conf,
                "early_warning": evidence_conf,
            }

            components_for_decision = {
                "relational_drift": relational_drift_for_decision,
                "regime_drift": regime_drift_for_decision,
                "directional_divergence": 0.0,
                "spectral": 0.0,
                "entropy": 0.0,
                "subsystem_instability": 0.0,
                "early_warning": 0.0,
            }

            # Confidence-weighted composite: mirror backend by scaling weights
            # per component (then pass confidence-weighted components into the
            # decision layer for interpreted_state/explanations).
            base_weights = canonicalize_weights()
            weights_for_composite: dict[str, float] = {}
            for k, w in base_weights.items():
                weights_for_composite[k] = float(w) * float(component_confidence.get(k, 0.0))

            composite_score = composite_instability_score_normalized(
                components_for_decision,
                weights=weights_for_composite,
            )

            components_for_decision_scaled = {
                k: float(v) * float(component_confidence.get(k, 0.0)) if k in component_confidence else float(v)
                for k, v in components_for_decision.items()
            }
            decision = decision_output(
                composite_score=composite_score,
                components=components_for_decision_scaled,
                forecast={"trend": 0.0},
            )

            interpreted_state = decision.get("interpreted_state")
            if phase_missing:
                phase = decision.get("phase", phase)
            if risk_missing:
                risk_level = decision.get("risk_level", risk_level)
            if signal_emitted_missing:
                signal_emitted = bool(decision.get("signal_emitted", signal_emitted))
            if signal_strength_missing:
                signal_strength = decision.get("signal_strength", signal_strength)
            if operator_message_missing:
                operator_message = decision.get("operator_message", operator_message)
        except Exception:
            # Last-resort: fall back to the default mock values rather than propagating
            # a literal unknown state into the UI.
            if interpreted_state is None:
                try:
                    interpreted_state = decision_output(
                        composite_score=0.0,
                        components={
                            "relational_drift": 0.0,
                            "regime_drift": 0.0,
                            "directional_divergence": 0.0,
                            "spectral": 0.0,
                        },
                        forecast={"trend": 0.0},
                    ).get("interpreted_state")
                except Exception:
                    interpreted_state = default["interpreted_state"]

    latest_drift = pick("latest_drift")
    if latest_drift is None:
        latest_drift = pick("structural_drift_score")
    if latest_drift is None and has_fd004_avg_drift:
        latest_drift = fd004_avg_drift
    if latest_drift is None:
        latest_drift = 0.0
    try:
        latest_drift = float(latest_drift)
    except (TypeError, ValueError):
        latest_drift = 0.0

    latest_instability = pick("latest_instability")
    if latest_instability is None and has_fd004_avg_instability:
        latest_instability = fd004_avg_instability
    if latest_instability is None:
        latest_instability = 0.0
    try:
        latest_instability = float(latest_instability)
    except (TypeError, ValueError):
        latest_instability = 0.0

    regime_name = pick("regime_name")

    regime_distance_raw = pick("regime_distance")
    if regime_distance_raw is None:
        regime_distance = None
    else:
        try:
            regime_distance = float(regime_distance_raw)
        except (TypeError, ValueError):
            regime_distance = None

    regime_drift = pick("regime_drift")
    if regime_drift is None and has_fd004_avg_drift:
        # Mapping requested for FD004 real schema.
        regime_drift = fd004_avg_drift * 0.3
    if regime_drift is None:
        regime_drift = 0.0
    try:
        regime_drift = float(regime_drift)
    except (TypeError, ValueError):
        regime_drift = 0.0

    assigned_regime: str | None = None
    interpreted_state_str = str(interpreted_state or "").upper().strip()


    if "INSTABILITY" in interpreted_state_str:
        assigned_regime = "unstable"
    elif "NOMINAL" in interpreted_state_str:
        assigned_regime = "stable"
    elif "TRANSITION" in interpreted_state_str:
        assigned_regime = "transitional"
    else:
        assigned_regime = "unknown"

    return {
        "phase": phase,
        "risk_level": risk_level,
        "signal_emitted": signal_emitted,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "interpreted_state": interpreted_state,
        "assigned_regime": assigned_regime,
        "operator_message": operator_message,
        "latest_drift": latest_drift,
        "latest_instability": latest_instability,
        "regime_name": regime_name,
        "regime_distance": regime_distance,
        "regime_drift": regime_drift,
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
    if not isinstance(summary, dict):
        return _mock_status("stable")

    return _coerce_status(summary)


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
