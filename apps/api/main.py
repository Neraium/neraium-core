from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from neraium_core import __version__
from neraium_core.grow.state_engine import GrowStateEngine
from neraium_core.grow.validation import parse_csv_text, parse_json_payload
from neraium_core.service import StructuralMonitoringService

DEFAULT_MAX_REQUEST_BODY_BYTES = 5 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = REPO_ROOT / "static"
WEB_DIR = STATIC_DIR / "web"


def is_api_key_valid(expected_api_key: str | None, provided_api_key: str | None) -> bool:
    if not expected_api_key:
        return True
    return bool(provided_api_key and provided_api_key == expected_api_key)


def _read_text(path: Path) -> str:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"asset not found: {path.name}")
    return path.read_text(encoding="utf-8")


def _db_path() -> str:
    return os.environ.get("NERAIUM_DB_PATH", str(REPO_ROOT / "artifacts" / "neraium-grow.db"))


def _deployment_mode() -> str:
    return os.environ.get("NERAIUM_DEPLOYMENT_MODE", "grow")


DEFAULT_GROW_ENGINE = GrowStateEngine(
    db_path=_db_path(),
    baseline_window=int(os.environ.get("NERAIUM_GROW_BASELINE_WINDOW", "8")),
    recent_window=int(os.environ.get("NERAIUM_GROW_RECENT_WINDOW", "4")),
)


def create_app(
    *,
    service: StructuralMonitoringService | None = None,
    grow_engine: GrowStateEngine | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
) -> FastAPI:
    app = FastAPI(title="Neraium Grow API", version="0.1.0")
    grow = grow_engine or DEFAULT_GROW_ENGINE
    legacy_service = service or StructuralMonitoringService()
    client_errors: list[dict[str, Any]] = []

    def _scope_list(scope_type: str) -> list[dict[str, Any]]:
        return grow.list_states(scope_type)

    def _scope_state(scope_type: str, scope_id: str) -> dict[str, Any]:
        state = grow.get_state(scope_type, scope_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"{scope_type} not found")
        return state.model_dump(mode="json")

    def _scope_timeline(scope_type: str, scope_id: str) -> dict[str, Any]:
        return {"scope_type": scope_type, "scope_id": scope_id, "timeline": grow.timeline(scope_type, scope_id)}

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _read_text(STATIC_DIR / "dashboard.html")

    @app.get("/dashboard")
    def dashboard_redirect() -> RedirectResponse:
        return RedirectResponse(url="/", status_code=307)

    @app.get("/demo/sii")
    def demo_redirect() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @app.get("/web/{asset_path:path}")
    def web_asset(asset_path: str) -> Response:
        path = WEB_DIR / asset_path
        text = _read_text(path)
        media_type = "text/plain; charset=utf-8"
        suffix = path.suffix.lower()
        if suffix in {".js", ".mjs"}:
            media_type = "text/javascript; charset=utf-8"
        elif suffix == ".css":
            media_type = "text/css; charset=utf-8"
        elif suffix == ".html":
            media_type = "text/html; charset=utf-8"
        return Response(content=text, media_type=media_type)

    @app.get("/health")
    def health() -> dict[str, Any]:
        latest = legacy_service.get_latest_result()
        setup = grow.setup_summary().model_dump(mode="json")
        return {
            "status": "ok",
            "service": "neraium-grow",
            "version": __version__,
            "deployment_mode": _deployment_mode(),
            "latest_result_available": latest is not None,
            "auth_configured": bool(os.environ.get("NERAIUM_API_KEY")),
            "persistence_available": True,
            **setup,
        }

    @app.get("/setup/summary")
    def setup_summary() -> dict[str, Any]:
        return grow.setup_summary().model_dump(mode="json")

    @app.post("/ingest/json")
    def ingest_json(payload: Any = Body(...)) -> dict[str, Any]:
        records = parse_json_payload(payload)
        decisions = grow.ingest_records(records)
        return {
            "accepted": len(records),
            "deployment_mode": "grow",
            "read_only": True,
            "setup_summary": grow.setup_summary().model_dump(mode="json"),
            "latest_decisions": [decision.model_dump(mode="json") for decision in decisions],
        }

    @app.post("/ingest/batch")
    def ingest_batch(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return ingest_json(payload)

    @app.post("/ingest/csv")
    async def ingest_csv(file: UploadFile = File(...)) -> dict[str, Any]:
        content = await file.read()
        if len(content) > max_request_body_bytes:
            raise HTTPException(status_code=413, detail="request body too large")
        records = parse_csv_text(content.decode("utf-8"))
        decisions = grow.ingest_records(records)
        return {
            "accepted": len(records),
            "deployment_mode": "grow",
            "read_only": True,
            "setup_summary": grow.setup_summary().model_dump(mode="json"),
            "latest_decisions": [decision.model_dump(mode="json") for decision in decisions],
        }

    @app.get("/facility/summary")
    def facility_summary() -> dict[str, Any]:
        return grow.facility_summary()

    @app.get("/portfolio/summary")
    def portfolio_summary() -> dict[str, Any]:
        return grow.portfolio_summary()

    @app.get("/plants")
    def plants() -> list[dict[str, Any]]:
        return _scope_list("plant")

    @app.get("/zones")
    def zones() -> list[dict[str, Any]]:
        return _scope_list("zone")

    @app.get("/rooms")
    def rooms() -> list[dict[str, Any]]:
        return _scope_list("room")

    @app.get("/assets")
    def assets() -> list[dict[str, Any]]:
        return _scope_list("subsystem")

    @app.get("/subsystems")
    def subsystems() -> list[dict[str, Any]]:
        return _scope_list("subsystem")

    @app.get("/plants/{plant_id}/state")
    def plant_state(plant_id: str) -> dict[str, Any]:
        return _scope_state("plant", plant_id)

    @app.get("/plants/{plant_id}/timeline")
    def plant_timeline(plant_id: str) -> dict[str, Any]:
        return _scope_timeline("plant", plant_id)

    @app.get("/rooms/{room_id}/state")
    def room_state(room_id: str) -> dict[str, Any]:
        return _scope_state("room", room_id)

    @app.get("/rooms/{room_id}/timeline")
    def room_timeline(room_id: str) -> dict[str, Any]:
        return _scope_timeline("room", room_id)

    @app.get("/zones/{zone_id}/state")
    def zone_state(zone_id: str) -> dict[str, Any]:
        return _scope_state("zone", zone_id)

    @app.get("/zones/{zone_id}/timeline")
    def zone_timeline(zone_id: str) -> dict[str, Any]:
        return _scope_timeline("zone", zone_id)

    @app.get("/assets/{asset_id}/state")
    def asset_state(asset_id: str) -> dict[str, Any]:
        return _scope_state("subsystem", asset_id)

    @app.get("/assets/{asset_id}/timeline")
    def asset_timeline(asset_id: str) -> dict[str, Any]:
        return _scope_timeline("subsystem", asset_id)

    @app.get("/subsystems/{subsystem_id}/state")
    def subsystem_state(subsystem_id: str) -> dict[str, Any]:
        return _scope_state("subsystem", subsystem_id)

    @app.get("/subsystems/{subsystem_id}/timeline")
    def subsystem_timeline(subsystem_id: str) -> dict[str, Any]:
        return _scope_timeline("subsystem", subsystem_id)

    @app.get("/facilities/{facility_id}/state")
    def facility_state(facility_id: str) -> dict[str, Any]:
        return _scope_state("facility", facility_id)

    @app.get("/facilities/{facility_id}/timeline")
    def facility_timeline(facility_id: str) -> dict[str, Any]:
        return _scope_timeline("facility", facility_id)

    @app.get("/fleet/summary")
    def fleet_summary() -> dict[str, Any]:
        return grow.fleet_summary()

    @app.get("/compare")
    def compare(scope_type: str = Query(...), scope_id: str | None = Query(default=None)) -> dict[str, Any]:
        return grow.compare_scope(scope_type, scope_id)

    @app.post("/compare/entities")
    def compare_entities(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        scope_type = str(payload.get("scope_type") or "")
        scope_ids = [str(item) for item in payload.get("scope_ids", [])]
        if not scope_type or not scope_ids:
            raise HTTPException(status_code=400, detail="scope_type and scope_ids are required")
        return grow.compare_entities(scope_type, scope_ids)

    @app.get("/decisions/latest")
    def decisions_latest(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
        return grow.latest_decisions(limit)

    @app.get("/decisions/{decision_id}")
    def decision_by_id(decision_id: str) -> dict[str, Any]:
        decision = grow.get_decision(decision_id)
        if decision is None:
            raise HTTPException(status_code=404, detail="decision not found")
        return decision

    @app.post("/client-errors")
    async def client_error(request: Request) -> dict[str, Any]:
        payload = await request.json()
        client_errors.append(payload)
        return {"accepted": True, "count": len(client_errors)}

    @app.post("/ingest")
    @app.post("/ingest/frame")
    def ingest_legacy(
        payload: dict[str, Any] = Body(...),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        result = legacy_service.ingest_payload(payload, run_id=run_id, customer_id=customer_id)
        return {"latest": result, "results": [result], "count": 1}

    @app.post("/ingest/csv/preview")
    async def ingest_csv_preview(file: UploadFile = File(...)) -> dict[str, Any]:
        content = (await file.read()).decode("utf-8")
        rows = parse_csv_text(content)
        return {"rows_detected": len(rows), "preview": [row.model_dump(mode="json") for row in rows[:3]]}

    @app.get("/results/latest")
    def results_latest(
        customer_id: str | None = Query(default=None),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        latest = legacy_service.get_latest_result(run_id=run_id, customer_id=customer_id)
        results = [latest] if latest else []
        return {"latest": latest, "results": results, "count": len(results)}

    @app.get("/results/recent")
    def results_recent(
        limit: int = Query(100, ge=1, le=1000),
        customer_id: str | None = Query(default=None),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        results = legacy_service.list_recent_results(limit=limit, run_id=run_id, customer_id=customer_id)
        return {"latest": results[0] if results else None, "results": results, "count": len(results)}

    @app.get("/state")
    def state(customer_id: str | None = Query(default=None), run_id: str | None = Query(default=None)) -> dict[str, Any]:
        return {"state": legacy_service.get_current_state(run_id=run_id, customer_id=customer_id)}

    @app.get("/history")
    def history(
        limit: int = Query(100, ge=1, le=1000),
        customer_id: str | None = Query(default=None),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        rows = legacy_service.get_recent_history(limit=limit, run_id=run_id, customer_id=customer_id)
        return {"history": rows, "count": len(rows)}

    @app.get("/recommendation")
    @app.get("/recommendations/latest")
    def recommendation(
        customer_id: str | None = Query(default=None),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        return {
            "operational_recommendation": legacy_service.get_latest_recommendation(
                run_id=run_id,
                customer_id=customer_id,
            )
        }

    @app.get("/decision")
    def decision(customer_id: str | None = Query(default=None), run_id: str | None = Query(default=None)) -> dict[str, Any]:
        return {"decision": legacy_service.get_latest_decision(run_id=run_id, customer_id=customer_id)}

    @app.get("/explanation")
    def explanation(customer_id: str | None = Query(default=None), run_id: str | None = Query(default=None)) -> dict[str, Any]:
        return {"explanation_text": legacy_service.get_latest_explanation(run_id=run_id, customer_id=customer_id)}

    @app.get("/events/latest")
    def events_latest(customer_id: str | None = Query(default=None), run_id: str | None = Query(default=None)) -> dict[str, Any]:
        state_row = legacy_service.get_current_state(run_id=run_id, customer_id=customer_id) or {}
        events = state_row.get("events", [])
        cycle = state_row.get("cycle", 0)
        return {"events": events, "cycle": cycle}

    @app.post("/reset")
    def reset() -> dict[str, Any]:
        legacy_service.reset()
        return {"reset": True}

    @app.post("/runs")
    def create_run(payload: dict[str, Any] = Body(...), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        run = legacy_service.create_run(
            name=str(payload.get("name") or "run"),
            config=payload.get("config") if isinstance(payload.get("config"), dict) else None,
            activate=bool(payload.get("activate", True)),
            customer_id=customer_id,
        )
        return {"run": run}

    @app.get("/v2/state")
    def state_v2(customer_id: str | None = Query(default=None), run_id: str | None = Query(default=None)) -> dict[str, Any]:
        current = legacy_service.get_current_state(run_id=run_id, customer_id=customer_id) or {}
        contract = {
            "contract_version": "decision-contract.v2",
            "policy": {"read_only": True, "advisory_only": True},
            "data_quality": current.get("data_quality_summary", {}),
            **current,
        }
        return {"state": contract}

    @app.get("/v2/recommendation")
    def recommendation_v2(customer_id: str | None = Query(default=None), run_id: str | None = Query(default=None)) -> dict[str, Any]:
        rec = legacy_service.get_latest_recommendation(run_id=run_id, customer_id=customer_id) or {}
        return {"operator_action": {"contract_version": "decision-contract.v2", **rec}}

    @app.get("/v2/history")
    def history_v2(
        limit: int = Query(100, ge=1, le=1000),
        customer_id: str | None = Query(default=None),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        rows = legacy_service.get_recent_history(limit=limit, run_id=run_id, customer_id=customer_id)
        wrapped = [{"contract_version": "decision-contract.v2", **row} for row in rows]
        return {"history": wrapped, "count": len(wrapped)}

    @app.get("/v2/results/latest")
    def results_latest_v2(customer_id: str | None = Query(default=None), run_id: str | None = Query(default=None)) -> dict[str, Any]:
        latest = legacy_service.get_current_state(run_id=run_id, customer_id=customer_id)
        wrapped = {"contract_version": "decision-contract.v2", **latest} if latest else None
        return {"latest": wrapped, "count": 1 if wrapped else 0}

    @app.post("/onboarding/verify")
    def onboarding_verify(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        expected_api_key = os.environ.get("NERAIUM_API_KEY")
        provided = payload.get("api_key")
        if not is_api_key_valid(expected_api_key, None if provided is None else str(provided)):
            raise HTTPException(status_code=401, detail="invalid api key")
        return {
            "credentials_verified": True,
            "customer_id": payload.get("customer_id"),
            "environment_label": payload.get("environment_label", "grow"),
        }

    @app.post("/onboarding/profile")
    def onboarding_profile(payload: dict[str, Any] = Body(...), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        return {"saved": True, "customer_id": customer_id, "profile": payload}

    @app.post("/onboarding/bootstrap-run")
    def onboarding_bootstrap_run(customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        run = legacy_service.create_run(name="bootstrap-grow-run", customer_id=customer_id)
        return {"run": run}

    @app.post("/onboarding/test-frame")
    def onboarding_test_frame(customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        run = legacy_service.get_active_run(customer_id=customer_id)
        run_id = run.get("run_id") if isinstance(run, dict) else None
        result = legacy_service.ingest_payload(
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "site_id": "grow-onboarding",
                "asset_id": "baseline-seed",
                "sensor_values": {"pressure": 50.0, "flow": 10.0},
            },
            run_id=run_id,
            customer_id=customer_id,
        )
        return {"ingest_accepted": True, "active_run_id": run_id, "latest": result}

    @app.get("/onboarding/status")
    def onboarding_status(customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        run = legacy_service.get_active_run(customer_id=customer_id)
        latest = legacy_service.get_latest_result(customer_id=customer_id)
        return {
            "ready": latest is not None,
            "test_frame_sent": latest is not None,
            "active_run_id": run.get("run_id") if isinstance(run, dict) else None,
        }

    return app


app = create_app()
