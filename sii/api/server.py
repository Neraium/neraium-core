from __future__ import annotations

from fastapi import FastAPI, HTTPException

from sii.models import HealthResponse, TelemetryIngestRequest
from sii.service import StructuralIntelligenceService

app = FastAPI(title="SII Service")
service = StructuralIntelligenceService()


@app.post("/telemetry")
def ingest(payload: TelemetryIngestRequest) -> dict[str, str]:
    service.ingest(payload)
    return {"status": "accepted"}


@app.get("/windows/latest")
def latest_window() -> dict:
    result = service.latest_window_snapshot()
    if result is None:
        raise HTTPException(status_code=404, detail="No windows available")
    return result.model_dump(mode="json")


@app.get("/observables/latest")
def latest_observables() -> dict:
    if service.latest_observables is None:
        raise HTTPException(status_code=404, detail="No observables available")
    return service.latest_observables.model_dump(mode="json")


@app.get("/drift/latest")
def latest_drift() -> dict:
    if service.latest_drift is None:
        raise HTTPException(status_code=404, detail="No drift available")
    return service.latest_drift.model_dump(mode="json")


@app.get("/regime/latest")
def latest_regime() -> dict:
    if service.latest_regime is None:
        raise HTTPException(status_code=404, detail="No regime available")
    return service.latest_regime.model_dump(mode="json")


@app.get("/forecast/latest")
def latest_forecast() -> dict:
    if service.latest_forecast is None:
        raise HTTPException(status_code=404, detail="No forecast available")
    return service.latest_forecast.model_dump(mode="json")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()
