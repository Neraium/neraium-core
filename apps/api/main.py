from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from neraium_core.service import StructuralIntelligenceService, StructuralResult


class CSVIngestRequest(BaseModel):
    csv_text: str


app = FastAPI(title="Neraium SII API")
service = StructuralIntelligenceService()


@app.post("/ingest", response_model=StructuralResult)
def ingest(payload: dict[str, Any]) -> StructuralResult:
    try:
        return service.ingest_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/batch", response_model=StructuralResult)
def ingest_batch(payloads: list[dict[str, Any]]) -> StructuralResult:
    try:
        return service.ingest_batch(payloads)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/csv", response_model=StructuralResult)
def ingest_csv(request: CSVIngestRequest) -> StructuralResult:
    try:
        return service.ingest_csv(request.csv_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/result", response_model=StructuralResult)
def result() -> StructuralResult:
    try:
        return service.get_latest_result()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/reset")
def reset() -> dict[str, bool]:
    service.reset()
    return {"ok": True}
