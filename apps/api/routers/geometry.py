from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..services.geometry import build_geometry_payload


def _empty_geometry(run_id: str | None = None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "result_id": None,
        "timestamp": None,
        "available": False,
        "reason": "No results available for this run yet.",
        "metrics": {},
        "nodes": [],
        "edges": [],
        "projection": {
            "method": "spectral_projection_from_engine_correlation_geometry",
            "is_visualization_projection": True,
            "source": "engine correlation geometry + graph analytics",
            "note": (
                "Node positions are a deterministic visualization projection derived from engine "
                "correlation outputs; they are not the core SII computation space."
            ),
        },
        "provenance": {
            "engine_fields": [
                "sensor_relationships",
                "experimental_analytics.correlation_geometry.current",
                "experimental_analytics.correlation_geometry.baseline",
            ],
            "positions": "deterministic projection from engine outputs",
        },
        "graph_analytics": None,
        "system_state": None,
    }


def build_geometry_router(*, service_instance, resolve_customer_id, geometry_envelope_model) -> APIRouter:
    router = APIRouter(tags=["geometry"])

    @router.get("/runs/{run_id}/geometry", response_model=geometry_envelope_model)
    def get_run_geometry(
        run_id: str,
        result_id: int | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        run = service_instance.get_run(run_id, customer_id=resolved_customer)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")

        if result_id is not None:
            result = service_instance.get_result_by_id(
                result_id,
                run_id=run_id,
                customer_id=resolved_customer,
            )
            if result is None:
                raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
        else:
            result = service_instance.get_latest_result(run_id=run_id, customer_id=resolved_customer)

        if result is None:
            return _empty_geometry(run_id)

        return build_geometry_payload(result, run_id=run_id)

    @router.get("/results/{result_id}/geometry", response_model=geometry_envelope_model)
    def get_geometry(
        result_id: int,
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved_run = run_id.strip() if isinstance(run_id, str) and run_id.strip() else None
        result = service_instance.get_result_by_id(
            result_id,
            run_id=resolved_run,
            customer_id=resolved_customer,
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
        return build_geometry_payload(result, run_id=resolved_run)

    return router
