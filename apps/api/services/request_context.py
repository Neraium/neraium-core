from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def resolve_customer_id(customer_id: str | None) -> str:
    text = str(customer_id or "").strip()
    return text or "default-customer"


def resolve_customer_id_for_run(service: Any, customer_id: str | None, run_id: str | None) -> str:
    resolved_customer = resolve_customer_id(customer_id)
    if customer_id is not None and str(customer_id).strip():
        return resolved_customer
    run_text = str(run_id or "").strip()
    if not run_text:
        return resolved_customer
    lookup = getattr(service, "get_run_customer_id", None)
    if callable(lookup):
        owner_customer = lookup(run_text)
        if owner_customer is not None and str(owner_customer).strip():
            return str(owner_customer).strip()
    return resolved_customer


def resolve_run_id(service: Any, run_id: str | None, *, customer_id: str | None) -> str | None:
    if run_id is not None and str(run_id).strip():
        return str(run_id).strip()
    active = service.get_active_run(customer_id=resolve_customer_id(customer_id))
    if active is None:
        return None
    rid = active.get("run_id")
    return str(rid) if rid is not None else None


def require_run_id(service: Any, run_id: str | None, *, customer_id: str | None) -> str:
    resolved = resolve_run_id(service, run_id, customer_id=customer_id)
    if resolved is None:
        raise HTTPException(status_code=400, detail="No active run. Create or activate a run first.")
    return resolved


def request_run_id_or_active(service: Any, run_id: str | None, *, customer_id: str | None) -> str | None:
    if run_id is None:
        return resolve_run_id(service, None, customer_id=customer_id)
    text = str(run_id).strip()
    if not text:
        return resolve_run_id(service, None, customer_id=customer_id)
    return text


def resolve_run_id_with_default(service: Any, run_id: str | None, *, customer_id: str | None, ensure_default_run: Any | None = None) -> str:
    resolved = request_run_id_or_active(service, run_id, customer_id=customer_id)
    if resolved is not None:
        return resolved
    if callable(ensure_default_run):
        created = ensure_default_run(service, customer_id=resolve_customer_id(customer_id))
    else:
        created = service.create_run(
            name="Default Run",
            config={"source": "api-default"},
            activate=True,
            customer_id=resolve_customer_id(customer_id),
        )
    return str(created.get("run_id"))
