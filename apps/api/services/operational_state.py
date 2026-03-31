from __future__ import annotations

import logging
from typing import Any


class OperationalStateService:
    def __init__(self, *, store_instance: Any, enabled: bool, logger: logging.Logger) -> None:
        self._store_instance = store_instance
        self.enabled = bool(enabled)
        self._logger = logger

    def persist(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        customer_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        if not self.enabled or self._store_instance is None:
            return
        try:
            self._store_instance.upsert_operational_state(
                state_key=key,
                state=payload,
                customer_id=customer_id,
                run_id=run_id,
            )
        except Exception:
            self._logger.exception("Failed to persist operational state key=%s", key)

    def delete(self, key: str) -> None:
        if not self.enabled or self._store_instance is None:
            return
        try:
            self._store_instance.delete_operational_state(state_key=key)
        except Exception:
            self._logger.exception("Failed to delete operational state key=%s", key)

    def list(self, *, key_prefix: str) -> list[dict[str, Any]]:
        if not self.enabled or self._store_instance is None:
            return []
        return list(self._store_instance.list_operational_state(key_prefix=key_prefix))

    def persist_pull_state(self, customer_id: str, state: dict[str, Any], *, public_state_builder: Any) -> None:
        public_state = public_state_builder(state, customer_id=customer_id)
        public_state["resume_on_startup"] = bool(public_state.get("running"))
        self.persist(
            f"pull_integration:{customer_id}",
            public_state,
            customer_id=customer_id,
            run_id=str(public_state.get("run_id")) if public_state.get("run_id") is not None else None,
        )
