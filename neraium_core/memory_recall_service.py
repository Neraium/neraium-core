from __future__ import annotations

from typing import Any

from neraium_core.memory_recall import (
    DEDUPLICATE_SIMILARITY_THRESHOLD,
    build_structural_memory_signature,
    classify_novelty,
    determine_pattern_family,
    short_memory_summary,
    signature_similarity,
)
from neraium_core.result_projector import CanonicalOutputProjector
from neraium_core.store import ResultStore


class MemoryRecallService:
    """Builds structural memory recall payloads and persists memory records."""

    def __init__(self, store: ResultStore, projector: CanonicalOutputProjector):
        self._store = store
        self._projector = projector

    def latest_history_context(self, *, run_id: str | None, customer_id: str) -> dict[str, Any] | None:
        return self._store.get_latest_service_history(run_id=run_id, customer_id=customer_id)

    @staticmethod
    def _recall_scope(match: dict[str, Any], *, site_id: str | None, asset_id: str | None) -> str:
        if asset_id is not None and str(match.get("asset_id")) == str(asset_id):
            return "same_asset"
        if site_id is not None and str(match.get("site_id")) == str(site_id):
            return "same_site"
        return "cross_asset"

    def _build_signature(
        self,
        result: dict[str, Any],
        *,
        cycle: int,
        run_id: str | None,
        customer_id: str,
    ) -> dict[str, Any]:
        canonical = self._projector.project_canonical_output(
            result,
            cycle=cycle,
            run_id=run_id,
            customer_id=customer_id,
            previous=None,
            memory_recall=None,
        )
        return build_structural_memory_signature(canonical)

    def build_memory_recall(
        self,
        result: dict[str, Any],
        *,
        cycle: int,
        run_id: str | None,
        customer_id: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        signature = self._build_signature(result, cycle=cycle, run_id=run_id, customer_id=customer_id)
        site_id = str(result.get("site_id")) if result.get("site_id") is not None else None
        asset_id = str(result.get("asset_id")) if result.get("asset_id") is not None else None
        history = self._store.list_structural_memory(
            customer_id=customer_id,
            limit=1000,
            exclude_cycle=cycle,
        )
        scored: list[dict[str, Any]] = []
        for item in history:
            prior_sig = item.get("signature")
            if not isinstance(prior_sig, dict):
                continue
            breakdown = signature_similarity(signature, prior_sig)
            scored.append(
                {
                    **item,
                    "similarity": breakdown.total,
                    "drivers_overlap": breakdown.drivers_overlap,
                    "scope": self._recall_scope(item, site_id=site_id, asset_id=asset_id),
                }
            )
        scored.sort(
            key=lambda x: (
                {"same_asset": 0, "same_site": 1, "cross_asset": 2}.get(str(x.get("scope")), 3),
                -float(x.get("similarity", 0.0)),
                -int(x.get("memory_id", 0)),
            )
        )
        nearest = scored[0] if scored else None
        best_similarity = float(nearest.get("similarity", 0.0)) if nearest else 0.0
        best_overlap = float(nearest.get("drivers_overlap", 0.0)) if nearest else 0.0
        novelty = classify_novelty(
            best_similarity=best_similarity,
            best_overlap=best_overlap,
            memory_count=len(scored),
        )
        top_matches = []
        for item in scored[:top_k]:
            top_matches.append(
                {
                    "memory_id": item.get("memory_id"),
                    "similarity": float(item.get("similarity", 0.0)),
                    "asset_id": item.get("asset_id"),
                    "run_id": item.get("run_id"),
                    "cycle": item.get("cycle"),
                    "summary": item.get("summary"),
                    "scope": item.get("scope"),
                }
            )
        pattern_family = determine_pattern_family(signature, similarity=best_similarity if nearest else None)
        nearest_match = {
            "found": nearest is not None,
            "similarity": best_similarity,
            "memory_id": nearest.get("memory_id") if nearest else None,
            "asset_id": nearest.get("asset_id") if nearest else None,
            "run_id": nearest.get("run_id") if nearest else None,
            "cycle": nearest.get("cycle") if nearest else None,
            "summary": nearest.get("summary") if nearest else None,
            "scope": nearest.get("scope") if nearest else None,
        }
        return {
            "status": {
                "enabled": True,
                "memory_records_considered": len(scored),
                "scope": "customer",
            },
            "signature": signature,
            "novelty": novelty,
            "nearest_match": nearest_match,
            "top_matches": top_matches,
            "pattern_family": pattern_family,
        }

    def persist_structural_memory(
        self,
        canonical: dict[str, Any],
        memory_recall: dict[str, Any],
        *,
        customer_id: str,
    ) -> dict[str, Any] | None:
        signature = memory_recall.get("signature")
        if not isinstance(signature, dict):
            return None
        nearest = memory_recall.get("nearest_match") if isinstance(memory_recall.get("nearest_match"), dict) else {}
        nearest_similarity = float(nearest.get("similarity", 0.0) or 0.0)
        if nearest_similarity >= DEDUPLICATE_SIMILARITY_THRESHOLD and nearest.get("scope") == "same_asset":
            return None
        risk = canonical.get("risk_assessment") if isinstance(canonical.get("risk_assessment"), dict) else {}
        decision = canonical.get("decision") if isinstance(canonical.get("decision"), dict) else {}
        session = canonical.get("session") if isinstance(canonical.get("session"), dict) else {}
        family = memory_recall.get("pattern_family") if isinstance(memory_recall.get("pattern_family"), dict) else {}
        novelty = memory_recall.get("novelty") if isinstance(memory_recall.get("novelty"), dict) else {}
        record = {
            "customer_id": customer_id,
            "run_id": session.get("run_id"),
            "site_id": session.get("site_id"),
            "asset_id": session.get("asset_id"),
            "timestamp": canonical.get("timestamp"),
            "cycle": canonical.get("cycle"),
            "memory_key": f"{signature.get('risk_level', 'UNKNOWN')}|{signature.get('decision_state', 'UNKNOWN')}|{signature.get('regime_key', 'unknown')}",
            "family_label": family.get("label"),
            "novelty": novelty,
            "signature": signature,
            "summary": short_memory_summary(signature),
            "decision": decision.get("action"),
            "risk_level": risk.get("risk_level"),
            "event_flags": canonical.get("events", []),
            "source_history_id": memory_recall.get("source_history_id"),
            "metadata": {
                "nearest_similarity": nearest_similarity,
                "nearest_memory_id": nearest.get("memory_id"),
                "pattern_family_confidence": family.get("confidence"),
            },
        }
        return self._store.save_structural_memory(record, customer_id=customer_id)

    def get_memory_matches(
        self,
        *,
        customer_id: str,
        run_id: str | None = None,
        site_id: str | None = None,
        asset_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._store.list_structural_memory(
            customer_id=customer_id,
            run_id=run_id,
            site_id=site_id,
            asset_id=asset_id,
            limit=limit,
        )
