"""Assemble product layer blocks onto an engine result dict (additive fields only)."""

from __future__ import annotations

from typing import Any

from neraium_core.product.decision_layer import build_decision_layer
from neraium_core.product.operational_recommendation import build_operational_recommendation
from neraium_core.product.trust_layer import build_trust_layer


def build_product_layer(result: dict[str, Any]) -> dict[str, Any]:
    """
    Return new top-level keys to merge into the engine result.

    Does not remove or rename existing keys. Structural ``attribution`` is produced
    by the engine; this layer does not overwrite it.
    """
    decision_layer = build_decision_layer(result)
    enriched = {**result, "decision_layer": decision_layer}
    return {
        "decision_layer": decision_layer,
        "trust_layer": build_trust_layer(result),
        "operational_recommendation": build_operational_recommendation(enriched),
    }
