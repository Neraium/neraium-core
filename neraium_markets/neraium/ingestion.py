"""Day 13: orchestrate connector → normalize → quality → accepted asset dict."""

from __future__ import annotations

from typing import Any

import pandas as pd

from neraium.connectors.registry import get_connector
from neraium.normalization import normalize_many_assets
from neraium.source_quality import assess_source_quality
from neraium.settings import Settings


def ingest_market_data(
    assets: list[str],
    timeframe: str | None,
    settings: Settings,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """
    Load via connector, normalize, assess quality, return accepted frames and summary.

    Returns ``(accepted_assets, summary_dict)``. Rejected assets are listed in summary.
    """
    if timeframe and settings.allowed_timeframes and timeframe not in settings.allowed_timeframes:
        raise ValueError(
            f"timeframe {timeframe!r} not in allowed_timeframes {settings.allowed_timeframes}"
        )

    connector = get_connector(settings.source_type, settings)
    source_meta = connector.get_source_metadata()
    raw = connector.load_many_assets(assets, timeframe)
    normalized = normalize_many_assets(raw, connector.get_name())

    th = dict(settings.source_quality_thresholds)
    accepted: dict[str, pd.DataFrame] = {}
    rejected: dict[str, Any] = {}
    quality_rows: list[dict[str, Any]] = []

    for asset, df in normalized.items():
        q = assess_source_quality(df, asset, thresholds=th)
        quality_rows.append(q)
        take = q["valid_for_pipeline"]
        if settings.reject_invalid_assets and q["quality_status"] != "good":
            take = False
        if take:
            accepted[asset] = df
        else:
            rejected[asset] = q

    summary = build_ingestion_summary(
        source_type=settings.source_type,
        assets_requested=list(assets),
        assets_loaded=list(normalized.keys()),
        assets_valid=list(accepted.keys()),
        assets_rejected=list(rejected.keys()),
        quality_rows=quality_rows,
        source_metadata=source_meta,
        rejected_details=rejected,
    )
    return accepted, summary


def build_ingestion_summary(
    *,
    source_type: str,
    assets_requested: list[str],
    assets_loaded: list[str],
    assets_valid: list[str],
    assets_rejected: list[str],
    quality_rows: list[dict[str, Any]],
    source_metadata: dict[str, Any],
    rejected_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured summary for manifests, CLI, and logs."""
    counts: dict[str, int] = {"good": 0, "degraded": 0, "invalid": 0}
    for q in quality_rows:
        st = q.get("quality_status", "invalid")
        if st in counts:
            counts[st] += 1
    return {
        "source_type": source_type,
        "assets_requested": assets_requested,
        "assets_loaded": assets_loaded,
        "assets_valid": assets_valid,
        "assets_rejected": assets_rejected,
        "quality_status_counts": counts,
        "source_metadata": source_metadata,
        "quality_by_asset": {q["asset"]: q for q in quality_rows},
        "rejected_details": rejected_details or {},
    }
