"""Grouped transition detection summaries for replay, pilots, and batch exports."""

from __future__ import annotations

from typing import Any

import pandas as pd

from neraium_core.detection.transition_config import TransitionDetectionConfig
from neraium_core.detection.transition_detector import detect_transition


def summarize_entity_transitions(
    df: pd.DataFrame,
    group_column: str,
    *,
    config: TransitionDetectionConfig | None = None,
    rul_column: str | None = None,
    entity_id_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run :func:`detect_transition` per group and build a summary table plus aggregate stats.

    Parameters
    ----------
    df
        Combined dataframe for many entities (e.g. all units).
    group_column
        Column to group by (e.g. ``unit`` or ``asset_id``).
    rul_column
        Optional remaining-life column for lead-time style metrics.
    entity_id_column
        If set, copy this column into ``entity_id`` in the output; otherwise ``entity_id`` is the group key.
    """
    cfg = config or TransitionDetectionConfig()
    rows: list[dict[str, Any]] = []

    for key, g in df.groupby(group_column, sort=False):
        g = g.sort_values(cfg.index_column)
        r = detect_transition(g, cfg)
        entity_id = g[entity_id_column].iloc[0] if entity_id_column and entity_id_column in g.columns else key
        out: dict[str, Any] = {
            "entity_id": entity_id,
            "group_key": key,
            "detected": r.get("detected", False),
            "transition_position": r.get("transition_position"),
            "transition_index": r.get("transition_index"),
            "normalized_progress_at_detection": r.get("normalized_progress_at_detection"),
            "remaining_life_normalized": r.get("remaining_life_normalized"),
            "lead_time": r.get("lead_time"),
            "signal_used": r.get("signal_used"),
            "confidence": r.get("confidence"),
            "reason": r.get("reason") or r.get("detection_reason"),
            "detection_reason": r.get("detection_reason"),
        }
        if rul_column and rul_column in g.columns and r.get("transition_position") is not None:
            try:
                pos = float(r["transition_position"])
                row_at = g.loc[g[cfg.index_column] == pos]
                if len(row_at) == 0:
                    row_at = g.iloc[[int(r["transition_index"] or 0)]]
                rul_v = pd.to_numeric(row_at[rul_column], errors="coerce").iloc[0]
                out["rul_at_detection"] = float(rul_v) if pd.notna(rul_v) else None
            except Exception:
                out["rul_at_detection"] = None
        rows.append(out)

    summary = pd.DataFrame(rows)
    agg = aggregate_detection_summary(summary)
    return summary, agg


def aggregate_detection_summary(per_entity: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate statistics from a per-entity summary dataframe."""
    if per_entity is None or len(per_entity) == 0:
        return {"n_entities": 0, "n_detected": 0, "detection_rate": 0.0}
    n = len(per_entity)
    det = per_entity["detected"].astype(bool) if "detected" in per_entity.columns else pd.Series(False, index=per_entity.index)
    n_det = int(det.sum())
    prog = per_entity.loc[det, "normalized_progress_at_detection"] if "normalized_progress_at_detection" in per_entity.columns else pd.Series(dtype=float)
    conf = per_entity.loc[det, "confidence"] if "confidence" in per_entity.columns else pd.Series(dtype=float)
    return {
        "n_entities": n,
        "n_detected": n_det,
        "detection_rate": float(n_det / n) if n else 0.0,
        "mean_normalized_progress": float(prog.mean()) if len(prog) else None,
        "std_normalized_progress": float(prog.std()) if len(prog) > 1 else None,
        "mean_confidence": float(conf.mean()) if len(conf) else None,
    }


def export_detection_tables(
    per_entity: pd.DataFrame,
    aggregate: dict[str, Any],
    *,
    prefix: str = "transition",
) -> dict[str, pd.DataFrame]:
    """Return dataframes suitable for CSV export (per-entity + one-row aggregate)."""
    agg_df = pd.DataFrame([aggregate])
    return {
        f"{prefix}_per_entity": per_entity,
        f"{prefix}_aggregate": agg_df,
    }
