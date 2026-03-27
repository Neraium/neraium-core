"""
FD004 / CMAPSS convenience wrappers around :mod:`neraium_core.detection`.

Dataset-specific scripts should pass parameters via :class:`~neraium_core.detection.TransitionDetectionConfig`;
this module keeps stable function names for existing CLIs and notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neraium_core.detection.transition_config import TransitionDetectionConfig
from neraium_core.detection.transition_detector import detect_transition
from neraium_core.detection.transition_evaluation import summarize_entity_transitions
from neraium_core.detection.transition_plots import (
    plot_normalized_detection_histogram,
    plot_raw_position_histogram,
    save_figure,
)

EPS = 1e-9


def _fd004_config(
    *,
    min_cycle: int,
    window: int,
    threshold: float,
    pressure_col: str,
    cycle_col: str,
    sustained_points: int,
    verbose: bool = False,
) -> TransitionDetectionConfig:
    return TransitionDetectionConfig(
        index_column=cycle_col,
        primary_signal=pressure_col,
        warmup_cycles=float(min_cycle),
        warmup_fraction=None,
        smoothing_window=int(window),
        threshold=float(threshold),
        sustained_points=int(sustained_points),
        fallback_mode="max_slope",
        entity_group_column="unit",
        verbose=verbose,
    )


def detect_structural_transition(
    unit_df: pd.DataFrame,
    *,
    min_cycle: int = 30,
    window: int = 5,
    threshold: float = 0.5,
    pressure_col: str = "transition_pressure",
    cycle_col: str = "cycle",
    sustained_points: int = 2,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Backward-compatible wrapper for :func:`neraium_core.detection.detect_transition`.

    See :class:`TransitionDetectionConfig` for the full parameter surface.
    """
    cfg = _fd004_config(
        min_cycle=min_cycle,
        window=window,
        threshold=threshold,
        pressure_col=pressure_col,
        cycle_col=cycle_col,
        sustained_points=sustained_points,
        verbose=verbose,
    )
    r = detect_transition(unit_df, cfg)
    wm = r.get("warmup_applied") or {}
    meta: dict[str, Any] = {
        "min_cycle": int(min_cycle),
        "window": int(window),
        "threshold": float(threshold),
        "pressure_col": pressure_col,
        "rows_input": int(len(unit_df)) if unit_df is not None else 0,
        "rows_after_warmup": wm.get("rows_after_warmup"),
        "warmup_mode": wm.get("warmup_mode"),
    }
    if r.get("diagnostics"):
        meta["detector_diagnostics"] = r["diagnostics"]
    return {
        "detected": bool(r.get("detected")),
        "transition_index": r.get("transition_index"),
        "transition_cycle": r.get("transition_position"),
        "remaining_life_normalized": r.get("remaining_life_normalized"),
        "lead_time_cycles": r.get("lead_time_cycles"),
        "reason": r.get("reason") or r.get("detection_reason"),
        "metadata": meta,
        "transition_position": r.get("transition_position"),
        "confidence": r.get("confidence"),
        "signal_used": r.get("signal_used"),
        "detection_reason": r.get("detection_reason"),
    }


def summarize_fd004_transitions(
    replay_df: pd.DataFrame,
    *,
    unit_col: str = "unit",
    min_cycle: int = 30,
    window: int = 5,
    threshold: float = 0.5,
    pressure_col: str = "transition_pressure",
    cycle_col: str = "cycle",
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Per-unit summary using the shared detector; output shape matches historical FD004 scripts.
    """
    cfg = _fd004_config(
        min_cycle=min_cycle,
        window=window,
        threshold=threshold,
        pressure_col=pressure_col,
        cycle_col=cycle_col,
        sustained_points=2,
        verbose=verbose,
    )
    summary_df, agg = summarize_entity_transitions(replay_df, unit_col, config=cfg)

    if len(summary_df) == 0:
        summary = {
            "units_evaluated": 0,
            "units_skipped": 0,
            "units_with_detection": 0,
            "units_without_detection": 0,
            "mean_remaining_life_normalized": None,
            "std_remaining_life_normalized": None,
            "mean_lead_time_cycles": None,
            "std_lead_time_cycles": None,
            "detection_params": {"min_cycle": min_cycle, "window": window, "threshold": threshold},
            "aggregate_detection": agg,
        }
        return summary_df, summary

    evaluated = len(summary_df)
    det_mask = summary_df["detected"].astype(bool)
    detected_count = int(det_mask.sum())
    rln_list = [
        float(x)
        for x in summary_df.loc[det_mask, "remaining_life_normalized"].dropna()
        if pd.notna(x)
    ]
    lead_list = [float(x) for x in summary_df.loc[det_mask, "lead_time"].dropna() if pd.notna(x)]

    legacy_rows: list[dict[str, Any]] = []
    for _, r in summary_df.iterrows():
        legacy_rows.append(
            {
                "unit": r["entity_id"],
                "detected": bool(r["detected"]),
                "transition_cycle": r.get("transition_position"),
                "remaining_life_normalized": r.get("remaining_life_normalized"),
                "lead_time_cycles": r.get("lead_time"),
                "reason": r.get("reason") or r.get("detection_reason"),
                "normalized_progress_at_detection": r.get("normalized_progress_at_detection"),
                "confidence": r.get("confidence"),
            }
        )
    legacy = pd.DataFrame(legacy_rows)

    summary = {
        "units_evaluated": evaluated,
        "units_skipped": evaluated - detected_count,
        "units_with_detection": detected_count,
        "units_without_detection": evaluated - detected_count,
        "mean_remaining_life_normalized": float(np.mean(rln_list)) if rln_list else None,
        "std_remaining_life_normalized": float(np.std(rln_list)) if rln_list else None,
        "mean_lead_time_cycles": float(np.mean(lead_list)) if lead_list else None,
        "std_lead_time_cycles": float(np.std(lead_list)) if lead_list else None,
        "detection_params": {"min_cycle": min_cycle, "window": window, "threshold": threshold},
        "aggregate_detection": agg,
    }
    return legacy, summary


def save_transition_artifacts(
    summary_df: pd.DataFrame,
    json_summary: dict[str, Any],
    *,
    csv_path: Path,
    json_path: Path | None = None,
) -> None:
    """Write per-unit CSV and optional JSON summary."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(json_summary, indent=2, default=str), encoding="utf-8")


def plot_unit_transition(
    unit_df: pd.DataFrame,
    detection: dict[str, Any],
    *,
    unit_id: int,
    output_path: Path | None = None,
    min_cycle: int = 30,
    pressure_col: str = "transition_pressure",
    cycle_col: str = "cycle",
    show: bool = True,
) -> None:
    """Single-unit plot: normalized remaining life (cycles), smoothed pressure, transition line."""
    import matplotlib.pyplot as plt

    df = unit_df.sort_values(cycle_col)
    c = pd.to_numeric(df[cycle_col], errors="coerce")
    cmax = float(c.max())
    rem = (cmax - c).clip(lower=0.0)
    rnorm = rem / max(EPS, cmax - c.min())

    pr = pd.to_numeric(df[pressure_col], errors="coerce").interpolate(limit_direction="both").ffill().bfill()
    pmin, pmax = float(pr.min()), float(pr.max())
    span = max(EPS, pmax - pmin)
    pnorm = ((pr - pmin) / span).clip(0.0, 1.0)
    smooth = pnorm.rolling(window=5, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(c, rnorm, label="Normalized remaining life (cycles)", color="C0")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Norm. remaining life", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(c, smooth, label="Smoothed norm. pressure", color="C1", alpha=0.85)
    ax2.set_ylabel("Smoothed pressure [0-1]", color="C1")

    tc = detection.get("transition_cycle") or detection.get("transition_position")
    if tc is not None:
        ax1.axvline(float(tc), color="red", ls="--", lw=1.2, label="Detected transition")

    fig.suptitle(f"FD004 unit {unit_id} (cycles ≥ {min_cycle} used in detector)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_transition_histograms(
    summary_df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Histograms: remaining life at detection and transition cycle (detected rows only)."""
    import matplotlib.pyplot as plt

    det = summary_df["detected"].map(lambda x: str(x).lower() in ("true", "1", "yes"))
    sub = summary_df[det]
    if len(sub) == 0:
        return

    col_rln = "remaining_life_normalized" if "remaining_life_normalized" in sub.columns else None
    col_tc = "transition_cycle" if "transition_cycle" in sub.columns else "transition_position"

    if col_rln and sub[col_rln].notna().any():
        _, ax1 = plt.subplots(figsize=(8, 4))
        plot_normalized_detection_histogram(
            sub[col_rln].dropna(),
            title="Remaining life at detection (normalized)",
            ax=ax1,
            show=False,
        )
        plt.tight_layout()
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_figure(output_dir / "fd004_hist_remaining_life.png")
        elif show:
            plt.show()
            plt.close()
        else:
            plt.close()

    if col_tc in sub.columns and sub[col_tc].notna().any():
        _, ax2 = plt.subplots(figsize=(8, 4))
        plot_raw_position_histogram(
            sub[col_tc].dropna(),
            title="Transition cycle",
            ax=ax2,
            show=False,
        )
        plt.tight_layout()
        if output_dir is not None:
            save_figure(output_dir / "fd004_hist_transition_cycle.png")
        elif show:
            plt.show()
            plt.close()
        else:
            plt.close()
