"""
FD004 replay helpers: robust structural transition detection on per-unit timeseries.

Designed for post-warmup analysis using ``transition_pressure`` (or another column)
so detections are not dominated by initialization artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9


def detect_structural_transition(
    unit_df: pd.DataFrame,
    *,
    min_cycle: int = 30,
    window: int = 5,
    threshold: float = 0.5,
    pressure_col: str = "transition_pressure",
    cycle_col: str = "cycle",
    sustained_points: int = 2,
) -> dict[str, Any]:
    """
    Detect the first sustained post-warmup transition from normalized pressure.

    Steps: sort by cycle; drop rows with cycle < ``min_cycle``; coerce pressure to
    numeric; interpolate missing; normalize pressure to [0, 1] on the valid slice;
    smooth with rolling mean (``window``, ``min_periods=1``); find first run of
    ``sustained_points`` values strictly above ``threshold``. If none, fall back to
    the index of maximum smoothed first-difference (largest sustained slope bump).

    Returns a dict with ``detected``, ``transition_cycle``, ``remaining_life_normalized``,
    ``lead_time_cycles``, ``reason``, and ``metadata``. Never raises on bad input.
    """
    meta: dict[str, Any] = {
        "min_cycle": int(min_cycle),
        "window": int(window),
        "threshold": float(threshold),
        "pressure_col": pressure_col,
        "rows_input": int(len(unit_df)),
    }
    empty = {
        "detected": False,
        "transition_index": None,
        "transition_cycle": None,
        "remaining_life_normalized": None,
        "lead_time_cycles": None,
        "reason": "no_data",
        "metadata": meta,
    }

    if unit_df is None or len(unit_df) == 0 or cycle_col not in unit_df.columns:
        return empty

    df = unit_df.sort_values(cycle_col).copy()
    if pressure_col not in df.columns:
        return {**empty, "reason": f"missing_column:{pressure_col}"}

    df = df.loc[df[cycle_col] >= float(min_cycle)]
    meta["rows_after_min_cycle"] = int(len(df))
    if len(df) < sustained_points:
        return {**empty, "reason": "insufficient_post_warmup_rows"}

    raw = pd.to_numeric(df[pressure_col], errors="coerce")
    raw = raw.interpolate(limit_direction="both").ffill().bfill()
    if raw.isna().all() or float(raw.std(skipna=True) or 0.0) < EPS:
        return {**empty, "reason": "flat_or_invalid_pressure"}

    pmin = float(raw.min())
    pmax = float(raw.max())
    span = pmax - pmin
    if span < EPS:
        norm = pd.Series(0.5, index=raw.index, dtype=float)
    else:
        norm = (raw - pmin) / span
    norm = norm.clip(0.0, 1.0)

    smooth = norm.rolling(window=int(window), min_periods=1).mean()
    cycles = pd.to_numeric(df[cycle_col], errors="coerce").values
    sm = smooth.values.astype(float)
    n = len(sm)

    # First sustained run above threshold
    run = 0
    start_idx: int | None = None
    for i in range(n):
        if sm[i] > float(threshold):
            run += 1
            if run >= int(sustained_points):
                start_idx = i - int(sustained_points) + 1
                break
        else:
            run = 0

    reason = "threshold_crossing"
    t_idx: int | None = start_idx

    if t_idx is None:
        # Fallback: largest post-warmup bump in smoothed slope (avoid index 0 edge)
        d1 = np.diff(sm, prepend=sm[0])
        d1_smooth = pd.Series(d1).rolling(max(2, min(5, window)), min_periods=1).mean().values
        search_from = min(3, max(0, n - 1))
        if n <= search_from + 1:
            return {**empty, "reason": "no_threshold_no_fallback"}
        seg = d1_smooth[search_from:]
        if np.all(np.isnan(seg)):
            return {**empty, "reason": "no_threshold_no_fallback"}
        j = int(np.nanargmax(seg)) + search_from
        t_idx = j
        reason = "max_slope_increase"

    if t_idx is None or t_idx < 0 or t_idx >= n:
        return {**empty, "reason": "index_error"}

    c_sel = float(cycles[t_idx])
    c_max = float(np.nanmax(cycles))
    c_min = float(np.nanmin(cycles))
    lead = max(0.0, c_max - c_sel)
    denom = max(EPS, c_max - c_min)
    rln = float(np.clip(lead / denom, 0.0, 1.0))

    meta.update(
        {
            "pressure_min": pmin,
            "pressure_max": pmax,
            "normalized_pressure_tail": float(norm.iloc[t_idx]),
            "smoothed_at_transition": float(sm[t_idx]),
            "max_cycle": c_max,
        }
    )

    return {
        "detected": True,
        "transition_index": int(t_idx),
        "transition_cycle": c_sel,
        "remaining_life_normalized": rln,
        "lead_time_cycles": lead,
        "reason": reason,
        "metadata": meta,
    }


def summarize_fd004_transitions(
    replay_df: pd.DataFrame,
    *,
    unit_col: str = "unit",
    min_cycle: int = 30,
    window: int = 5,
    threshold: float = 0.5,
    pressure_col: str = "transition_pressure",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run :func:`detect_structural_transition` per unit and aggregate metrics.

    Returns (per_unit DataFrame, json-ready summary dict).
    """
    rows: list[dict[str, Any]] = []
    if replay_df is None or len(replay_df) == 0 or unit_col not in replay_df.columns:
        summary = {
            "units_evaluated": 0,
            "units_skipped": 0,
            "mean_remaining_life_normalized": None,
            "std_remaining_life_normalized": None,
            "mean_lead_time_cycles": None,
            "std_lead_time_cycles": None,
            "detection_params": {"min_cycle": min_cycle, "window": window, "threshold": threshold},
        }
        return pd.DataFrame(), summary

    evaluated = 0
    skipped = 0
    rln_list: list[float] = []
    lead_list: list[float] = []

    for uid, g in replay_df.groupby(unit_col):
        res = detect_structural_transition(
            g,
            min_cycle=min_cycle,
            window=window,
            threshold=threshold,
            pressure_col=pressure_col,
        )
        evaluated += 1
        det = bool(res.get("detected"))
        if not det:
            skipped += 1
        row = {
            "unit": int(uid) if pd.notna(uid) else uid,
            "detected": det,
            "transition_cycle": res.get("transition_cycle"),
            "remaining_life_normalized": res.get("remaining_life_normalized"),
            "lead_time_cycles": res.get("lead_time_cycles"),
            "reason": res.get("reason"),
        }
        if det and res.get("remaining_life_normalized") is not None:
            rln_list.append(float(res["remaining_life_normalized"]))
        if det and res.get("lead_time_cycles") is not None:
            lead_list.append(float(res["lead_time_cycles"]))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    detected_count = int(summary_df["detected"].sum()) if len(summary_df) else 0
    summary = {
        "units_evaluated": evaluated,
        "units_skipped": skipped,
        "units_with_detection": detected_count,
        "units_without_detection": evaluated - detected_count,
        "mean_remaining_life_normalized": float(np.mean(rln_list)) if rln_list else None,
        "std_remaining_life_normalized": float(np.std(rln_list)) if rln_list else None,
        "mean_lead_time_cycles": float(np.mean(lead_list)) if lead_list else None,
        "std_lead_time_cycles": float(np.std(lead_list)) if lead_list else None,
        "detection_params": {"min_cycle": min_cycle, "window": window, "threshold": threshold},
    }
    return summary_df, summary


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
        json_path.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")


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
    """Single-unit plot: normalized remaining life (proxy), smoothed pressure, transition line."""
    import matplotlib.pyplot as plt

    df = unit_df.sort_values(cycle_col)
    c = pd.to_numeric(df[cycle_col], errors="coerce")
    cmax = float(c.max())
    # Remaining life in cycles (same as FD004 convention: cycles left to max)
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

    tc = detection.get("transition_cycle")
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
    """Histograms: remaining life at detection and transition cycle (detected units only)."""
    import matplotlib.pyplot as plt

    det = summary_df["detected"].map(lambda x: str(x).lower() in ("true", "1", "yes"))
    sub = summary_df[det]
    if len(sub) == 0:
        return

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.hist(sub["remaining_life_normalized"].dropna(), bins=min(20, max(5, len(sub))), color="C0", edgecolor="white")
    ax1.set_title("Remaining life at detection (normalized)")
    ax1.set_xlabel("Value")
    fig1.tight_layout()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(output_dir / "fd004_hist_remaining_life.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(sub["transition_cycle"].dropna(), bins=min(20, max(5, len(sub))), color="C1", edgecolor="white")
    ax2.set_title("Transition cycle")
    ax2.set_xlabel("Cycle")
    fig2.tight_layout()
    if output_dir is not None:
        fig2.savefig(output_dir / "fd004_hist_transition_cycle.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig2)
