"""
Diagnostic plotting for the FD00x structural early-warning framework.

All plots are designed to be *decision-grade*: a reviewer should be able to
understand at a glance WHY a warning fired (or didn't), where the healthy
region ends, and how the anomaly score relates to the threshold.

Each diagnostic plot includes:
  • Normalised signals (top panel)
  • Raw structural drift and EMA anomaly score (middle panel)
  • Threshold line, warning line, degradation-onset line
  • Healthy-region shading

Plots are saved to files automatically — no interactive display is required.
Matplotlib is imported lazily so that non-plotting code paths never fail due
to a missing display.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _require_matplotlib():
    """Lazy import to avoid hard dependency on display environments."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting.  "
            "Install it with: pip install matplotlib"
        ) from exc


def _save_or_show(plt, path: Optional[str]) -> None:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Unit diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────


def plot_unit_diagnosis(
    unit_id: int,
    sensor_data: np.ndarray,
    raw_drift: np.ndarray,
    ema_drift: np.ndarray,
    threshold: float,
    warning_index: Optional[int],
    healthy_end_index: int,
    degradation_index: int,
    save_path: Optional[str] = None,
    title_suffix: str = "",
) -> None:
    """
    Full three-panel diagnostic plot for one unit.

    Panel 1 — Normalised sensor signals
        Shows how individual sensor readings evolve.  The healthy-region
        shading makes clear which portion was used as the reference.

    Panel 2 — Structural drift scores
        Shows raw drift (thin) and EMA-smoothed anomaly score (thick).
        The threshold line shows the exact level at which the warning rule
        triggers.  The warning line and degradation line show when the system
        actually warned vs. when ground-truth degradation started.

    Panel 3 — Exceedance indicator
        Binary bar showing which timesteps had EMA >= threshold.  Gives an
        intuitive view of the persistence count leading to confirmation.

    Parameters
    ----------
    unit_id           : Unit identifier for the title.
    sensor_data       : (n_cycles, n_sensors) raw (un-normalised) sensor matrix.
    raw_drift         : (n_cycles,) raw structural drift scores.
    ema_drift         : (n_cycles,) EMA-smoothed anomaly scores.
    threshold         : Absolute EMA threshold.
    warning_index     : Confirmed warning cycle index, or None.
    healthy_end_index : Last cycle of the frozen reference window.
    degradation_index : Estimated degradation onset index.
    save_path         : File path to save the figure.  None = skip saving.
    title_suffix      : Appended to the figure title (e.g. "(best unit)").
    """
    plt = _require_matplotlib()
    n_cycles = len(raw_drift)
    cycles = np.arange(n_cycles)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Unit {unit_id} — Structural Drift Diagnosis {title_suffix}",
        fontsize=13,
        fontweight="bold",
    )

    # ── Panel 1: Normalised sensor signals ───────────────────────────────────
    ax1 = axes[0]
    normed = (sensor_data - sensor_data.mean(axis=0)) / (
        sensor_data.std(axis=0) + 1e-8
    )
    for col in range(normed.shape[1]):
        ax1.plot(cycles, normed[:, col], alpha=0.25, linewidth=0.6)
    _add_reference_shading(ax1, healthy_end_index, n_cycles)
    _add_event_lines(ax1, warning_index, degradation_index, threshold=None,
                     draw_threshold=False)
    ax1.set_ylabel("Normalised sensor value", fontsize=9)
    ax1.set_title("Sensor signals (normalised)", fontsize=10)
    ax1.set_ylim(-4, 4)

    # ── Panel 2: Drift scores ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(cycles, raw_drift, color="#aaaaaa", linewidth=0.8,
             label="Raw structural drift", zorder=2)
    ax2.plot(cycles, ema_drift, color="#1f77b4", linewidth=1.8,
             label="EMA anomaly score", zorder=3)
    ax2.axhline(threshold, color="#d62728", linewidth=1.2, linestyle="--",
                label=f"Threshold ({threshold:.3f})", zorder=4)
    _add_reference_shading(ax2, healthy_end_index, n_cycles)
    _add_event_lines(ax2, warning_index, degradation_index, threshold=threshold,
                     draw_threshold=False)
    ax2.set_ylabel("Drift score", fontsize=9)
    ax2.set_title(
        "Structural drift: raw (grey) and EMA-smoothed anomaly score (blue)",
        fontsize=10,
    )
    ax2.legend(fontsize=8, loc="upper left")

    # ── Panel 3: Exceedance indicator ─────────────────────────────────────────
    ax3 = axes[2]
    exceed = (ema_drift >= threshold).astype(float)
    ax3.bar(cycles, exceed, width=1.0, color="#d62728", alpha=0.6,
            label="EMA ≥ threshold")
    if warning_index is not None:
        ax3.axvline(warning_index, color="#d62728", linewidth=2.0,
                    label=f"Warning confirmed @ {warning_index}")
    _add_reference_shading(ax3, healthy_end_index, n_cycles)
    ax3.axvline(degradation_index, color="#2ca02c", linewidth=1.5,
                linestyle=":", label=f"Degradation onset @ {degradation_index}")
    ax3.set_ylabel("Exceedance", fontsize=9)
    ax3.set_xlabel("Cycle", fontsize=9)
    ax3.set_title(
        "Persistence indicator — consecutive exceedances leading to warning confirmation",
        fontsize=10,
    )
    ax3.legend(fontsize=8, loc="upper left")
    ax3.set_ylim(-0.1, 1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_or_show(plt, save_path)


def _add_reference_shading(ax, healthy_end: int, n_cycles: int) -> None:
    """Shade the healthy reference region in light green."""
    ax.axvspan(0, healthy_end, alpha=0.07, color="#2ca02c", label="Healthy reference")


def _add_event_lines(
    ax,
    warning_index: Optional[int],
    degradation_index: int,
    threshold: Optional[float],
    draw_threshold: bool = True,
) -> None:
    """Draw vertical lines for warning and degradation onset."""
    if warning_index is not None:
        ax.axvline(
            warning_index,
            color="#d62728",
            linewidth=1.8,
            linestyle="-",
            label=f"Warning @ {warning_index}",
            zorder=5,
        )
    ax.axvline(
        degradation_index,
        color="#2ca02c",
        linewidth=1.5,
        linestyle=":",
        label=f"Degradation onset @ {degradation_index}",
        zorder=5,
    )
    if draw_threshold and threshold is not None:
        ax.axhline(threshold, color="#d62728", linewidth=1.0, linestyle="--")


# ─────────────────────────────────────────────────────────────────────────────
# Summary / comparison plots
# ─────────────────────────────────────────────────────────────────────────────


def plot_baseline_comparison(
    comparison: Dict[str, object],  # method → AggregateMetrics
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart comparing structural detector vs. baselines on three metrics.

    Parameters
    ----------
    comparison : dict from compare_baselines() {method_name: AggregateMetrics}
    save_path  : Output file path.
    """
    plt = _require_matplotlib()
    from dataclasses import asdict

    methods = list(comparison.keys())
    metrics_list = [asdict(comparison[m]) for m in methods]

    coverage = [m["coverage"] for m in metrics_list]
    fpr = [m["false_positive_rate"] for m in metrics_list]
    mean_lead = [m["mean_lead"] for m in metrics_list]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, coverage, width, label="Coverage", color="#1f77b4")
    ax.bar(x, fpr, width, label="FPR (↓ better)", color="#d62728")

    ax2 = ax.twinx()
    ax2.bar(x + width, mean_lead, width, label="Mean lead (cycles)", color="#2ca02c",
            alpha=0.8)
    ax2.set_ylabel("Mean lead (cycles)", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Rate [0–1]", fontsize=9)
    ax.set_title("Baseline comparison: structural detector vs. simple methods",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    _save_or_show(plt, save_path)


def plot_lead_time_histogram(
    results,  # List[UnitResult]
    save_path: Optional[str] = None,
    title: str = "Lead-time distribution",
) -> None:
    """
    Histogram of per-unit lead times (cycles from warning to degradation onset).

    Positive = early warning.  Negative = late warning (after degradation onset).
    """
    plt = _require_matplotlib()

    leads = [
        r.lead_vs_degradation
        for r in results
        if r.covered and r.lead_vs_degradation is not None
    ]
    if not leads:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(leads, bins=30, edgecolor="white", color="#1f77b4", alpha=0.85)
    ax.axvline(0, color="#d62728", linewidth=1.5, linestyle="--",
               label="Degradation onset")
    ax.axvline(float(np.mean(leads)), color="#2ca02c", linewidth=1.5,
               linestyle="-", label=f"Mean lead = {np.mean(leads):.1f}")
    ax.set_xlabel("Lead time (cycles): positive = early warning", fontsize=9)
    ax.set_ylabel("Units", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_or_show(plt, save_path)


def plot_tuning_heatmap(
    tuning_rows,  # List[TuningRow]
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap of tuning scores as a function of threshold_std × persistence.

    If multiple healthy_fraction values were swept, each gets its own facet.
    """
    plt = _require_matplotlib()

    if not tuning_rows:
        return

    import collections

    # Group by healthy_fraction
    by_hf: Dict[float, list] = collections.defaultdict(list)
    for row in tuning_rows:
        by_hf[row.healthy_fraction].append(row)

    hf_values = sorted(by_hf.keys())
    n_facets = len(hf_values)
    fig, axes = plt.subplots(1, n_facets, figsize=(5 * n_facets, 4), squeeze=False)
    fig.suptitle("Tuning grid: composite score", fontsize=12, fontweight="bold")

    for col_idx, hf in enumerate(hf_values):
        rows = by_hf[hf]
        thresholds = sorted(set(r.threshold_std for r in rows))
        persistences = sorted(set(r.persistence for r in rows))
        grid = np.full((len(persistences), len(thresholds)), np.nan)
        for r in rows:
            ti = thresholds.index(r.threshold_std)
            pi = persistences.index(r.persistence)
            grid[pi, ti] = r.score

        ax = axes[0][col_idx]
        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f"{t:.1f}" for t in thresholds], fontsize=8)
        ax.set_yticks(range(len(persistences)))
        ax.set_yticklabels([str(p) for p in persistences], fontsize=8)
        ax.set_xlabel("threshold_std", fontsize=9)
        ax.set_ylabel("persistence", fontsize=9)
        ax.set_title(f"healthy_frac={hf:.2f}", fontsize=10)
        plt.colorbar(im, ax=ax, label="score")

        # Mark best cell
        if not np.all(np.isnan(grid)):
            best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
            ax.add_patch(
                plt.Rectangle(
                    (best_idx[1] - 0.5, best_idx[0] - 0.5),
                    1, 1,
                    linewidth=2, edgecolor="black", facecolor="none",
                )
            )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save_or_show(plt, save_path)


def plot_cross_dataset_summary(
    summary: Dict[str, object],  # dataset → AggregateMetrics
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart of coverage / FPR / mean_lead across datasets.
    """
    plt = _require_matplotlib()
    from dataclasses import asdict

    if not summary:
        return

    datasets = list(summary.keys())
    dicts = [asdict(summary[d]) for d in datasets]

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, [d["coverage"] for d in dicts], width,
           label="Coverage", color="#1f77b4")
    ax.bar(x, [d["false_positive_rate"] for d in dicts], width,
           label="FPR", color="#d62728")

    ax2 = ax.twinx()
    ax2.bar(x + width, [d["mean_lead"] for d in dicts], width,
            label="Mean lead", color="#2ca02c", alpha=0.8)
    ax2.set_ylabel("Mean lead (cycles)", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Rate [0–1]", fontsize=9)
    ax.set_title("Cross-dataset performance summary", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    _save_or_show(plt, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Batch unit-level plot generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_unit_plots(
    config,               # DetectorConfig
    unit_data: Dict[int, np.ndarray],
    sensor_cols: List[int],
    results,              # List[UnitResult]
    output_dir: str,
    max_units: int = 5,
    select: str = "best_worst_fp",
) -> None:
    """
    Generate diagnostic plots for a selected subset of units.

    Parameters
    ----------
    select : "best_worst_fp" | "all"
        "best_worst_fp" plots the best-lead, worst-lead, and a FP example.
        "all" plots every unit (can be slow for large datasets).
    """
    from .detector import StructuralDriftDetector

    detector = StructuralDriftDetector(config)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if select == "best_worst_fp":
        units_to_plot = _select_representative(results, max_units)
    else:
        units_to_plot = [(r, "") for r in results[:max_units]]

    for result, label in units_to_plot:
        uid = result.unit_id
        raw = unit_data.get(uid)
        if raw is None:
            continue

        data = raw[:, sensor_cols]
        scores = detector.process_unit(data)

        save_path = os.path.join(plots_dir, f"unit_{uid:04d}_{label}.png")
        plot_unit_diagnosis(
            unit_id=uid,
            sensor_data=data,
            raw_drift=scores.raw_drift,
            ema_drift=scores.ema_drift,
            threshold=scores.threshold,
            warning_index=scores.warning_index,
            healthy_end_index=result.healthy_end_index,
            degradation_index=result.degradation_index,
            save_path=save_path,
            title_suffix=f"({label})" if label else "",
        )


def _select_representative(
    results,  # List[UnitResult]
    max_units: int,
) -> List[Tuple]:
    """Pick best-lead, worst-lead, and first FP for plotting."""
    covered = [r for r in results if r.covered and not r.false_positive]
    fps = [r for r in results if r.false_positive]

    selected = []

    if covered:
        best = max(covered, key=lambda r: r.lead_vs_degradation or -9999)
        worst = min(covered, key=lambda r: r.lead_vs_degradation or 9999)
        selected.append((best, "best_lead"))
        if worst.unit_id != best.unit_id:
            selected.append((worst, "worst_lead"))

    if fps:
        selected.append((fps[0], "false_positive"))

    # Fill remaining slots with covered units
    used_ids = {r.unit_id for r, _ in selected}
    for r in covered:
        if len(selected) >= max_units:
            break
        if r.unit_id not in used_ids:
            selected.append((r, ""))
            used_ids.add(r.unit_id)

    return selected[:max_units]
