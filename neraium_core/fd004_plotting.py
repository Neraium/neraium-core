from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any

PHASE_COLORS = {
    "stable": "#DFF3E3",
    "drift": "#FFF4CC",
    "unstable": "#FDE2E2",
}
PHASE_ORDER = ("stable", "drift", "unstable")


def select_representative_units(
    unit_summaries: list[dict[str, Any]],
    max_units: int = 3,
) -> list[str]:
    """
    Select representative units by low/median/high peak instability.

    Falls back to lexical asset order when instability values are missing.
    """
    if max_units <= 0 or not unit_summaries:
        return []

    ordered = sorted(unit_summaries, key=lambda item: item.get("asset_id", ""))
    with_peak = [item for item in ordered if isinstance(item.get("peak_instability"), (int, float))]

    if len(with_peak) < max_units:
        return [item["asset_id"] for item in ordered[:max_units]]

    by_peak = sorted(with_peak, key=lambda item: float(item["peak_instability"]))
    low = by_peak[0]["asset_id"]
    high = by_peak[-1]["asset_id"]
    peak_values = [float(item["peak_instability"]) for item in by_peak]
    median_peak = median(peak_values)
    median_item = min(by_peak, key=lambda item: abs(float(item["peak_instability"]) - median_peak))

    picked: list[str] = []
    for asset_id in (low, median_item["asset_id"], high):
        if asset_id not in picked:
            picked.append(asset_id)

    if len(picked) < max_units:
        for item in by_peak:
            asset_id = item["asset_id"]
            if asset_id not in picked:
                picked.append(asset_id)
            if len(picked) == max_units:
                break

    return picked[:max_units]


def plot_fd004_unit_timeseries(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    include_rul_curve: bool = True,
    title: str | None = None,
) -> None:
    """Plot drift and instability with phase overlays for one FD004 unit."""
    if not rows:
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sorted_rows = sorted(rows, key=lambda row: int(row["cycle"]))
    cycles = [int(row["cycle"]) for row in sorted_rows]
    drift = [float(row.get("structural_drift_score", 0.0)) for row in sorted_rows]
    instability = [float(row.get("composite_instability", 0.0)) for row in sorted_rows]
    rul_values = [float(row.get("estimated_rul", 0.0)) for row in sorted_rows]
    phases = [str(row.get("phase", "stable")) for row in sorted_rows]
    asset_id = str(sorted_rows[0].get("asset_id", "unknown"))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    run_start = cycles[0]
    current_phase = phases[0]
    for index in range(1, len(cycles)):
        if phases[index] == current_phase:
            continue
        ax.axvspan(run_start, cycles[index], color=PHASE_COLORS.get(current_phase, "#EFEFEF"), alpha=0.4)
        run_start = cycles[index]
        current_phase = phases[index]
    ax.axvspan(run_start, cycles[-1], color=PHASE_COLORS.get(current_phase, "#EFEFEF"), alpha=0.4)

    drift_line = ax.plot(
        cycles,
        drift,
        label="Structural drift score",
        color="#2F4858",
        linewidth=2,
    )[0]
    instability_line = ax.plot(
        cycles,
        instability,
        label="Composite instability",
        color="#1F77B4",
        linewidth=2,
    )[0]

    legend_handles = [drift_line, instability_line]

    if include_rul_curve:
        ax2 = ax.twinx()
        rul_line = ax2.plot(
            cycles,
            rul_values,
            label="Estimated RUL",
            color="#7A5195",
            linewidth=1.6,
            linestyle="--",
        )[0]
        ax2.set_ylabel("Estimated RUL", color="#7A5195")
        ax2.tick_params(axis="y", labelcolor="#7A5195")
        legend_handles.append(rul_line)

    phase_handles = [
        Patch(facecolor=PHASE_COLORS[name], alpha=0.4, edgecolor="none", label=f"Phase: {name}")
        for name in PHASE_ORDER
    ]
    legend_handles.extend(phase_handles)

    ax.set_title(title or f"FD004 SII evaluation: {asset_id}")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def generate_fd004_subset_plots(
    timeseries: list[dict[str, Any]],
    unit_summaries: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    max_units: int = 3,
    include_rul_curve: bool = True,
) -> list[Path]:
    """Generate per-unit FD004 visuals for representative units."""
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for row in timeseries:
        asset_id = str(row.get("asset_id", ""))
        by_unit.setdefault(asset_id, []).append(row)

    selected = select_representative_units(unit_summaries, max_units=max_units)
    plots_dir = Path(output_dir) / "plots"
    generated: list[Path] = []
    for asset_id in selected:
        if asset_id not in by_unit:
            continue
        out_path = plots_dir / f"{asset_id}_sii_plot.png"
        plot_fd004_unit_timeseries(by_unit[asset_id], out_path, include_rul_curve=include_rul_curve)
        generated.append(out_path)
    return generated
