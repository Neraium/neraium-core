from __future__ import annotations

from pathlib import Path
from typing import Any


def select_representative_units(unit_summaries: list[dict[str, Any]], max_units: int = 3) -> list[str]:
    if not unit_summaries:
        return []
    ordered = sorted(unit_summaries, key=lambda s: float(s.get("peak_instability", 0.0)))
    if max_units <= 1:
        return [str(ordered[0].get("asset_id"))]
    mid = ordered[len(ordered) // 2]
    picks = [ordered[0], mid, ordered[-1]]
    dedup: list[str] = []
    for row in picks:
        asset = str(row.get("asset_id"))
        if asset not in dedup:
            dedup.append(asset)
        if len(dedup) >= max_units:
            break
    return dedup


def select_hero_unit(unit_summaries: list[dict[str, Any]], timeseries: list[dict[str, Any]]) -> str | None:
    candidates = [s for s in unit_summaries if s.get("first_MEDIUM_step") is not None and s.get("first_HIGH_step") is not None]
    if not candidates:
        return str(unit_summaries[0].get("asset_id")) if unit_summaries else None

    def _roughness(asset_id: str) -> float:
        values = [float(r.get("composite_instability", 0.0)) for r in timeseries if str(r.get("asset_id")) == asset_id]
        if len(values) < 2:
            return 0.0
        return sum(abs(values[i] - values[i - 1]) for i in range(1, len(values)))

    ranked = sorted(candidates, key=lambda s: (_roughness(str(s.get("asset_id"))), float(s.get("peak_instability", 0.0))))
    return str(ranked[0].get("asset_id"))


def plot_fd004_unit_timeseries(rows: list[dict[str, Any]], output_path: str | Path, include_rul_curve: bool = True) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return

    x = [float(r.get("cycle", 0.0)) for r in rows]
    y = [float(r.get("composite_instability", r.get("instability", 0.0))) for r in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def generate_fd004_hero_plot(timeseries: list[dict[str, Any]], unit_summaries: list[dict[str, Any]], *, output_path: str | Path, include_rul_curve: bool = True) -> tuple[str | None, bool]:
    hero = select_hero_unit(unit_summaries, timeseries)
    print(f"Selected hero unit: {hero}")
    if not hero:
        return None, False
    rows = [r for r in timeseries if str(r.get("asset_id")) == hero]
    try:
        plot_fd004_unit_timeseries(rows, output_path, include_rul_curve=include_rul_curve)
    except ImportError:
        print("Skipping hero FD004 plot because matplotlib is unavailable")
        return hero, False
    return hero, True


def generate_fd004_subset_plots(timeseries: list[dict[str, Any]], unit_summaries: list[dict[str, Any]], *, output_dir: str | Path, max_units: int = 3, include_rul_curve: bool = True) -> list[Path]:
    output_base = Path(output_dir) / "plots"
    output_base.mkdir(parents=True, exist_ok=True)
    selected = select_representative_units(unit_summaries, max_units=max_units)
    generated: list[Path] = []
    for unit in selected:
        rows = [r for r in timeseries if str(r.get("asset_id")) == unit]
        out = output_base / f"{unit}_sii_plot.png"
        plot_fd004_unit_timeseries(rows, out, include_rul_curve=include_rul_curve)
        generated.append(out)
    return generated


__all__ = [
    "select_hero_unit",
    "select_representative_units",
    "plot_fd004_unit_timeseries",
    "generate_fd004_hero_plot",
    "generate_fd004_subset_plots",
]
