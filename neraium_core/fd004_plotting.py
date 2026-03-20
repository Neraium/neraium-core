"""
Compatibility wrapper for FD004 plotting helpers.

The implementation lives under `examples/fd004/fd004_plotting.py`.

This wrapper is intentionally written so `tests/` can monkeypatch
`plot_fd004_unit_timeseries` and have it affect hero/subset plot generation.
"""

from __future__ import annotations

from typing import Any

import examples.fd004.fd004_plotting as impl

# Re-export selectors (pure functions; safe to bind directly).
select_hero_unit = impl.select_hero_unit
select_representative_units = impl.select_representative_units

# Re-export the plot function symbol itself. Tests monkeypatch this name.
plot_fd004_unit_timeseries = impl.plot_fd004_unit_timeseries

__all__ = [
    "select_hero_unit",
    "select_representative_units",
    "plot_fd004_unit_timeseries",
    "generate_fd004_hero_plot",
    "generate_fd004_subset_plots",
]


def generate_fd004_hero_plot(
    timeseries: list[dict[str, Any]],
    unit_summaries: list[dict[str, Any]],
    *,
    output_path: str | Any,
    include_rul_curve: bool = True,
) -> tuple[str | None, bool]:
    # Ensure monkeypatched plot function is used by the implementation module.
    impl.plot_fd004_unit_timeseries = plot_fd004_unit_timeseries
    return impl.generate_fd004_hero_plot(
        timeseries,
        unit_summaries,
        output_path=output_path,
        include_rul_curve=include_rul_curve,
    )


def generate_fd004_subset_plots(
    timeseries: list[dict[str, Any]],
    unit_summaries: list[dict[str, Any]],
    *,
    output_dir: str | Any,
    max_units: int = 3,
    include_rul_curve: bool = True,
) -> list[Any]:
    impl.plot_fd004_unit_timeseries = plot_fd004_unit_timeseries
    return impl.generate_fd004_subset_plots(
        timeseries,
        unit_summaries,
        output_dir=output_dir,
        max_units=max_units,
        include_rul_curve=include_rul_curve,
    )

