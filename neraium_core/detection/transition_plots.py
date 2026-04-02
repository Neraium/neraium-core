"""Reusable matplotlib helpers for transition detection visualization (any dataset)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_timeseries_with_transition(
    df: pd.DataFrame,
    *,
    index_column: str = "cycle",
    signal_column: str = "transition_pressure",
    transition_position: float | None = None,
    title: str | None = None,
    ax: Any = None,
    show: bool = True,
) -> Any:
    """Plot a signal over index with optional vertical marker at detected transition."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    x = pd.to_numeric(df[index_column], errors="coerce")
    y = pd.to_numeric(df[signal_column], errors="coerce")
    ax.plot(x, y, linewidth=1.2, label=signal_column)
    if transition_position is not None:
        ax.axvline(transition_position, color="red", linestyle="--", linewidth=1.5, label="transition")
    ax.set_xlabel(index_column)
    ax.set_ylabel(signal_column)
    ax.set_title(title or f"{signal_column} vs {index_column}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_normalized_detection_histogram(
    normalized_progress: pd.Series | np.ndarray,
    *,
    title: str = "Normalized detection progress",
    bins: int = 20,
    ax: Any = None,
    show: bool = True,
) -> Any:
    """Histogram of normalized_progress_at_detection (0–1 life consumed)."""
    s = np.asarray(pd.to_numeric(pd.Series(normalized_progress), errors="coerce"), dtype=float)
    s = s[np.isfinite(s)]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("normalized progress at detection")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_raw_position_histogram(
    positions: pd.Series | np.ndarray,
    *,
    title: str = "Raw transition positions",
    bins: int = 30,
    ax: Any = None,
    show: bool = True,
) -> Any:
    """Histogram of raw index values at detection (e.g. cycles)."""
    s = np.asarray(pd.to_numeric(pd.Series(positions), errors="coerce"), dtype=float)
    s = s[np.isfinite(s)]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=bins, color="coral", edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("transition position")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_signal_vs_outcome(
    df: pd.DataFrame,
    *,
    index_column: str = "cycle",
    signal_column: str = "transition_pressure",
    outcome_column: str | None = None,
    transition_position: float | None = None,
    title: str | None = None,
    ax: Any = None,
    show: bool = True,
) -> Any:
    """Optional dual-axis: structural signal vs outcome (e.g. RUL) when ``outcome_column`` is set."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    x = pd.to_numeric(df[index_column], errors="coerce")
    y1 = pd.to_numeric(df[signal_column], errors="coerce")
    ax.plot(x, y1, color="C0", label=signal_column)
    if transition_position is not None:
        ax.axvline(transition_position, color="red", linestyle="--", linewidth=1.2, label="transition")
    ax.set_xlabel(index_column)
    ax.set_ylabel(signal_column, color="C0")
    if outcome_column and outcome_column in df.columns:
        ax2 = ax.twinx()
        y2 = pd.to_numeric(df[outcome_column], errors="coerce")
        ax2.plot(x, y2, color="C1", alpha=0.8, label=outcome_column)
        ax2.set_ylabel(outcome_column, color="C1")
    ax.set_title(title or signal_column)
    ax.grid(True, alpha=0.3)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def save_figure(path: str | Path, dpi: int = 120) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
