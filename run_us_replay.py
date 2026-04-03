#!/usr/bin/env python3
"""Run a deterministic US market replay and export structured signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import random
import pandas as pd

from neraium_core.markets.evidence.evidence_log import EvidenceLog
from neraium_core.markets.replay import run_signal_replay
from neraium_core.markets.signals.emission_control import EmissionDecision, SignalEmissionController

TIMEFRAME = "15m"
DATA_ROOT = Path("neraium_core/markets/sample_data")
OUTPUT_CSV = Path("artifacts/replay_output.csv")
EVIDENCE_PATH = Path("artifacts/neraium_markets/replay_evidence.jsonl")
ROWS = 240
SEED = 1729

# Requested synthetic assets + pipeline-required assets for state vector construction.
SYNTHETIC_UNIVERSE = [
    "SPY",
    "AAPL",
    "NVDA",
    "QQQ",
    "IWM",
    "XLF",
    "XLK",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "DXY",
    "US2Y",
    "US10Y",
    "CRUDE",
    "GOLD",
    "VIX",
]


@dataclass
class ReplayStats:
    total_emitted: int = 0
    suppressed: int = 0
    abstentions: int = 0


class TrackingEmissionController(SignalEmissionController):
    """Track suppression and abstention counts without changing core engine logic."""

    def __init__(self) -> None:
        super().__init__()
        self.stats = ReplayStats()

    def evaluate(self, *args, abstain: bool, **kwargs) -> EmissionDecision:
        decision = super().evaluate(*args, abstain=abstain, **kwargs)
        if abstain:
            self.stats.abstentions += 1
        if not decision.emit:
            self.stats.suppressed += 1
        return decision


def _build_series(index: pd.DatetimeIndex, start: float, drift: float, noise: float, seed_offset: int) -> pd.Series:
    rng = random.Random(SEED + seed_offset)
    values: list[float] = []
    current = start
    for _ in index:
        current += drift + rng.gauss(0.0, noise)
        current = max(current, 0.5)
        values.append(current)
    return pd.Series(values, index=index)


def _write_asset_csv(timeframe_dir: Path, asset: str, close_series: pd.Series) -> None:
    df = pd.DataFrame(
        {
            "timestamp": close_series.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "close": close_series.round(4),
        }
    )
    df.to_csv(timeframe_dir / f"{asset}.csv", index=False)


def ensure_sample_data(data_root: Path, timeframe: str) -> None:
    timeframe_dir = data_root / timeframe
    existing = sorted(timeframe_dir.glob("*.csv")) if timeframe_dir.exists() else []
    if existing:
        return

    timeframe_dir.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp("2025-12-31 20:00:00", tz="UTC")
    index = pd.date_range(end=end, periods=ROWS, freq="15min", tz="UTC")

    params = {
        "SPY": (470.0, 0.06, 0.9),
        "AAPL": (190.0, 0.05, 1.1),
        "NVDA": (780.0, 0.08, 2.2),
        "QQQ": (400.0, 0.07, 1.0),
        "IWM": (210.0, 0.04, 0.8),
        "XLF": (38.0, 0.01, 0.12),
        "XLK": (175.0, 0.03, 0.25),
        "XLE": (88.0, 0.02, 0.4),
        "XLV": (132.0, 0.02, 0.2),
        "XLI": (108.0, 0.015, 0.18),
        "XLY": (165.0, 0.025, 0.22),
        "XLP": (72.0, 0.008, 0.07),
        "XLU": (66.0, 0.006, 0.06),
        "DXY": (103.0, -0.002, 0.04),
        "US2Y": (4.9, -0.0005, 0.015),
        "US10Y": (4.2, -0.0003, 0.012),
        "CRUDE": (76.0, 0.01, 0.5),
        "GOLD": (2050.0, 0.12, 2.0),
        "VIX": (16.0, -0.005, 0.15),
    }

    for i, asset in enumerate(SYNTHETIC_UNIVERSE):
        start, drift, noise = params[asset]
        series = _build_series(index, start, drift, noise, i)
        _write_asset_csv(timeframe_dir, asset, series)


def main() -> None:
    ensure_sample_data(DATA_ROOT, TIMEFRAME)

    evidence_log = EvidenceLog(path=EVIDENCE_PATH)
    controller = TrackingEmissionController()

    rows = run_signal_replay(
        data_dir=DATA_ROOT,
        evidence_log=evidence_log,
        timeframe=TIMEFRAME,
        csv_output_path=OUTPUT_CSV,
        emission_controller=controller,
    )

    controller.stats.total_emitted = len(rows)
    print(f"Replay complete: timeframe={TIMEFRAME}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Total signals emitted: {controller.stats.total_emitted}")
    print(f"Suppressed signals: {controller.stats.suppressed}")
    print(f"Abstentions: {controller.stats.abstentions}")


if __name__ == "__main__":
    main()
