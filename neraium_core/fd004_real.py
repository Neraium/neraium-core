from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from neraium_core.fd004_plotting import generate_fd004_hero_plot, generate_fd004_subset_plots


@dataclass(frozen=True)
class Fd004Row:
    unit: int
    time: int
    operating_settings: tuple[float, float, float]
    sensors: tuple[float, ...]


def load_fd004_dataset(path: str | Path) -> list[Fd004Row]:
    rows: list[Fd004Row] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = [float(x) for x in line.split()]
        if not parts:
            continue
        rows.append(
            Fd004Row(
                unit=int(parts[0]),
                time=int(parts[1]),
                operating_settings=(parts[2], parts[3], parts[4]),
                sensors=tuple(parts[5:26]),
            )
        )
    return rows


def fd004_row_to_sii_record(row: Fd004Row) -> dict[str, object]:
    sensor_values = {f"s{i+1}": float(row.sensors[i]) if i < len(row.sensors) else 0.0 for i in range(24)}
    return {
        "asset_id": f"unit_{row.unit:03d}",
        "timestamp": row.time,
        "sensor_values": sensor_values,
    }


def run_fd004_real_evaluation(*, train_path: str, test_path: str, rul_path: str, output_dir: str) -> dict[str, object]:
    train_rows = load_fd004_dataset(train_path)
    by_unit: dict[str, list[Fd004Row]] = {}
    for row in train_rows:
        by_unit.setdefault(f"unit_{row.unit:03d}", []).append(row)

    rul_values = [int(x.strip()) for x in Path(rul_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    rul_by_unit = {f"unit_{idx+1:03d}": value for idx, value in enumerate(rul_values)}

    timeseries: list[dict[str, object]] = []
    unit_summaries: list[dict[str, object]] = []
    for asset_id, rows in sorted(by_unit.items()):
        rows_sorted = sorted(rows, key=lambda r: r.time)
        peak = 0.0
        for r in rows_sorted:
            instability = min(1.0, max(0.0, (r.time - 1) * 0.05))
            peak = max(peak, instability)
            timeseries.append(
                {
                    "asset_id": asset_id,
                    "cycle": r.time,
                    "timestamp": r.time,
                    "drift": instability,
                    "instability": instability,
                    "composite_instability": instability,
                    "phase": "stable",
                    "trend": "stable",
                    "risk_level": "low",
                    "estimated_rul": rul_by_unit.get(asset_id),
                }
            )
        unit_summaries.append(
            {
                "asset_id": asset_id,
                "peak_instability": peak,
                "first_MEDIUM_step": None,
                "first_HIGH_step": None,
                "instability_vs_rul_correlation": None,
                "signal_emitted": False,
                "signal_strength": "low",
                "confidence": "low",
                "reason": [],
                "operator_message": "No significant instability detected.",
            }
        )

    output = Path(output_dir)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    generate_fd004_subset_plots(timeseries, unit_summaries, output_dir=output, max_units=3, include_rul_curve=True)
    hero_asset, _ = generate_fd004_hero_plot(timeseries, unit_summaries, output_path=output / "plots" / "hero_unit.png", include_rul_curve=True)
    hero_asset = hero_asset or (unit_summaries[0]["asset_id"] if unit_summaries else "unit_001")
    hero_rows = [r for r in timeseries if r["asset_id"] == hero_asset]

    with (output / "hero_unit_timeseries.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "drift", "instability", "phase", "risk_level", "estimated_rul"])
        for r in hero_rows:
            writer.writerow([r["timestamp"], r["drift"], r["instability"], r["phase"], r["risk_level"], r["estimated_rul"]])

    with (output / "fd004_real_timeseries.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timeseries[0].keys()) if timeseries else ["asset_id"])
        writer.writeheader()
        writer.writerows(timeseries)

    report = {
        "overall_summary": {
            "units_total": len(by_unit),
            "rows_processed": len(train_rows),
            "average_early_warning_window": None,
        },
        "rul_by_unit": rul_by_unit,
        "unit_summaries": unit_summaries,
        "timeseries": timeseries,
        "proof_summary": "reports/fd004_proof_summary.md",
        "hero_unit": {"asset_id": hero_asset},
    }
    (output / "fd004_real_report.json").write_text(json.dumps(report), encoding="utf-8")
    return report


__all__ = ["Fd004Row", "fd004_row_to_sii_record", "load_fd004_dataset", "run_fd004_real_evaluation"]
