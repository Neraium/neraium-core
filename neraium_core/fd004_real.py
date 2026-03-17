from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from neraium_core.fd004_synthetic import Fd004RiskEscalator
from neraium_core.service import StructuralMonitoringService

EXPECTED_FD004_COLUMNS = 26
FD004_SENSOR_COUNT = 21
TARGET_SENSOR_COUNT = 24
RISK_PHASE_MAP = {"LOW": "stable", "MEDIUM": "drift", "HIGH": "unstable"}
RUL_THRESHOLDS = (100, 50, 30)


@dataclass(frozen=True)
class Fd004Row:
    unit: int
    time: int
    operating_settings: tuple[float, float, float]
    sensors: tuple[float, ...]


def parse_fd004_line(line: str) -> Fd004Row:
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty line")
    if len(parts) < EXPECTED_FD004_COLUMNS:
        raise ValueError(
            f"FD004 line must contain at least {EXPECTED_FD004_COLUMNS} columns, got {len(parts)}"
        )

    values = [float(x) for x in parts[:EXPECTED_FD004_COLUMNS]]
    unit = int(values[0])
    time = int(values[1])
    operating_settings = tuple(values[2:5])
    sensors = tuple(values[5 : 5 + FD004_SENSOR_COUNT])
    return Fd004Row(unit=unit, time=time, operating_settings=operating_settings, sensors=sensors)


def load_fd004_dataset(path: str | Path) -> list[Fd004Row]:
    rows: list[Fd004Row] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if raw_line.strip() == "":
            continue
        rows.append(parse_fd004_line(raw_line))
    return rows


def fd004_row_to_sii_record(
    row: Fd004Row,
    *,
    site_id: str = "fd004-real",
    start_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Convert FD004 row into the same SII-shaped payload used by existing ingestion.

    NASA FD004 has 21 sensor channels. The current synthetic workflow expects 24
    channels (`s01`..`s24`), so we copy the 21 measured values to `s01`..`s21`
    and pad `s22`..`s24` with zeros.
    """
    base_time = start_time or datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamp = base_time + timedelta(minutes=row.time)

    sensor_values = {f"s{index:02d}": float(value) for index, value in enumerate(row.sensors, start=1)}
    for pad_index in range(FD004_SENSOR_COUNT + 1, TARGET_SENSOR_COUNT + 1):
        sensor_values[f"s{pad_index:02d}"] = 0.0

    return {
        "timestamp": timestamp.isoformat(),
        "site_id": site_id,
        "asset_id": f"unit_{row.unit:03d}",
        "sensor_values": sensor_values,
        "_meta": {
            "cycle": row.time,
            "operating_settings": row.operating_settings,
            "padding": "s22-s24 set to 0.0 (FD004 provides 21 sensors)",
        },
    }


def load_fd004_rul(path: str | Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    for idx, line in enumerate(lines, start=1):
        mapping[f"unit_{idx:03d}"] = int(float(line.split()[0]))
    return mapping


def run_fd004_real_evaluation(
    train_path: str = "data/fd004/train_FD004.txt",
    test_path: str = "data/fd004/test_FD004.txt",
    rul_path: str = "data/fd004/RUL_FD004.txt",
    output_dir: str = "fd004_outputs",
    site_id: str = "fd004-real",
) -> dict[str, Any]:
    train_rows = load_fd004_dataset(train_path)
    records = [fd004_row_to_sii_record(row, site_id=site_id) for row in train_rows]

    by_unit: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_unit.setdefault(record["asset_id"], []).append(record)

    all_rows: list[dict[str, Any]] = []
    unit_summaries: list[dict[str, Any]] = []
    rul_mapping: dict[str, int] = {}
    if Path(rul_path).exists():
        rul_mapping = load_fd004_rul(rul_path)

    for asset_id, unit_records in by_unit.items():
        service = StructuralMonitoringService()
        escalator = Fd004RiskEscalator()
        transitions = {"MEDIUM": None, "HIGH": None}
        instabilities: list[float] = []
        drift_scores: list[float] = []
        per_cycle_instability: list[tuple[int, float]] = []
        per_cycle_drift: list[tuple[int, float]] = []
        max_cycle = max(record["_meta"]["cycle"] for record in unit_records)
        baseline_rul = rul_mapping.get(asset_id, 0)
        threshold_increase_windows: dict[int, bool] = {}

        for record in sorted(unit_records, key=lambda r: r["_meta"]["cycle"]):
            result = service.ingest_payload(
                {
                    "timestamp": record["timestamp"],
                    "site_id": record["site_id"],
                    "asset_id": record["asset_id"],
                    "sensor_values": record["sensor_values"],
                }
            )
            cycle = int(record["_meta"]["cycle"])
            instability = float(result.get("experimental_analytics", {}).get("composite_instability", 0.0))
            drift_score = float(result.get("structural_drift_score", 0.0))
            risk_level, smoothed = escalator.update(instability, regime_index=0)
            phase = RISK_PHASE_MAP.get(risk_level, "stable")
            cycle_rul = max_cycle - cycle + baseline_rul

            row = {
                "asset_id": asset_id,
                "cycle": cycle,
                "structural_drift_score": drift_score,
                "composite_instability": instability,
                "trend": phase,
                "phase": phase,
                "risk_level": risk_level,
                "estimated_rul": cycle_rul,
                "operator_message": result.get("operator_message", ""),
                "structural_analysis_available": bool(result.get("structural_analysis_available", False)),
            }
            all_rows.append(row)
            instabilities.append(smoothed)
            drift_scores.append(drift_score)
            per_cycle_instability.append((cycle_rul, smoothed))
            per_cycle_drift.append((cycle_rul, drift_score))

            if risk_level in transitions and transitions[risk_level] is None:
                transitions[risk_level] = cycle

        for threshold in RUL_THRESHOLDS:
            below_index = next(
                (index for index, (rul_value, _) in enumerate(per_cycle_instability) if rul_value < threshold),
                None,
            )
            if below_index is None or below_index < 2:
                threshold_increase_windows[threshold] = False
                continue

            window_start = max(0, below_index - 10)
            window = [value for _, value in per_cycle_instability[window_start:below_index]]
            threshold_increase_windows[threshold] = window[-1] > window[0] if len(window) >= 2 else False

        early_warning_window = None
        if transitions["MEDIUM"] is not None and transitions["HIGH"] is not None:
            early_warning_window = transitions["HIGH"] - transitions["MEDIUM"]

        instability_rul_correlation = _pearson_correlation(
            [float(rul_value) for rul_value, _ in per_cycle_instability],
            [value for _, value in per_cycle_instability],
        )
        drift_rul_correlation = _pearson_correlation(
            [float(rul_value) for rul_value, _ in per_cycle_drift],
            [value for _, value in per_cycle_drift],
        )

        unit_summaries.append(
            {
                "asset_id": asset_id,
                "first_MEDIUM_step": transitions["MEDIUM"],
                "first_HIGH_step": transitions["HIGH"],
                "early_warning_window": early_warning_window,
                "peak_instability": round(max(instabilities) if instabilities else 0.0, 4),
                "average_instability": round(mean(instabilities) if instabilities else 0.0, 4),
                "average_drift": round(mean(drift_scores) if drift_scores else 0.0, 4),
                "instability_vs_rul_correlation": instability_rul_correlation,
                "drift_vs_rul_correlation": drift_rul_correlation,
                "instability_increases_before_rul_thresholds": {
                    str(threshold): threshold_increase_windows[threshold] for threshold in RUL_THRESHOLDS
                },
            }
        )

    valid_windows = [
        summary["early_warning_window"]
        for summary in unit_summaries
        if summary["early_warning_window"] is not None
    ]

    overall_summary = {
        "units_total": len(by_unit),
        "rows_processed": len(records),
        "train_path": train_path,
        "test_path": test_path,
        "rul_path": rul_path,
        "average_early_warning_window": round(mean(valid_windows), 2) if valid_windows else None,
    }

    output = {
        "config": {
            "source": "nasa-fd004-real",
            "num_sensors_input": FD004_SENSOR_COUNT,
            "num_sensors_output": TARGET_SENSOR_COUNT,
            "padding_strategy": "s22-s24 padded with 0.0",
            "site_id": site_id,
        },
        "unit_summaries": sorted(unit_summaries, key=lambda x: x["asset_id"]),
        "overall_summary": overall_summary,
        "timeseries": all_rows,
        "rul_by_unit": rul_mapping,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fd004_real_report.json"
    csv_path = out_dir / "fd004_real_timeseries.csv"
    rul_json_path = out_dir / "fd004_real_rul_map.json"

    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(all_rows[0].keys()) if all_rows else ["asset_id"])
        writer.writeheader()
        if all_rows:
            writer.writerows(all_rows)

    if rul_mapping:
        rul_json_path.write_text(json.dumps(rul_mapping, indent=2), encoding="utf-8")

    print(f"Saved report JSON: {json_path}")
    print(f"Saved timeseries CSV: {csv_path}")
    if rul_mapping:
        print(f"Saved RUL map JSON: {rul_json_path}")

    return output


def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(xs, ys, strict=True))
    denominator_x = sum((x_value - x_mean) ** 2 for x_value in xs)
    denominator_y = sum((y_value - y_mean) ** 2 for y_value in ys)
    denominator = (denominator_x * denominator_y) ** 0.5
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NASA FD004 real-data workflow")
    parser.add_argument("--train-path", default="data/fd004/train_FD004.txt")
    parser.add_argument("--test-path", default="data/fd004/test_FD004.txt")
    parser.add_argument("--rul-path", default="data/fd004/RUL_FD004.txt")
    parser.add_argument("--output-dir", default="fd004_outputs")
    parser.add_argument("--site-id", default="fd004-real")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_fd004_real_evaluation(
        train_path=args.train_path,
        test_path=args.test_path,
        rul_path=args.rul_path,
        output_dir=args.output_dir,
        site_id=args.site_id,
    )


if __name__ == "__main__":
    main()
