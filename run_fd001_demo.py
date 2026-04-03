#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from neraium_core.fd001_validation import (
    group_rows_by_unit,
    load_fd001_dataset,
    replay_fd001_units,
    write_jsonl,
    write_quick_report,
    write_summary_csv,
    write_unit_milestones_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal FD001 sequential validation replay on Neraium.")
    parser.add_argument("--input", type=Path, default=Path("test_FD001.txt"), help="Path to CMAPSS FD001 test file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fd001_validation"),
        help="Directory for JSONL and CSV outputs.",
    )
    parser.add_argument(
        "--units",
        type=str,
        default="",
        help="Comma-separated unit ids to replay (example: 1,2). Empty means all units.",
    )
    parser.add_argument("--max-units", type=int, default=2, help="Cap on number of units to replay (smoke-safe default: 2).")
    parser.add_argument("--max-cycles", type=int, default=120, help="Cap on cycles per unit (smoke-safe default: 120).")
    parser.add_argument("--site-id", type=str, default="cmapss-fd001", help="site_id field used in generated payloads.")
    return parser.parse_args()


def _parse_units(value: str) -> list[int] | None:
    if not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _print_confidence_comparison(summary: list[dict[str, object]], *, unit_id: int, start_cycle: int = 12, end_cycle: int = 40) -> None:
    rows = [
        row
        for row in summary
        if int(row.get("unit_id", -1) or -1) == int(unit_id)
        and start_cycle <= int(row.get("cycle", 0) or 0) <= end_cycle
    ]
    if not rows:
        return
    print(f"confidence_comparison_unit={unit_id} cycles={start_cycle}-{end_cycle}")
    print("cycle,decision_conf_raw,decision_conf_smoothed")
    for row in rows:
        cycle = int(row.get("cycle", 0) or 0)
        raw = float(row.get("decision_confidence_raw", 0.0) or 0.0)
        smoothed = float(row.get("decision_confidence", 0.0) or 0.0)
        print(f"{cycle},{raw:.4f},{smoothed:.4f}")


def main() -> int:
    args = parse_args()
    rows = load_fd001_dataset(args.input)
    grouped = group_rows_by_unit(rows)
    unit_ids = _parse_units(args.units)

    if unit_ids is None:
        unit_ids = sorted(grouped)
    if args.max_units is not None and args.max_units > 0:
        unit_ids = unit_ids[: int(args.max_units)]

    print(
        "loaded_rows={} units_available={} units_selected={} max_cycles={}".format(
            len(rows), len(grouped), len(unit_ids), args.max_cycles
        )
    )

    full, summary, milestones = replay_fd001_units(
        grouped,
        unit_ids=unit_ids,
        max_cycles=args.max_cycles,
        site_id=args.site_id,
    )

    jsonl_path = args.output_dir / "fd001_validation_full.jsonl"
    csv_path = args.output_dir / "fd001_validation_summary.csv"
    milestones_csv_path = args.output_dir / "fd001_unit_milestones.csv"
    quick_report_path = args.output_dir / "fd001_quick_report.md"
    write_jsonl(jsonl_path, full)
    write_summary_csv(csv_path, summary)
    write_unit_milestones_csv(milestones_csv_path, milestones)
    write_quick_report(quick_report_path, milestones=milestones, summary_rows=summary)

    print(f"processed_rows={len(full)}")
    print(f"full_jsonl={jsonl_path}")
    print(f"summary_csv={csv_path}")
    print(f"milestones_csv={milestones_csv_path}")
    print(f"quick_report={quick_report_path}")
    if unit_ids:
        _print_confidence_comparison(summary, unit_id=unit_ids[0], start_cycle=12, end_cycle=40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
