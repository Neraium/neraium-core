#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from neraium_core.fd001_validation import (
    group_rows_by_unit,
    load_fd001_dataset,
    replay_fd001_units,
    write_jsonl,
    write_summary_csv,
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
    parser.add_argument("--max-units", type=int, default=None, help="Optional cap on number of units to replay.")
    parser.add_argument("--max-cycles", type=int, default=None, help="Optional cap on cycles per unit.")
    parser.add_argument("--site-id", type=str, default="cmapss-fd001", help="site_id field used in generated payloads.")
    return parser.parse_args()


def _parse_units(value: str) -> list[int] | None:
    if not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    rows = load_fd001_dataset(args.input)
    grouped = group_rows_by_unit(rows)
    unit_ids = _parse_units(args.units)

    if unit_ids is None:
        unit_ids = sorted(grouped)
    if args.max_units is not None and args.max_units > 0:
        unit_ids = unit_ids[: int(args.max_units)]

    print(f"loaded_rows={len(rows)} units_available={len(grouped)} units_selected={len(unit_ids)}")

    full, summary = replay_fd001_units(
        grouped,
        unit_ids=unit_ids,
        max_cycles=args.max_cycles,
        site_id=args.site_id,
    )

    jsonl_path = args.output_dir / "fd001_validation_full.jsonl"
    csv_path = args.output_dir / "fd001_validation_summary.csv"
    write_jsonl(jsonl_path, full)
    write_summary_csv(csv_path, summary)

    print(f"processed_rows={len(full)}")
    print(f"full_jsonl={jsonl_path}")
    print(f"summary_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
