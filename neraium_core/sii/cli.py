from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from neraium_core.sii.orchestration import run_structural_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neraium_core.sii.cli",
        description="Run the SII structural pipeline on a local dataset file.",
    )
    parser.add_argument("--input", required=True, help="Input dataset path (.json primary, .csv optional).")
    parser.add_argument("--output", help="Optional output file path for structured JSON.")
    return parser


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in input file: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("JSON input must be a list of record objects")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"JSON record at index {index} must be an object")
        records.append(row)
    return records


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV input must include a header row")

        metadata_keys = {"timestamp", "site_id", "system_id", "asset_id", "node_id"}
        records: list[dict[str, Any]] = []
        for index, row in enumerate(reader):
            sensor_values: dict[str, float | None] = {}
            for key, raw_value in row.items():
                if key is None or key in metadata_keys:
                    continue
                text = (raw_value or "").strip()
                if text == "":
                    sensor_values[key] = None
                    continue
                try:
                    sensor_values[key] = float(text)
                except ValueError as exc:
                    raise ValueError(f"CSV value at row {index + 2}, column {key!r} must be numeric") from exc

            records.append(
                {
                    "timestamp": row.get("timestamp"),
                    "site_id": row.get("site_id"),
                    "system_id": row.get("system_id"),
                    "asset_id": row.get("asset_id"),
                    "node_id": row.get("node_id"),
                    "sensor_values": sensor_values,
                }
            )

    return records


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_records(path)
    if suffix == ".csv":
        return _load_csv_records(path)
    raise ValueError(f"Unsupported input file type: {path.suffix or '<none>'}. Use .json or .csv")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        records = _load_records(input_path)
        output = run_structural_pipeline(records)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    serialized = json.dumps(output, indent=2, sort_keys=True)
    print(serialized)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
