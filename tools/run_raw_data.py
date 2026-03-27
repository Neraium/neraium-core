from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neraium_core.service import StructuralMonitoringService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neraium on raw industrial data (tabular or directory blocks).")
    parser.add_argument("--input", required=True, help="Raw input path (file or directory).")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--site-id", default="raw-telemetry", help="Site id used when source does not provide one.")
    parser.add_argument("--customer-id", default="default-customer", help="Customer id for runtime persistence.")
    parser.add_argument(
        "--validation-sample-size",
        type=int,
        default=None,
        help="Small-sample validation mode: only ingest first N timesteps.",
    )
    args = parser.parse_args()

    service = StructuralMonitoringService()
    out = service.ingest_raw_industrial_data(
        args.input,
        customer_id=args.customer_id,
        site_id=args.site_id,
        sample_size=args.validation_sample_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    diag = out["diagnostics"]
    print(
        "detected_input_type={detected_input_type} timestep_count={timestep_count} "
        "sensor_count={sensor_count} preprocessing_mode={preprocessing_mode} "
        "has_valid_signals={has_valid_signals}".format(**diag)
    )
    print(f"processed_results={len(out['results'])} output={output_path}")


if __name__ == "__main__":
    main()
