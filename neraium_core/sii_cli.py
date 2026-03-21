from __future__ import annotations

import argparse
import json
from pathlib import Path

from neraium_core.sii import (
    SIIConfig,
    SIIEngine,
    load_frames_from_csv,
    load_frames_from_json,
    write_csv_report,
    write_json_report,
)
from neraium_core.sii.errors import SIIValidationError


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Neraium SII CLI: read-only systemic infrastructure intelligence "
            "using multivariate geometry and graph structure."
        )
    )
    p.add_argument("--input", required=True, help="Input telemetry file (.json or .csv)")
    p.add_argument("--output", required=True, help="Output report file (.json or .csv)")
    p.add_argument("--baseline-window", type=int, default=50)
    p.add_argument("--recent-window", type=int, default=12)
    p.add_argument("--relation-threshold", type=float, default=0.6)
    p.add_argument("--regime-distance-threshold", type=float, default=2.0)
    p.add_argument("--watch-threshold", type=float, default=1.5)
    p.add_argument("--alert-threshold", type=float, default=2.5)
    p.add_argument("--regime-store-path", default="sii_regime_library.json")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    cfg = SIIConfig(
        baseline_window=int(args.baseline_window),
        recent_window=int(args.recent_window),
        relation_threshold=float(args.relation_threshold),
        regime_distance_threshold=float(args.regime_distance_threshold),
        watch_threshold=float(args.watch_threshold),
        alert_threshold=float(args.alert_threshold),
        regime_store_path=str(args.regime_store_path),
    )
    engine = SIIEngine(cfg)

    in_suffix = input_path.suffix.lower()
    if in_suffix == ".json":
        frames = load_frames_from_json(str(input_path))
    elif in_suffix == ".csv":
        frames = load_frames_from_csv(str(input_path))
    else:
        raise SIIValidationError(f"Unsupported input format: {in_suffix!r}")

    outputs = [engine.process_payload(frame) for frame in frames]

    out_suffix = output_path.suffix.lower()
    if out_suffix == ".json":
        write_json_report(str(output_path), outputs)
    elif out_suffix == ".csv":
        write_csv_report(str(output_path), outputs)
    else:
        raise SIIValidationError(f"Unsupported output format: {out_suffix!r}")

    print(
        json.dumps(
            {
                "frames_processed": len(frames),
                "results_emitted": len(outputs),
                "output_path": str(output_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
