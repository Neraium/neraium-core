from __future__ import annotations

import argparse
import json
from pathlib import Path

from neraium_core.sii import SIIApplication, SIIConfig, configure_structured_logging, run_structural_pipeline
from neraium_core.sii.calibration import derive_calibration_from_validation, load_config, save_config
from neraium_core.sii.errors import SIIConfigurationError, SIIError
from neraium_core.sii.ingestion import load_frames_from_csv, load_frames_from_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Neraium SII CLI: read-only systemic infrastructure intelligence "
            "using multivariate geometry and graph structure."
        )
    )
    p.add_argument("--input", help="Input telemetry file (.json or .csv)")
    p.add_argument("--output", required=True, help="Output report file (.json or .csv)")
    p.add_argument("--baseline-window", type=int, default=50)
    p.add_argument("--recent-window", type=int, default=12)
    p.add_argument("--max-history", type=int, default=500)
    p.add_argument("--relation-threshold", type=float, default=0.6)
    p.add_argument("--regime-distance-threshold", type=float, default=2.0)
    p.add_argument("--watch-threshold", type=float, default=0.50)
    p.add_argument("--alert-threshold", type=float, default=0.74)
    p.add_argument("--regime-store-path", default="sii_regimes.json")
    p.add_argument("--live", action="store_true", help="Run live API ingestion path")
    p.add_argument(
        "--live-polls",
        type=int,
        default=1,
        help="Number of live polling fetches to execute",
    )
    p.add_argument("--allow-context-provider", action="store_true", default=False)
    p.add_argument("--config", help="Optional structural calibration config JSON path")
    p.add_argument("--save-config", help="Optional output path to persist derived structural calibration JSON")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _config_from_args(args: argparse.Namespace) -> SIIConfig:
    return SIIConfig(
        baseline_window=int(args.baseline_window),
        recent_window=int(args.recent_window),
        max_history=int(args.max_history),
        relation_threshold=float(args.relation_threshold),
        regime_distance_threshold=float(args.regime_distance_threshold),
        watch_threshold=float(args.watch_threshold),
        alert_threshold=float(args.alert_threshold),
        regime_store_path=str(args.regime_store_path),
        allow_context_provider=bool(args.allow_context_provider),
        log_level=str(args.log_level).upper(),
    )


def main() -> int:
    args = _parse_args()
    config = _config_from_args(args)
    logger = configure_structured_logging(config.log_level)
    app = SIIApplication.from_config(config)
    try:
        output_path = Path(args.output)
        ingest_errors: list[dict[str, object]] = []
        if bool(args.live):
            outputs, live_diagnostics = app.run_live_ingestion_poll(max_polls=int(args.live_polls))
            logger.info(
                "sii_live_cli_diagnostics",
                extra={"poll_count": len(live_diagnostics), "diagnostics": live_diagnostics},
            )
            frames_succeeded = len(outputs)
            frames_failed = 0
        else:
            if not args.input:
                raise SIIConfigurationError("--input is required unless --live is set")
            input_path = Path(args.input)
            calibration = load_config(args.config) if args.config else None
            if input_path.suffix.lower() == ".json":
                payloads, ingest_errors = load_frames_from_json(str(input_path), return_ingest_errors=True)
            elif input_path.suffix.lower() == ".csv":
                payloads, ingest_errors = load_frames_from_csv(str(input_path), return_ingest_errors=True)
            else:
                payloads = []
                ingest_errors = []
            outputs = app.run_payloads(payloads)
            frames_succeeded = len(payloads)
            frames_failed = len(ingest_errors)
            structural_output = run_structural_pipeline(payloads, config=calibration)
            if args.save_config:
                derived = derive_calibration_from_validation(structural_output.get("validation_results") or {})
                save_config(derived, args.save_config)
        app.write_output_file(output_path, outputs)
        app.engine.close()
        input_empty = (frames_succeeded + frames_failed) == 0
        all_failed = frames_failed > 0 and frames_succeeded == 0
        partial_success = frames_succeeded > 0 and frames_failed > 0
        total_error_count = len(ingest_errors)
        if len(ingest_errors) > 20:
            ingest_errors = [*ingest_errors[:20], {"code": "truncated", "message": "additional errors omitted"}]
        returned_error_count = len(ingest_errors)
        errors_truncated = total_error_count != returned_error_count
        exit_code = 0
        if frames_failed > 0 or all_failed or partial_success:
            exit_code = 1
        print(
            json.dumps(
                {
                    "frames_processed": frames_succeeded + frames_failed,
                    "frames_succeeded": frames_succeeded,
                    "frames_failed": frames_failed,
                    "results_emitted": len(outputs),
                    "input_empty": input_empty,
                    "all_failed": all_failed,
                    "partial_success": partial_success,
                    "ingest_errors": ingest_errors,
                    "errors_truncated": errors_truncated,
                    "total_error_count": total_error_count,
                    "returned_error_count": returned_error_count,
                    "output_path": str(output_path),
                }
            )
        )
        return exit_code
    except (SIIError, SIIConfigurationError) as exc:
        logger.error("sii_cli_failed", extra={"error": str(exc)})
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
