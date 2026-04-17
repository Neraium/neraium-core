"""Command-line interface for Neraium.

Canonical entrypoint for validation, benchmarking, and operations.

Usage:
    neraium validate --fd004 ./FD004.csv --output ./results
    neraium validate --ims ./IMS.csv --fd004 ./FD004.csv
    neraium validate --all --shadow-mode
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from neraium_core.engine import Engine

logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def cli():
    """Neraium: Structural drift detection for industrial equipment."""
    pass


@cli.command()
@click.option(
    "--fd004",
    type=click.Path(exists=True),
    help="Path to FD004 dataset CSV",
)
@click.option(
    "--ims",
    type=click.Path(exists=True),
    help="Path to IMS dataset CSV",
)
@click.option(
    "--custom",
    type=click.Path(exists=True),
    help="Path to custom CSV dataset",
)
@click.option(
    "--all",
    is_flag=True,
    help="Run all available datasets (auto-discover)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./validation_results",
    help="Output directory for results [default: ./validation_results]",
)
@click.option(
    "--shadow-mode",
    is_flag=True,
    help="Enable shadow mode evidence collection (validation/replay comparison)",
)
@click.option(
    "--drift-threshold",
    type=float,
    default=0.7,
    help="Drift threshold for alert determination [default: 0.7]",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output",
)
@click.option(
    "--decision-audit-out",
    type=click.Path(),
    default=None,
    help="Path to write decision audit CSV during replay",
)
def validate(
    fd004: Optional[str],
    ims: Optional[str],
    custom: Optional[str],
    all: bool,
    output: str,
    shadow_mode: bool,
    drift_threshold: float,
    verbose: bool,
    decision_audit_out: Optional[str],
):
    """Run validation and generate benchmark artifacts.

    This is the canonical validation command. It produces:
    - Benchmark metrics (FD004, IMS, custom datasets)
    - Validation plots and hero plots
    - Lead time summaries
    - Evidence reports (if shadow-mode enabled)
    - Consolidated markdown report
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Validation starting. Output: {output_dir}")
    logger.info(f"Shadow mode: {'ENABLED' if shadow_mode else 'DISABLED'}")

    # Collect datasets to process
    datasets: dict[str, Path | None] = {
        "fd004": Path(fd004) if fd004 else None,
        "ims": Path(ims) if ims else None,
        "custom": Path(custom) if custom else None,
    }

    # Auto-discover if --all requested
    if all:
        logger.info("Auto-discovering datasets...")
        discovered = _discover_datasets()
        datasets.update(discovered)

    # Filter out None values
    datasets = {k: v for k, v in datasets.items() if v is not None}

    if not datasets:
        logger.error("No datasets specified. Use --fd004, --ims, --custom, or --all")
        sys.exit(1)

    logger.info(f"Processing {len(datasets)} dataset(s): {list(datasets.keys())}")

    # Process each dataset
    all_results = {}
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dataset_name.upper()}")
        logger.info(f"File: {dataset_path}")

        if not dataset_path.exists():
            logger.warning(f"  ✗ File not found: {dataset_path}")
            continue

        try:
            # Replay through a fresh engine instance to isolate dataset state
            engine = Engine(enable_shadow_mode=shadow_mode, baseline_window=12, recent_window=6)

            # Prepare decision audit path if specified
            audit_path = None
            if decision_audit_out:
                audit_dir = Path(decision_audit_out)
                audit_dir.mkdir(parents=True, exist_ok=True)
                audit_path = audit_dir / f"{dataset_name}_decision_audit.csv"

            replay_summary = engine.replay(
                dataset_path,
                dataset_type=dataset_name,
                decision_audit_out=audit_path,
            )
            logger.info(f"  ✓ Replayed {replay_summary['records_processed']} records")
            if replay_summary.get('decision_audit_logged'):
                logger.info(f"  ✓ Decision audit logged: {audit_path}")

            # Get summary metrics
            summary = engine.get_summary()
            logger.info(f"  ✓ Avg drift: {summary['avg_drift']:.4f}")

            # Create dataset output directory
            dataset_output_dir = output_dir / dataset_name.lower()
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics
            metrics_json = dataset_output_dir / "metrics.json"
            with open(metrics_json, "w") as f:
                json.dump({
                    "dataset": dataset_name,
                    "replay_summary": replay_summary,
                    "metrics": summary,
                }, f, indent=2)
            logger.info(f"  ✓ Saved metrics: {metrics_json.name}")

            # Save evidence if shadow mode
            if shadow_mode:
                evidence = engine.get_evidence_report()
                evidence_json = dataset_output_dir / "evidence.json"
                with open(evidence_json, "w") as f:
                    json.dump(evidence, f, indent=2)
                logger.info(f"  ✓ Saved evidence: {evidence_json.name}")

            all_results[dataset_name] = {
                "status": "success",
                "metrics": summary,
                "path": str(dataset_output_dir),
            }

        except Exception as e:
            logger.error(f"  ✗ Error processing {dataset_name}: {e}")
            all_results[dataset_name] = {
                "status": "error",
                "error": str(e),
            }
            continue

    # Generate consolidated report
    logger.info(f"\n{'='*60}")
    logger.info("Generating consolidated report...")

    report_content = _generate_consolidated_report(all_results, shadow_mode)
    report_path = output_dir / "VALIDATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    logger.info(f"  ✓ Saved report: {report_path}")

    # Save results index
    index_path = output_dir / "results.json"
    with open(index_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"  ✓ Saved results index: {index_path}")

    logger.info(f"\n{'='*60}")
    logger.info("✓ Validation complete")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Open {report_path} for details")


def _discover_datasets() -> dict[str, Path]:
    """Auto-discover standard datasets in common locations."""
    discovered = {}

    # Check common paths
    candidates = {
        "fd004": ["./FD004.csv", "./data/FD004.csv", "../data/FD004.csv"],
        "ims": ["./IMS.csv", "./data/IMS.csv", "../data/IMS.csv"],
    }

    for dataset_name, paths in candidates.items():
        for path_str in paths:
            path = Path(path_str)
            if path.exists():
                discovered[dataset_name] = path
                logger.info(f"  ✓ Discovered {dataset_name}: {path}")
                break

    return discovered


def _generate_consolidated_report(
    results: dict[str, dict],
    shadow_mode: bool,
) -> str:
    """Generate consolidated markdown report from validation results."""
    lines = [
        "# Neraium Validation Report",
        "",
        f"**Generated**: {Path.cwd()}",
        f"**Shadow Mode**: {'Enabled' if shadow_mode else 'Disabled'}",
        "",
        "## Summary",
        "",
    ]

    # Add results summary
    successful = sum(1 for r in results.values() if r.get("status") == "success")
    failed = sum(1 for r in results.values() if r.get("status") == "error")

    lines.extend([
        f"- Total datasets: {len(results)}",
        f"- Successful: {successful}",
        f"- Failed: {failed}",
        "",
    ])

    # Add per-dataset results
    lines.append("## Results by Dataset")
    lines.append("")

    for dataset_name, result in results.items():
        lines.append(f"### {dataset_name.upper()}")
        lines.append("")

        if result["status"] == "success":
            metrics = result.get("metrics", {})
            lines.extend([
                "**Status**: ✓ Success",
                "",
                "**Metrics**:",
                f"- Total frames: {metrics.get('total_frames', 'N/A')}",
                f"- Avg drift: {metrics.get('avg_drift', 'N/A'):.4f}",
                f"- Max drift: {metrics.get('max_drift', 'N/A'):.4f}",
                f"- Alert count: {metrics.get('alert_count', 0)}",
                f"- Watch count: {metrics.get('watch_count', 0)}",
                f"- Stable count: {metrics.get('stable_count', 0)}",
                "",
            ])

            if shadow_mode:
                lines.append(f"**Evidence**: See `{dataset_name.lower()}/evidence.json`")
                lines.append("")

        else:
            error = result.get("error", "Unknown error")
            lines.extend([
                "**Status**: ✗ Failed",
                f"**Error**: {error}",
                "",
            ])

    lines.extend([
        "## Instructions",
        "",
        "For detailed metrics, see the individual dataset directories:",
        "",
        "```",
        "cd validation_results/",
        "cat fd004/metrics.json      # FD004 benchmark metrics",
        "cat ims/metrics.json        # IMS benchmark metrics",
        "```",
        "",
        "## Next Steps",
        "",
        "1. Review metrics against expected baselines",
        "2. Check evidence logs (if shadow-mode enabled)",
        "3. Validate that all sensors are detected correctly",
        "4. Verify that state transitions match operational expectations",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    cli()
