#!/usr/bin/env python
"""
Full validation pipeline for neraium-core datasets.

Processes local FD001, FD004, IMS, and igrow datasets to compute benchmark metrics,
generate representative plots, and produce a summary report.

Usage:
    python scripts/run_full_validation.py

    python scripts/run_full_validation.py --fd004-path ./FD004_results.csv --output-dir ./results

    python scripts/run_full_validation.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from scripts.validation.config import build_config, parse_args, discover_local_paths
from scripts.validation.loaders import load_dataset
from scripts.validation.metrics import compute_metrics, create_lead_time_summary
from scripts.validation.plots import generate_plots, generate_hero_plot
from scripts.validation.report import generate_markdown_report, write_report, generate_hero_summary, write_hero_summary


def print_discovery_status(paths: dict[str, Path | None]) -> None:
    """Print auto-discovered dataset paths."""
    print("\nDataset Discovery:")
    print("=" * 60)
    for dataset_name, path in paths.items():
        if path:
            print(f"  {dataset_name.upper():8} ✓ Found: {path}")
        else:
            print(f"  {dataset_name.upper():8} ✗ Not found")


def process_dataset(
    dataset_name: str,
    path: Path | None,
    output_dir: Path,
    drift_threshold: float,
) -> tuple[bool, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Process a single dataset.

    Returns:
        (success, df, metrics_dict, lead_time_summary)
    """
    if path is None:
        print(f"\n{dataset_name.upper()}: Skipped (path not provided)")
        return False, None, None, None

    if not path.exists():
        print(f"\n{dataset_name.upper()}: Skipped (file not found: {path})")
        return False, None, None, None

    print(f"\n{dataset_name.upper()}: Processing {path.name}")

    # Load dataset
    df = load_dataset(path, dataset_name)
    if df is None:
        print("  ✗ Failed to load dataset")
        return False, None, None, None

    print(f"  ✓ Loaded: {len(df)} records from {df['unit'].nunique()} units")

    try:
        # Compute metrics
        metrics = compute_metrics(df, dataset_name.upper(), drift_threshold)
        print("  ✓ Computed metrics")

        # Create output directory
        dataset_output_dir = output_dir / dataset_name.lower()
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Save stats
        stats_df = pd.DataFrame([metrics.to_dict()])
        stats_path = dataset_output_dir / "stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"  ✓ Saved stats: {stats_path.name}")

        # Create lead time summary
        lead_time_summary = create_lead_time_summary(df, metrics, drift_threshold)
        summary_path = dataset_output_dir / "lead_time_summary.csv"
        lead_time_summary.to_csv(summary_path, index=False)
        print(f"  ✓ Saved lead time summary: {summary_path.name}")

        # Generate plots
        plots_dir = dataset_output_dir / "plots"
        print("  Generating plots...")
        generate_plots(
            df,
            dataset_name.upper(),
            metrics.best_case_unit,
            metrics.median_case_unit,
            metrics.worst_case_unit,
            plots_dir,
            drift_threshold,
        )

        return True, df, metrics, lead_time_summary

    except Exception as e:
        print(f"  ✗ Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def main() -> int:
    """Main entry point."""
    args = parse_args()
    config = build_config(args)

    print("\n" + "=" * 60)
    print("NERAIUM LOCAL VALIDATION PIPELINE")
    print("=" * 60)

    # Discover paths
    discovered = discover_local_paths()
    print_discovery_status(discovered)

    # Show configuration
    print("\nConfiguration:")
    print(f"  Drift Threshold: {config.drift_threshold}")
    print(f"  Output Directory: {config.output_dir}")
    print(f"  Datasets to Process: {', '.join(config.datasets)}")

    if config.dry_run:
        print("\n[DRY RUN] No processing will occur.")
        return 0

    # Process datasets
    processed = {}
    all_metrics = []
    fd004_df = None  # Keep FD004 dataframe for hero plot

    for dataset_name in config.datasets:
        path = config.get_path(dataset_name)
        success, df, metrics, summary = process_dataset(
            dataset_name,
            path,
            config.output_dir,
            config.drift_threshold,
        )
        processed[dataset_name] = success
        if success and metrics:
            all_metrics.append(metrics)
            # Store FD004 data for hero plot
            if dataset_name.lower() == "fd004" and df is not None:
                fd004_df = df

    # Generate report
    print("\n" + "=" * 60)
    print("Generating validation report...")
    report_text = generate_markdown_report(all_metrics, processed, config.output_dir)
    report_path = write_report(report_text, config.output_dir)
    print(f"✓ Report written: {report_path}")

    # Generate hero summary
    print("Generating hero summary...")
    hero_summary_text = generate_hero_summary(all_metrics)
    hero_summary_path = write_hero_summary(hero_summary_text, config.output_dir)
    print(f"✓ Hero summary written: {hero_summary_path}")

    # Generate hero plot (FD004 median case)
    if fd004_df is not None:
        print("Generating hero plot...")
        fd004_metrics = next((m for m in all_metrics if m.dataset_name == "FD004"), None)
        if fd004_metrics:
            hero_plot_path = config.output_dir / "hero_plot.png"
            plot_result = generate_hero_plot(
                fd004_df,
                fd004_metrics.median_case_unit,
                hero_plot_path,
                config.drift_threshold,
            )
            if plot_result:
                print(f"✓ Hero plot written: {plot_result}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    success_count = sum(1 for v in processed.values() if v)
    print(f"\nProcessed: {success_count}/{len(config.datasets)} datasets")
    print(f"Output Directory: {config.output_dir.resolve()}")
    print("\nGenerated Files:")
    print(f"  - {report_path.name}")
    print(f"  - {hero_summary_path.name}")
    if fd004_df is not None and fd004_metrics:
        print("  - hero_plot.png")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
