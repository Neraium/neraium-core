"""
FD00x structural early-warning evaluation framework — experiment runner.

Entry point
-----------
Run from the repository root::

    python -m fd00x.experiment --mode evaluate --dataset FD004
    python -m fd00x.experiment --mode tune     --dataset FD004
    python -m fd00x.experiment --mode compare_baselines --dataset FD004
    python -m fd00x.experiment --mode all      --dataset FD001 FD002 FD003 FD004
    python -m fd00x.experiment --preset conservative --dataset FD004
    python -m fd00x.experiment --config outputs/my_run/config.json

Runtime modes
-------------
evaluate            Run the structural detector, save per-unit results and
                    aggregate metrics.  DEFAULT.
tune                Sweep the parameter grid defined in config and save a
                    ranked tuning_results.csv.  NOT the default.
plot                Generate diagnostic plots for the best/worst/FP units.
                    Requires a prior evaluate run in the same output directory.
compare_baselines   Run structural detector + three simple baselines on the
                    same data and save a comparison table.
all                 Run evaluate + tune + compare_baselines + plot.

Output layout
-------------
Every run writes to::

    outputs/<run_name>/
        config.json                   — effective config for reproducibility
        summary.json                  — aggregate metrics
        results.csv                   — per-unit results
        plots/                        — diagnostic plots
        tuning_results.csv            — tuning grid (tune / all modes)
        baseline_comparison.csv       — baseline comparison (compare / all)
        cross_dataset_summary.csv     — multi-dataset run (when > 1 dataset)

Output directories are created automatically.
Run names default to a UTC timestamp so consecutive runs never overwrite each other.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .config import ALL_DATASETS, DEFAULT_PRESET, PRESETS, DetectorConfig
from .evaluation import (
    AggregateMetrics,
    UnitResult,
    compare_baselines,
    compute_aggregate_metrics,
    evaluate_all_datasets,
    evaluate_detector,
    load_cmapss_dataset,
    run_tuning,
    save_comparison_table,
    save_cross_dataset_summary,
    save_summary_json,
    save_tuning_results,
    save_unit_results,
    select_informative_sensors,
)


# ─────────────────────────────────────────────────────────────────────────────
# Output directory helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_run_dir(config: DetectorConfig) -> str:
    """Create and return the output directory for this run."""
    run_name = config.run_name or datetime.now(tz=timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = os.path.join(config.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# Mode implementations
# ─────────────────────────────────────────────────────────────────────────────


def run_evaluate(
    config: DetectorConfig,
    unit_data: Dict[int, object],
    sensor_cols: List[int],
    run_dir: str,
) -> Tuple[List[UnitResult], AggregateMetrics]:
    """
    Run the structural detector on all units and save results.

    Walk-forward safety:
        Reference statistics are derived only from the first
        ``healthy_fraction × n_cycles`` of each unit (frozen before scoring).
        See StructuralDriftDetector.process_unit for implementation details.
    """
    print(f"\n[evaluate] Running structural drift detector …")
    results = evaluate_detector(config, unit_data, sensor_cols)
    metrics = compute_aggregate_metrics(results, config)

    # Save outputs
    save_unit_results(results, os.path.join(run_dir, "results.csv"))
    save_summary_json(metrics, os.path.join(run_dir, "summary.json"))

    _print_metrics("Structural detector", metrics)
    return results, metrics


def run_tune_mode(
    config: DetectorConfig,
    unit_data: Dict[int, object],
    sensor_cols: List[int],
    run_dir: str,
) -> None:
    """
    Sweep the parameter grid and save a ranked tuning_results.csv.

    Tuning is run on the CORRECTED detector (no backdating).
    The default mode is NOT tune — set mode='tune' explicitly.

    Grid swept (defined in config):
        threshold_std  × persistence  × healthy_fraction
    """
    print(f"\n[tune] Sweeping parameter grid …")
    grid_size = (
        len(config.tune_threshold_values)
        * len(config.tune_persistence_values)
        * len(config.tune_healthy_fraction_values)
    )
    print(f"  Grid: {grid_size} configurations")

    all_rows, best = run_tuning(config, unit_data, sensor_cols)

    outpath = os.path.join(run_dir, "tuning_results.csv")
    save_tuning_results(all_rows, outpath)
    print(f"  Tuning results saved → {outpath}")

    if best:
        print("\n  ── Best configuration ──────────────────────────────────")
        print(f"  threshold_std    : {best.threshold_std}")
        print(f"  persistence      : {best.persistence}")
        print(f"  healthy_fraction : {best.healthy_fraction}")
        print(f"  coverage         : {best.coverage:.3f}")
        print(f"  FPR              : {best.false_positive_rate:.3f}")
        print(f"  mean_lead        : {best.mean_lead:.1f} cycles")
        print(f"  score            : {best.score:.2f}")
        print(f"  ────────────────────────────────────────────────────────")

        # Save best config
        best_cfg = config.with_overrides(
            threshold_std=best.threshold_std,
            persistence=best.persistence,
            healthy_fraction=best.healthy_fraction,
        )
        best_cfg.save(os.path.join(run_dir, "best_config.json"))
        print(f"  Best config saved → {run_dir}/best_config.json")


def run_compare_baselines_mode(
    config: DetectorConfig,
    unit_data: Dict[int, object],
    sensor_cols: List[int],
    run_dir: str,
) -> None:
    """
    Run the structural detector and three simple baselines; save comparison.

    Baselines:
        raw_threshold : Per-channel threshold (mean + k*std from healthy window)
        ma_zscore     : Moving-average z-score (no inter-sensor structure)
        cusum         : Cumulative-sum drift accumulation

    All methods use the same persistence / threshold_std / EMA alpha from
    config and are evaluated with identical FP / coverage / lead definitions.
    """
    print(f"\n[compare_baselines] Running all methods …")
    comparison = compare_baselines(config, unit_data, sensor_cols)

    outpath = os.path.join(run_dir, "baseline_comparison.csv")
    save_comparison_table(comparison, outpath)
    print(f"  Comparison table saved → {outpath}")

    print(f"\n  {'Method':<20} {'Coverage':>8} {'FPR':>8} {'Lead':>8} {'Score':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for method, m in comparison.items():
        print(
            f"  {method:<20} {m.coverage:>8.3f} {m.false_positive_rate:>8.3f}"
            f" {m.mean_lead:>8.1f} {m.score:>8.2f}"
        )


def run_plot_mode(
    config: DetectorConfig,
    unit_data: Dict[int, object],
    sensor_cols: List[int],
    results: Optional[List[UnitResult]],
    run_dir: str,
) -> None:
    """
    Generate diagnostic plots.  If results is None, re-runs the detector.
    """
    print(f"\n[plot] Generating diagnostic plots …")

    if results is None:
        results = evaluate_detector(config, unit_data, sensor_cols)

    try:
        from .plotting import (
            generate_unit_plots,
            plot_lead_time_histogram,
        )

        generate_unit_plots(
            config, unit_data, sensor_cols, results,
            output_dir=run_dir,
            max_units=5,
            select="best_worst_fp",
        )
        plot_lead_time_histogram(
            results,
            save_path=os.path.join(run_dir, "plots", "lead_time_histogram.png"),
            title=f"Lead-time distribution — {config.dataset}",
        )
        print(f"  Plots saved → {run_dir}/plots/")
    except ImportError as exc:
        print(f"  [warning] Plotting skipped: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-dataset runner
# ─────────────────────────────────────────────────────────────────────────────


def run_multi_dataset(
    base_config: DetectorConfig,
    datasets: List[str],
    run_dir: str,
) -> None:
    """Run evaluate across all specified datasets and save a summary table."""
    print(f"\n[all datasets] Evaluating: {datasets}")
    summary = evaluate_all_datasets(base_config, tuple(datasets))

    outpath = os.path.join(run_dir, "cross_dataset_summary.csv")
    save_cross_dataset_summary(summary, outpath)
    print(f"  Cross-dataset summary saved → {outpath}")

    if summary:
        try:
            from .plotting import plot_cross_dataset_summary
            plot_cross_dataset_summary(
                summary,
                save_path=os.path.join(run_dir, "plots", "cross_dataset_summary.png"),
            )
        except ImportError:
            pass

    print(f"\n  {'Dataset':<10} {'Coverage':>8} {'FPR':>8} {'Lead':>8} {'Score':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for ds, m in summary.items():
        print(
            f"  {ds:<10} {m.coverage:>8.3f} {m.false_positive_rate:>8.3f}"
            f" {m.mean_lead:>8.1f} {m.score:>8.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _print_metrics(label: str, m: AggregateMetrics) -> None:
    print(f"\n  ── {label} results ──────────────────────────────────────")
    print(f"  Units evaluated  : {m.n_units}")
    print(f"  Coverage         : {m.coverage:.3f}")
    print(f"  False-positive rate: {m.false_positive_rate:.3f}")
    print(f"  Mean lead        : {m.mean_lead:.1f} cycles")
    print(f"  Median lead      : {m.median_lead:.1f} cycles")
    print(f"  Score            : {m.score:.2f}")
    print(f"  ────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    """
    Parse CLI arguments and dispatch to the appropriate mode.

    Returns 0 on success, non-zero on error.
    """
    parser = argparse.ArgumentParser(
        prog="python -m fd00x.experiment",
        description=(
            "FD00x structural early-warning evaluation framework.\n\n"
            "Evaluates a multivariate structural drift detector on CMAPSS "
            "turbofan datasets (FD001–FD004).\n\n"
            "Modes: evaluate | tune | plot | compare_baselines | all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["evaluate", "tune", "plot", "compare_baselines", "all"],
        default="evaluate",
        help="Runtime mode (default: evaluate)",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["FD004"],
        metavar="DS",
        help="Dataset(s) to evaluate: FD001 FD002 FD003 FD004 (default: FD004)",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="Named parameter preset: conservative | balanced | aggressive",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Load config from a JSON file (overrides --preset)",
    )
    parser.add_argument(
        "--train-path",
        default=None,
        metavar="PATH",
        help="Explicit path to train_FD00x.txt",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        metavar="DIR",
        help="Root directory for outputs (default: outputs/)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help="Human-readable run label (default: timestamp)",
    )
    parser.add_argument(
        "--threshold-std",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Override threshold_std",
    )
    parser.add_argument(
        "--persistence",
        type=int,
        default=None,
        metavar="INT",
        help="Override persistence",
    )

    args = parser.parse_args(argv)

    # ── Build config ──────────────────────────────────────────────────────────
    if args.config:
        config = DetectorConfig.load(args.config)
        print(f"Config loaded from: {args.config}")
    elif args.preset:
        config = PRESETS[args.preset]
        print(f"Using preset: {args.preset}")
    else:
        config = PRESETS[DEFAULT_PRESET]
        print(f"Using default preset: {DEFAULT_PRESET}")

    # CLI overrides
    overrides = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.train_path:
        overrides["train_path"] = args.train_path
    if args.output_root:
        overrides["output_root"] = args.output_root
    if args.run_name:
        overrides["run_name"] = args.run_name
    if args.threshold_std is not None:
        overrides["threshold_std"] = args.threshold_std
    if args.persistence is not None:
        overrides["persistence"] = args.persistence
    if args.dataset:
        overrides["dataset"] = args.dataset[0]  # primary dataset

    if overrides:
        config = config.with_overrides(**overrides)

    datasets = [d.upper() for d in args.dataset]
    multi_dataset = len(datasets) > 1

    # ── Set up run directory ──────────────────────────────────────────────────
    run_dir = _make_run_dir(config)
    print(f"\nRun directory: {run_dir}")

    # Save effective config for reproducibility
    config.save(os.path.join(run_dir, "config.json"))
    print(f"Config saved → {run_dir}/config.json")

    # ── Dispatch to mode ──────────────────────────────────────────────────────
    mode = config.mode

    # If multiple datasets requested, route through multi-dataset runner
    if multi_dataset and mode in ("evaluate", "all"):
        run_multi_dataset(config, datasets, run_dir)
        if mode == "all":
            # Also run the full pipeline for the primary dataset
            _run_single_dataset_pipeline(config, run_dir, mode="all")
        return 0

    _run_single_dataset_pipeline(config, run_dir, mode=mode)
    return 0


def _run_single_dataset_pipeline(
    config: DetectorConfig,
    run_dir: str,
    mode: str,
) -> None:
    """Load data and run the requested pipeline for a single dataset."""
    print(f"\nDataset: {config.dataset}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading dataset …")
    try:
        unit_data = load_cmapss_dataset(config.dataset, config.train_path)
    except FileNotFoundError as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        sys.exit(1)

    sensor_cols = select_informative_sensors(unit_data, config)
    print(
        f"  {len(unit_data)} units loaded, "
        f"{len(sensor_cols)} informative sensor columns selected"
    )

    results: Optional[List[UnitResult]] = None
    t0 = time.time()

    if mode in ("evaluate", "all"):
        results, _ = run_evaluate(config, unit_data, sensor_cols, run_dir)

    if mode in ("tune", "all"):
        run_tune_mode(config, unit_data, sensor_cols, run_dir)

    if mode in ("compare_baselines", "all"):
        run_compare_baselines_mode(config, unit_data, sensor_cols, run_dir)
        try:
            from .plotting import plot_baseline_comparison
            comp = compare_baselines(config, unit_data, sensor_cols)
            plot_baseline_comparison(
                comp,
                save_path=os.path.join(run_dir, "plots", "baseline_comparison.png"),
            )
        except (ImportError, Exception):
            pass

    if mode in ("plot", "all"):
        run_plot_mode(config, unit_data, sensor_cols, results, run_dir)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.  Outputs in: {run_dir}/")


if __name__ == "__main__":
    sys.exit(main())
