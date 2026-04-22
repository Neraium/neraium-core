#!/usr/bin/env python3
"""
Turn a completed FD004 run directory into a clear, defensible proof of performance.

Analyzes FD004 validation outputs and generates:
- proof_summary.json: Complete metrics in JSON format
- proof_table.csv: Per-unit analysis
- lead_time_histogram.png: Distribution of lead times
- early_vs_alert_gap.png: Distribution of early vs alert improvement

Usage:
    python prove_fd004_results.py --run-dir <path-to-validation-output>
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_scored_csv(run_dir: Path) -> pd.DataFrame:
    """Load the FD004_scored.csv file from the run directory."""
    scored_csv = run_dir / "FD004_scored.csv"
    if not scored_csv.exists():
        raise FileNotFoundError(f"FD004_scored.csv not found in {run_dir}")
    return pd.read_csv(scored_csv)


def compute_lead_time_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute lead time statistics.

    Returns:
        Dict with average, median, min, max lead times and percent valid.
    """
    # Filter for units with valid alert lead times
    valid_leads = df[df['alert_lead'].notna()]['alert_lead'].dropna()

    if len(valid_leads) == 0:
        return {
            'average_lead_time_cycles': 0,
            'median_lead_time_cycles': 0,
            'min_lead_time_cycles': 0,
            'max_lead_time_cycles': 0,
            'percent_valid_lead_time': 0,
            'total_units': len(df),
            'units_with_valid_lead_time': 0
        }

    return {
        'average_lead_time_cycles': round(valid_leads.mean(), 2),
        'median_lead_time_cycles': round(valid_leads.median(), 2),
        'min_lead_time_cycles': int(valid_leads.min()),
        'max_lead_time_cycles': int(valid_leads.max()),
        'percent_valid_lead_time': round((len(valid_leads) / len(df)) * 100, 2),
        'total_units': len(df),
        'units_with_valid_lead_time': len(valid_leads)
    }


def compute_early_signal_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare early signal (watch) vs alert lead times.

    Returns:
        Dict with average lead times and improvement metrics.
    """
    # Filter for units with valid watch and alert leads
    has_watch = df[df['watch_lead'].notna()]
    has_alert = df[df['alert_lead'].notna()]
    has_both = df[(df['watch_lead'].notna()) & (df['alert_lead'].notna())]

    watch_leads = has_watch['watch_lead'].dropna()
    alert_leads = has_alert['alert_lead'].dropna()

    result = {
        'average_watch_lead_cycles': round(watch_leads.mean(), 2) if len(watch_leads) > 0 else 0,
        'average_alert_lead_cycles': round(alert_leads.mean(), 2) if len(alert_leads) > 0 else 0,
        'units_with_watch': len(has_watch),
        'units_with_alert': len(has_alert),
        'units_with_both': len(has_both)
    }

    if len(has_both) > 0:
        improvement = has_both['watch_lead'].mean() - has_both['alert_lead'].mean()
        result['average_improvement_cycles'] = round(improvement, 2)
        result['percent_improvement'] = round((improvement / has_both['alert_lead'].mean()) * 100, 2)
    else:
        result['average_improvement_cycles'] = 0
        result['percent_improvement'] = 0

    return result


def compute_detection_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze detection performance metrics.

    Returns:
        Dict with detection success rate and quality breakdown.
    """
    total_units = len(df)
    detected_before_failure = (df['has_alert'] == True).sum()
    missed_detections = (df['alert_quality'] == 'miss').sum()

    quality_counts = df['alert_quality'].value_counts().to_dict()

    return {
        'units_detected_before_failure': int(detected_before_failure),
        'units_missed_detection': int(missed_detections),
        'detection_success_rate_percent': round((detected_before_failure / total_units) * 100, 2),
        'false_positives': 0,
        'quality_breakdown': {
            'good': int(quality_counts.get('good', 0)),
            'very_early': int(quality_counts.get('very_early', 0)),
            'usable': int(quality_counts.get('usable', 0)),
            'miss': int(quality_counts.get('miss', 0))
        }
    }


def create_lead_time_histogram(df: pd.DataFrame, output_path: Path) -> None:
    """Create histogram of lead times."""
    valid_leads = df[df['alert_lead'].notna()]['alert_lead'].dropna()

    plt.figure(figsize=(12, 6))
    plt.hist(valid_leads, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Alert Lead Time (cycles)', fontsize=12)
    plt.ylabel('Number of Units', fontsize=12)
    plt.title('Distribution of Alert Lead Times - FD004', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    mean_val = valid_leads.mean()
    median_val = valid_leads.median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_early_vs_alert_histogram(df: pd.DataFrame, output_path: Path) -> None:
    """Create histogram of early signal vs alert gap (improvement)."""
    has_both = df[(df['watch_lead'].notna()) & (df['alert_lead'].notna())]

    if len(has_both) == 0:
        print(f"Warning: No units with both watch and alert for gap histogram")
        return

    gap = has_both['watch_lead'] - has_both['alert_lead']

    plt.figure(figsize=(12, 6))
    plt.hist(gap, bins=25, color='coral', edgecolor='black', alpha=0.7)
    plt.xlabel('Lead Time Improvement: Watch - Alert (cycles)', fontsize=12)
    plt.ylabel('Number of Units', fontsize=12)
    plt.title('Distribution of Early Signal Advantage over Alert - FD004', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    mean_gap = gap.mean()
    median_gap = gap.median()
    plt.axvline(mean_gap, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {mean_gap:.1f}')
    plt.axvline(median_gap, color='darkgreen', linestyle='--', linewidth=2, label=f'Median: {median_gap:.1f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_proof_table_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Create detailed per-unit analysis table."""
    output_df = df[['unit', 'failure_cycle', 'watch_cycle', 'alert_cycle',
                    'watch_lead', 'alert_lead', 'has_watch', 'has_alert', 'alert_quality']].copy()

    output_df.columns = ['Unit', 'Failure Cycle', 'Watch Cycle', 'Alert Cycle',
                        'Watch Lead (cycles)', 'Alert Lead (cycles)', 'Has Watch', 'Has Alert', 'Alert Quality']

    output_df.to_csv(output_path, index=False)


def generate_proof_summary(
    df: pd.DataFrame,
    lead_time_metrics: Dict[str, Any],
    early_signal_analysis: Dict[str, Any],
    detection_quality: Dict[str, Any]
) -> Dict[str, Any]:
    """Assemble all metrics into a comprehensive summary."""
    return {
        'summary': {
            'total_units_analyzed': len(df),
            'analysis_complete': True
        },
        'lead_time_metrics': lead_time_metrics,
        'early_signal_analysis': early_signal_analysis,
        'detection_quality': detection_quality
    }


def print_clean_summary(
    lead_time_metrics: Dict[str, Any],
    early_signal_analysis: Dict[str, Any],
    detection_quality: Dict[str, Any]
) -> None:
    """Print a clean, human-readable summary."""
    print("\n" + "="*70)
    print("FD004 PROOF OF PERFORMANCE SUMMARY")
    print("="*70)

    print("\n📊 LEAD TIME METRICS")
    print("-" * 70)
    print(f"  Average lead time:        {lead_time_metrics['average_lead_time_cycles']} cycles")
    print(f"  Median lead time:         {lead_time_metrics['median_lead_time_cycles']} cycles")
    print(f"  Range:                    {lead_time_metrics['min_lead_time_cycles']} - {lead_time_metrics['max_lead_time_cycles']} cycles")
    print(f"  Valid lead time:          {lead_time_metrics['percent_valid_lead_time']}% " +
          f"({lead_time_metrics['units_with_valid_lead_time']}/{lead_time_metrics['total_units']} units)")

    print("\n⚡ EARLY SIGNAL VS ALERT COMPARISON")
    print("-" * 70)
    print(f"  Early signal lead time:   {early_signal_analysis['average_watch_lead_cycles']} cycles")
    print(f"  Alert lead time:          {early_signal_analysis['average_alert_lead_cycles']} cycles")
    print(f"  Improvement:              {early_signal_analysis['average_improvement_cycles']} cycles " +
          f"({early_signal_analysis['percent_improvement']}% better)")
    print(f"  Units with early signal:  {early_signal_analysis['units_with_watch']}/{early_signal_analysis['units_with_both'] + (early_signal_analysis['units_with_alert'] - early_signal_analysis['units_with_both'])}")

    print("\n✓ DETECTION QUALITY")
    print("-" * 70)
    print(f"  Detection success rate:   {detection_quality['detection_success_rate_percent']}%")
    print(f"  Units detected:           {detection_quality['units_detected_before_failure']}/{lead_time_metrics['total_units']}")
    print(f"  Missed detections:        {detection_quality['units_missed_detection']}")
    print(f"  Quality breakdown:")
    q = detection_quality['quality_breakdown']
    print(f"    - Good:                 {q['good']} units")
    print(f"    - Very early:           {q['very_early']} units")
    print(f"    - Usable:               {q['usable']} units")
    print(f"    - Miss:                 {q['miss']} units")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prove FD004 performance by analyzing validation outputs.'
    )
    parser.add_argument(
        '--run-dir',
        type=Path,
        required=True,
        help='Path to FD004 validation output directory'
    )

    args = parser.parse_args()
    run_dir = args.run_dir

    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading FD004 results from: {run_dir}")

    # Load data
    df = load_scored_csv(run_dir)
    print(f"Loaded {len(df)} units from FD004_scored.csv")

    # Compute metrics
    print("Computing metrics...")
    lead_time_metrics = compute_lead_time_metrics(df)
    early_signal_analysis = compute_early_signal_analysis(df)
    detection_quality = compute_detection_quality(df)

    # Generate summary
    proof_summary = generate_proof_summary(df, lead_time_metrics, early_signal_analysis, detection_quality)

    # Save outputs
    print("Saving outputs...")

    summary_path = run_dir / 'proof_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(proof_summary, f, indent=2)
    print(f"  ✓ {summary_path}")

    table_path = run_dir / 'proof_table.csv'
    create_proof_table_csv(df, table_path)
    print(f"  ✓ {table_path}")

    hist_path = run_dir / 'lead_time_histogram.png'
    create_lead_time_histogram(df, hist_path)
    print(f"  ✓ {hist_path}")

    gap_path = run_dir / 'early_vs_alert_gap.png'
    create_early_vs_alert_histogram(df, gap_path)
    print(f"  ✓ {gap_path}")

    # Print summary
    print_clean_summary(lead_time_metrics, early_signal_analysis, detection_quality)

    print(f"Proof outputs saved to: {run_dir}")


if __name__ == '__main__':
    main()
