"""Report generation for validation results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.validation.metrics import DatasetMetrics


def generate_markdown_report(
    metrics_list: list[DatasetMetrics],
    processed_datasets: dict[str, bool],
    output_dir: Path,
) -> str:
    """
    Generate a markdown validation report.

    Args:
        metrics_list: List of computed metrics for each dataset
        processed_datasets: Dict of {dataset_name: success_bool}
        output_dir: Output directory path

    Returns:
        Markdown report as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Validation Report

**Generated:** {timestamp}

## Summary

This report contains benchmark metrics from local validation runs across available datasets.

### Processing Status

| Dataset | Status |
|---------|--------|
"""

    for dataset_name in ["FD001", "FD004", "IMS", "igrow"]:
        status = "✓ Processed" if processed_datasets.get(dataset_name.lower(), False) else "✗ Not Available"
        report += f"| {dataset_name} | {status} |\n"

    report += "\n## Benchmark Results\n\n"

    if metrics_list:
        # Create summary table
        report += "### Overall Metrics\n\n"
        report += "| Dataset | Units | Mean Lead Time | Median Lead Time | >50 Cycles | >100 Cycles |\n"
        report += "|---------|-------|----------------|------------------|------------|-------------|\n"

        for metrics in metrics_list:
            report += (
                f"| {metrics.dataset_name} | {metrics.unit_count} | "
                f"{metrics.mean_lead_time:.2f} | {metrics.median_lead_time:.2f} | "
                f"{metrics.p50_gt_50_cycles:.1f}% | {metrics.p50_gt_100_cycles:.1f}% |\n"
            )

        # Per-dataset details
        report += "\n### Per-Dataset Details\n\n"

        for metrics in metrics_list:
            report += f"#### {metrics.dataset_name}\n\n"
            report += f"- **Total Units:** {metrics.unit_count}\n"
            report += f"- **Total Records:** {metrics.total_records}\n"
            report += f"- **Mean Lead Time:** {metrics.mean_lead_time:.2f} cycles\n"
            report += f"- **Median Lead Time:** {metrics.median_lead_time:.2f} cycles\n"
            report += f"- **Units with >50 cycles:** {metrics.p50_gt_50_cycles:.1f}%\n"
            report += f"- **Units with >100 cycles:** {metrics.p50_gt_100_cycles:.1f}%\n\n"

            report += f"**Best Case:**\n"
            report += f"- Unit {metrics.best_case_unit}: {metrics.best_case_lead_time:.2f} cycles\n\n"

            report += f"**Median Case:**\n"
            report += f"- Unit {metrics.median_case_unit}: {metrics.median_case_lead_time:.2f} cycles\n\n"

            report += f"**Worst Case:**\n"
            report += f"- Unit {metrics.worst_case_unit}: {metrics.worst_case_lead_time:.2f} cycles\n\n"

            report += "**Outputs:**\n"
            report += f"- `{metrics.dataset_name.lower()}/stats.csv` - Summary statistics\n"
            report += f"- `{metrics.dataset_name.lower()}/lead_time_summary.csv` - Per-unit details\n"
            report += f"- `{metrics.dataset_name.lower()}/plots/` - Representative plots\n\n"

    else:
        report += "No datasets were successfully processed.\n"

    report += "\n## Notes\n\n"
    report += "- **Lead Time:** Cycles from first structural divergence to failure (or end-of-run if explicit failure not defined).\n"
    report += "- **Structural Drift Score:** Primary indicator of system degradation based on relational pattern divergence.\n"
    report += "- **Relational Instability Score:** Secondary indicator based on temporal sensor relationships.\n"
    report += "- **Early Signal Threshold:** Default 0.5 (configurable) defines structural divergence detection point.\n"
    report += "- **IMS Dataset:** Represents continuous real-world system operation (1 unit, 961 cycles). Lead time reflects early detection of structural divergence without explicit failure labels or training artifacts.\n"

    report += "\n## Plots\n\n"
    report += "For each dataset, three representative plots are generated:\n\n"
    report += "- **Best Case:** Unit with longest lead time (maximum structural divergence detection window)\n"
    report += "- **Median Case:** Unit at 50th percentile of lead time distribution\n"
    report += "- **Worst Case:** Unit with shortest lead time (minimum structural divergence detection window)\n\n"
    report += "Each plot overlays:\n"
    report += "- Structural Drift Score (primary indicator)\n"
    report += "- Relational Instability Score (secondary indicator)\n"
    report += "- Early Signal Threshold line (configurable)\n"
    report += "- Failure Point marker (if available)\n"
    report += "- End of Run marker\n"

    return report


def write_report(
    report_text: str,
    output_dir: Path,
) -> Path:
    """
    Write markdown report to file.

    Args:
        report_text: Markdown report text
        output_dir: Output directory

    Returns:
        Path to written report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path
