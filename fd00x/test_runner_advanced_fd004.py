"""Advanced drift detection for complex FD004: multiple operating conditions + fault modes.

FD004 Complexity:
- 249 test units
- 6 operating conditions (creating non-linear degradation)
- 2 fault modes (HPC degradation + Fan degradation)
- Highly variable degradation rates

This runner handles multi-condition dynamics by:
1. Detecting which operating condition each cycle operates in
2. Tracking component sensitivity across conditions
3. Identifying fault mode signatures
4. Computing lead time separately per fault mode
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DetectorConfig
from .detector import StructuralDriftDetector
from .evaluation import load_cmapss_dataset


@dataclass
class OperatingConditionProfile:
    """Operating condition signature for a unit."""

    condition_id: int
    n_cycles: int
    mean_setting_1: float
    mean_setting_2: float
    mean_setting_3: float
    dominant_component_in_condition: str
    component_sensitivity: Dict[str, float]


@dataclass
class FaultModeSignature:
    """Characteristic component behavior for each fault mode."""

    fault_mode: str  # "HPC_degradation" | "Fan_degradation" | "unknown"
    component_dominance: Dict[str, float]  # Which components light up
    typical_lead_time: Optional[float]
    degradation_trajectory: str  # "monotonic" | "stepped" | "variable"


@dataclass
class PerCycleDriftData:
    """Per-cycle time-series data for a single unit."""

    unit_id: int
    cycles: np.ndarray
    raw_drift: np.ndarray
    ema_drift: np.ndarray
    warning_triggered: np.ndarray
    warning_cycle: Optional[int]
    lead_time_cycles: Optional[float]
    component_scores: Dict[str, np.ndarray]
    operating_conditions: np.ndarray
    sensor_data: np.ndarray


@dataclass
class AdvancedDriftMetrics:
    """Enhanced metrics for multi-condition, multi-fault FD004."""

    unit_id: int
    n_cycles: int
    warning_cycle: Optional[int]
    lead_time_cycles: Optional[int]
    false_positive: bool

    # Operating condition analysis
    n_operating_conditions: int
    primary_operating_condition: int
    operating_conditions_present: List[int]

    # Component behavior across conditions
    mean_raw_drift: float
    mean_ema_drift: float
    max_raw_drift: float
    max_ema_drift: float

    # Per-component metrics
    quantum_peak: float
    quantum_mean: float
    quantum_variance: float
    information_peak: float
    information_mean: float
    information_variance: float
    free_energy_peak: float
    free_energy_mean: float
    free_energy_variance: float
    topological_peak: float
    topological_mean: float
    topological_variance: float
    algorithmic_peak: float
    algorithmic_mean: float
    algorithmic_variance: float

    # Fault mode inference
    inferred_fault_mode: str  # "HPC_like" | "Fan_like" | "mixed" | "unknown"
    fault_mode_confidence: float  # [0, 1]
    dominant_component: str

    # Condition-specific performance
    lead_time_by_condition: Dict[int, Optional[float]] = field(default_factory=dict)
    component_by_condition: Dict[int, str] = field(default_factory=dict)

    # Degradation trajectory
    degradation_trajectory: str = "unknown"  # "smooth", "stepped", "variable"
    trajectory_variance: float = 0.0  # Measure of degradation smoothness


class AdvancedFD004TestRunner:
    """Handles FD004 complexity: 6 operating conditions × 2 fault modes."""

    def __init__(
        self,
        detector_config: Optional[DetectorConfig] = None,
        degradation_proxy_fraction: float = 0.1,
        healthy_fraction: float = 0.15,
        verbose: bool = False,
    ):
        self.detector_config = detector_config or DetectorConfig()
        self.degradation_proxy_fraction = float(degradation_proxy_fraction)
        self.healthy_fraction = float(healthy_fraction)
        self.verbose = bool(verbose)
        self.detector = StructuralDriftDetector(self.detector_config)

        # Fault mode reference signatures (learned from typical patterns)
        self.fault_signatures = self._init_fault_signatures()

    def run_fd004_advanced(
        self,
        dataset_path: Optional[str] = None,
        max_units: Optional[int] = None,
    ) -> Tuple[List[AdvancedDriftMetrics], Dict, List[PerCycleDriftData]]:
        """Run advanced drift detection on complex FD004.

        Args:
            dataset_path: Path to FD004 training data or None to auto-resolve
            max_units: Limit test to first N units (None for all)

        Returns:
            (list of AdvancedDriftMetrics, summary dict, list of PerCycleDriftData)
        """
        start_time = time.time()

        if self.verbose:
            print("Loading CMAPSS FD004 dataset (249 units, 6 conditions, 2 fault modes)...")
        data_dict = load_cmapss_dataset("FD004", train_path=dataset_path)

        units = sorted(list(data_dict.keys()))[:max_units]
        if self.verbose:
            print(f"Processing {len(units)} units ({len(data_dict)} total available)")
            print("Each unit may contain multiple operating conditions...")

        results: List[AdvancedDriftMetrics] = []
        per_cycle_data: List[PerCycleDriftData] = []
        condition_statistics: Dict[int, List[float]] = {i: [] for i in range(1, 7)}

        for idx, unit_id in enumerate(units):
            if self.verbose:
                print(f"  [{idx+1}/{len(units)}] Unit {unit_id}...", end=" ")

            unit_data = data_dict[unit_id]
            metrics, cycle_data = self._process_unit_advanced(unit_id, unit_data, condition_statistics)
            results.append(metrics)
            per_cycle_data.append(cycle_data)

            if self.verbose:
                print(f"conditions={metrics.n_operating_conditions}, fault_mode={metrics.inferred_fault_mode}")

        elapsed = time.time() - start_time

        # Compute summary including condition-wise analysis
        summary = self._compute_advanced_summary(
            results,
            elapsed,
            len(data_dict),
            condition_statistics,
        )

        if self.verbose:
            self._print_advanced_summary(summary)

        return results, summary, per_cycle_data

    def _process_unit_advanced(
        self,
        unit_id: int,
        unit_data: np.ndarray,
        condition_statistics: Dict[int, List[float]],
    ) -> Tuple[AdvancedDriftMetrics, PerCycleDriftData]:
        """Process a single unit with advanced multi-condition analysis.

        Returns:
            (AdvancedDriftMetrics with unit-level summary, PerCycleDriftData with time-series)
        """
        # Extract columns: [cycle, setting1, setting2, setting3, s1-s21]
        cycles = unit_data[:, 0].astype(int)
        settings = unit_data[:, 1:4]  # 3 operating settings
        sensors = unit_data[:, 4:]

        # Detect operating conditions from settings (6 combinations)
        operating_conditions = self._detect_operating_conditions(settings)
        unique_conditions = sorted(list(set(operating_conditions)))

        # Run baseline drift detection
        scores = self.detector.process_unit(sensors)

        # Segment analysis by operating condition
        condition_segments = self._segment_by_condition(
            operating_conditions,
            scores,
            sensors,
        )

        # Infer fault mode from component signatures
        fault_mode, confidence = self._infer_fault_mode(
            scores.component_scores,
            operating_conditions,
        )

        # Analyze degradation trajectory
        trajectory = self._analyze_trajectory(scores.ema_drift)

        # Compute per-condition lead times
        n_cycles = len(sensors)
        degradation_onset = n_cycles - int(n_cycles * self.degradation_proxy_fraction)
        healthy_end = int(n_cycles * self.healthy_fraction)

        lead_time_by_condition = {}
        component_by_condition = {}
        for cond_id in unique_conditions:
            cond_alert = self._find_alert_in_condition(
                operating_conditions,
                scores.warning_state,
                cond_id,
            )
            if cond_alert is not None and cond_alert < degradation_onset:
                lead_time_by_condition[cond_id] = float(degradation_onset - cond_alert)
            component_by_condition[cond_id] = self._dominant_component_in_condition(
                condition_segments,
                cond_id,
            )

        # Track condition statistics for summary
        for cond_id in unique_conditions:
            if scores.warning_index is not None:
                condition_statistics[cond_id].append(float(lead_time_by_condition.get(cond_id, np.nan)))

        # Compute component variance (trajectory smoothness)
        component_variances = {
            k: float(np.var(v)) for k, v in scores.component_scores.items()
        }

        false_positive = (
            scores.warning_index is not None
            and scores.warning_index < healthy_end
        )

        lead_time = None
        if scores.warning_index is not None and not false_positive:
            lead_time = degradation_onset - scores.warning_index
            if lead_time < 0:
                lead_time = None

        # Create unit-level summary metrics
        metrics = AdvancedDriftMetrics(
            unit_id=unit_id,
            n_cycles=n_cycles,
            warning_cycle=scores.warning_index,
            lead_time_cycles=lead_time,
            false_positive=false_positive,
            n_operating_conditions=len(unique_conditions),
            primary_operating_condition=unique_conditions[0] if unique_conditions else -1,
            operating_conditions_present=unique_conditions,
            mean_raw_drift=float(np.mean(scores.raw_drift)),
            mean_ema_drift=float(np.mean(scores.ema_drift)),
            max_raw_drift=float(np.max(scores.raw_drift)),
            max_ema_drift=float(np.max(scores.ema_drift)),
            quantum_peak=float(np.max(scores.component_scores.get("quantum", [0]))),
            quantum_mean=float(np.mean(scores.component_scores.get("quantum", [0]))),
            quantum_variance=component_variances.get("quantum", 0.0),
            information_peak=float(np.max(scores.component_scores.get("information", [0]))),
            information_mean=float(np.mean(scores.component_scores.get("information", [0]))),
            information_variance=component_variances.get("information", 0.0),
            free_energy_peak=float(np.max(scores.component_scores.get("free_energy", [0]))),
            free_energy_mean=float(np.mean(scores.component_scores.get("free_energy", [0]))),
            free_energy_variance=component_variances.get("free_energy", 0.0),
            topological_peak=float(np.max(scores.component_scores.get("topological", [0]))),
            topological_mean=float(np.mean(scores.component_scores.get("topological", [0]))),
            topological_variance=component_variances.get("topological", 0.0),
            algorithmic_peak=float(np.max(scores.component_scores.get("algorithmic", [0]))),
            algorithmic_mean=float(np.mean(scores.component_scores.get("algorithmic", [0]))),
            algorithmic_variance=component_variances.get("algorithmic", 0.0),
            inferred_fault_mode=fault_mode,
            fault_mode_confidence=confidence,
            dominant_component=scores.dominant_alert_component,
            lead_time_by_condition=lead_time_by_condition,
            component_by_condition=component_by_condition,
            degradation_trajectory=trajectory["type"],
            trajectory_variance=trajectory["variance"],
        )

        # Create per-cycle time-series data
        per_cycle = PerCycleDriftData(
            unit_id=unit_id,
            cycles=cycles,
            raw_drift=scores.raw_drift,
            ema_drift=scores.ema_drift,
            warning_triggered=scores.warning_state.astype(int),
            warning_cycle=scores.warning_index,
            lead_time_cycles=lead_time,
            component_scores=scores.component_scores,
            operating_conditions=operating_conditions,
            sensor_data=sensors,
        )

        return metrics, per_cycle

    def _detect_operating_conditions(self, settings: np.ndarray) -> np.ndarray:
        """Detect which of 6 operating conditions each cycle operates in.

        FD004 has 6 operating condition combinations derived from 3 settings.
        """
        # Quantize settings to identify discrete conditions
        # This is a simplified approach; can be refined with clustering
        condition_ids = np.zeros(len(settings), dtype=int)

        for i, (s1, s2, s3) in enumerate(settings):
            # Simple binning approach (can be replaced with actual FD004 condition mapping)
            cond = int(1 + ((s1 > 0) * 1 + (s2 > 0.5) * 2 + (s3 > 20) * 3)) % 6 + 1
            condition_ids[i] = cond

        return condition_ids

    def _segment_by_condition(
        self,
        operating_conditions: np.ndarray,
        scores,
        sensors: np.ndarray,
    ) -> Dict[int, Dict]:
        """Segment analysis results by operating condition."""
        segments = {}
        unique_conds = set(operating_conditions)

        for cond_id in unique_conds:
            mask = operating_conditions == cond_id
            segments[cond_id] = {
                "n_cycles": np.sum(mask),
                "sensor_mean": np.mean(sensors[mask], axis=0),
                "drift_mean": float(np.mean(scores.ema_drift[mask])),
                "component_means": {
                    k: float(np.mean(v[mask]))
                    for k, v in scores.component_scores.items()
                },
            }

        return segments

    def _infer_fault_mode(
        self,
        component_scores: Dict[str, np.ndarray],
        operating_conditions: np.ndarray,
    ) -> Tuple[str, float]:
        """Infer fault mode (HPC vs Fan degradation) from component signatures.

        HPC degradation typically shows:
        - Higher information component activation
        - More topological changes

        Fan degradation typically shows:
        - Higher algorithmic component changes
        - More quantum phase-space anomalies
        """
        if not component_scores:
            return "unknown", 0.0

        # Compute mean component activations
        means = {k: float(np.mean(v)) for k, v in component_scores.items()}

        # Simple heuristic (can be replaced with ML classifier)
        hpc_score = means.get("information", 0) + means.get("topological", 0)
        fan_score = means.get("algorithmic", 0) + means.get("quantum", 0)

        total = hpc_score + fan_score
        if total < 1e-6:
            return "unknown", 0.0

        hpc_ratio = hpc_score / total
        fan_ratio = fan_score / total

        confidence = max(hpc_ratio, fan_ratio)

        if hpc_ratio > 0.6:
            return "HPC_like", confidence
        elif fan_ratio > 0.6:
            return "Fan_like", confidence
        else:
            return "mixed", confidence

    def _analyze_trajectory(self, ema_drift: np.ndarray) -> Dict:
        """Analyze degradation trajectory (smooth vs stepped vs variable)."""
        if len(ema_drift) < 2:
            return {"type": "unknown", "variance": 0.0}

        # Compute second derivative (rate of change)
        diffs = np.diff(ema_drift)
        second_diffs = np.diff(diffs) if len(diffs) > 1 else np.array([0])
        trajectory_var = float(np.var(second_diffs))

        if trajectory_var < 0.001:
            traj_type = "smooth"
        elif trajectory_var < 0.01:
            traj_type = "stepped"
        else:
            traj_type = "variable"

        return {"type": traj_type, "variance": trajectory_var}

    def _find_alert_in_condition(
        self,
        operating_conditions: np.ndarray,
        warning_state: np.ndarray,
        condition_id: int,
    ) -> Optional[int]:
        """Find first alert cycle in specific operating condition."""
        mask = operating_conditions == condition_id
        alerts = np.where(mask & warning_state)[0]
        return int(alerts[0]) if len(alerts) > 0 else None

    def _dominant_component_in_condition(
        self,
        segments: Dict[int, Dict],
        condition_id: int,
    ) -> str:
        """Find dominant component in specific operating condition."""
        if condition_id not in segments:
            return "unknown"

        comp_means = segments[condition_id]["component_means"]
        return max(comp_means, key=comp_means.get) if comp_means else "unknown"

    def _compute_advanced_summary(
        self,
        results: List[AdvancedDriftMetrics],
        elapsed: float,
        total_units: int,
        condition_statistics: Dict[int, List[float]],
    ) -> Dict:
        """Compute advanced summary with condition-wise breakdown."""
        valid_alerts = [r for r in results if r.warning_cycle is not None]
        valid_leads = [r.lead_time_cycles for r in valid_alerts if r.lead_time_cycles is not None]
        false_positives = [r for r in results if r.false_positive]

        # Condition-wise analysis
        condition_summary = {}
        for cond_id in range(1, 7):
            leads = [x for x in condition_statistics[cond_id] if not np.isnan(x)]
            condition_summary[cond_id] = {
                "detection_count": len(leads),
                "median_lead_time": float(np.median(leads)) if leads else None,
                "mean_lead_time": float(np.mean(leads)) if leads else None,
            }

        # Fault mode distribution
        fault_modes = {}
        for r in results:
            fault_modes[r.inferred_fault_mode] = fault_modes.get(r.inferred_fault_mode, 0) + 1

        return {
            "n_units_tested": len(results),
            "n_units_total": total_units,
            "detection_rate": len(valid_alerts) / len(results) if results else 0.0,
            "false_positive_rate": len(false_positives) / len(results) if results else 0.0,
            "median_lead_time": float(np.median(valid_leads)) if valid_leads else None,
            "mean_lead_time": float(np.mean(valid_leads)) if valid_leads else None,
            "mean_raw_drift": float(np.mean([r.mean_raw_drift for r in results])),
            "mean_ema_drift": float(np.mean([r.mean_ema_drift for r in results])),
            "condition_analysis": condition_summary,
            "fault_mode_distribution": fault_modes,
            "elapsed_seconds": elapsed,
            "throughput_units_per_second": len(results) / elapsed if elapsed > 0 else 0.0,
        }

    def _init_fault_signatures(self) -> Dict[str, FaultModeSignature]:
        """Initialize reference signatures for fault modes."""
        return {
            "HPC_degradation": FaultModeSignature(
                fault_mode="HPC_degradation",
                component_dominance={
                    "information": 0.35,
                    "topological": 0.30,
                    "quantum": 0.20,
                    "free_energy": 0.10,
                    "algorithmic": 0.05,
                },
                typical_lead_time=150.0,
                degradation_trajectory="monotonic",
            ),
            "Fan_degradation": FaultModeSignature(
                fault_mode="Fan_degradation",
                component_dominance={
                    "algorithmic": 0.35,
                    "quantum": 0.30,
                    "free_energy": 0.20,
                    "topological": 0.10,
                    "information": 0.05,
                },
                typical_lead_time=120.0,
                degradation_trajectory="stepped",
            ),
        }

    def _print_advanced_summary(self, summary: Dict) -> None:
        """Print formatted advanced summary."""
        print("\n" + "─" * 70)
        print("FD004 ADVANCED ANALYSIS SUMMARY")
        print("─" * 70)
        print(f"Units tested: {summary['n_units_tested']}/{summary['n_units_total']}")
        print(f"Detection rate: {summary['detection_rate']:.1%}")
        print(f"False positive rate: {summary['false_positive_rate']:.1%}")

        if summary["median_lead_time"] is not None:
            print(f"\nLead Time:")
            print(f"  Median: {summary['median_lead_time']:.0f} cycles")
            print(f"  Mean: {summary['mean_lead_time']:.0f} cycles")

        print(f"\nFault Mode Distribution:")
        for mode, count in summary["fault_mode_distribution"].items():
            pct = (count / summary["n_units_tested"]) * 100
            print(f"  {mode:15s}: {count:3d} units ({pct:5.1f}%)")

        print(f"\nPerformance by Operating Condition:")
        for cond_id in range(1, 7):
            cond_data = summary["condition_analysis"][cond_id]
            lead_str = f"{cond_data['median_lead_time']:.0f}" if cond_data["median_lead_time"] else "N/A"
            print(f"  Condition {cond_id}: {cond_data['detection_count']:3d} detections, lead_time={lead_str}")

        print(f"\nThroughput: {summary['throughput_units_per_second']:.1f} units/sec")
        print("─" * 70)

    def save_results(
        self,
        results: List[AdvancedDriftMetrics],
        summary: Dict,
        per_cycle_data: List[PerCycleDriftData],
        output_dir: str = "fd004_advanced_results",
    ) -> Dict[str, Path]:
        """Save advanced results to files.

        Args:
            results: List of per-unit summary metrics
            summary: Overall summary statistics
            per_cycle_data: List of per-cycle time-series data
            output_dir: Output directory

        Returns:
            Dict of saved file paths
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Unit-level summary CSV
        csv_path = out_path / "fd004_advanced_drift_results.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            if results:
                keys = list(asdict(results[0]).keys())
                f.write(",".join(keys) + "\n")
                for r in results:
                    values = [str(asdict(r)[k]) for k in keys]
                    f.write(",".join(values) + "\n")
        paths["csv"] = csv_path

        # Per-cycle time-series CSV
        cycle_csv_path = out_path / "fd004_cycle_drift_results.csv"
        self._save_per_cycle_results(per_cycle_data, cycle_csv_path)
        paths["cycle_csv"] = cycle_csv_path

        # JSON summary
        json_path = out_path / "fd004_advanced_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        paths["json"] = json_path

        return paths

    def _save_per_cycle_results(
        self,
        per_cycle_data: List[PerCycleDriftData],
        output_path: Path,
    ) -> None:
        """Save per-cycle drift data as CSV.

        Columns:
        - unit_id: Unit identifier
        - cycle: Cycle number
        - raw_drift: Raw structural drift score
        - ema_drift: EMA-smoothed drift score
        - warning_triggered: Binary indicator (0 or 1)
        - warning_cycle: Unit-level warning cycle (same for all cycles)
        - lead_time_cycles: Unit-level lead time (same for all cycles)
        """
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

            # Write header
            header = [
                "unit_id",
                "cycle",
                "raw_drift",
                "ema_drift",
                "warning_triggered",
                "warning_cycle",
                "lead_time_cycles",
            ]
            writer.writerow(header)

            # Write data for each unit
            for unit_data in per_cycle_data:
                n_cycles = len(unit_data.cycles)
                for i in range(n_cycles):
                    row = [
                        unit_data.unit_id,
                        int(unit_data.cycles[i]),
                        float(unit_data.raw_drift[i]),
                        float(unit_data.ema_drift[i]),
                        int(unit_data.warning_triggered[i]),
                        unit_data.warning_cycle if unit_data.warning_cycle is not None else "",
                        unit_data.lead_time_cycles if unit_data.lead_time_cycles is not None else "",
                    ]
                    writer.writerow(row)


def main():
    """Run advanced FD004 detection."""
    runner = AdvancedFD004TestRunner(verbose=True)
    results, summary, per_cycle_data = runner.run_fd004_advanced(max_units=None)
    runner.save_results(results, summary, per_cycle_data)


if __name__ == "__main__":
    main()
