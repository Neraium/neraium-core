from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from neraium_core.service import StructuralMonitoringService


@dataclass
class UnitConfig:
    unit_id: str
    fault_mode: str
    total_steps: int
    stable_end: int
    degrade_end: int
    regime_schedule: list[int]


def _regime_signal_profiles(num_sensors: int, num_regimes: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=0.8, size=(num_regimes, num_sensors))


def _subsystem_map(num_sensors: int, num_subsystems: int = 4) -> list[np.ndarray]:
    return list(np.array_split(np.arange(num_sensors), num_subsystems))


def _degradation_vector(
    step: int,
    cfg: UnitConfig,
    fault_mode: str,
    num_sensors: int,
    subsystem_indices: list[np.ndarray],
) -> np.ndarray:
    if step < cfg.stable_end:
        return np.zeros(num_sensors)

    progress = (step - cfg.stable_end) / max(1, cfg.total_steps - cfg.stable_end)
    late_progress = max(0.0, (step - cfg.degrade_end) / max(1, cfg.total_steps - cfg.degrade_end))

    drift = np.zeros(num_sensors)

    if fault_mode == "coupling":
        scale = 0.4 * progress + 1.0 * late_progress
        for idx, subsystem in enumerate(subsystem_indices):
            drift[subsystem] += scale * (idx + 1) * 0.06
    elif fault_mode == "volatility":
        scale = 0.2 * progress + 0.8 * late_progress
        drift += scale * np.linspace(-0.3, 0.3, num_sensors)
    elif fault_mode == "imbalance":
        split = num_sensors // 2
        scale = 0.6 * progress + 0.6 * late_progress
        drift[:split] += scale * 0.5
        drift[split:] -= scale * 0.35
    elif fault_mode == "local_spread":
        lead = subsystem_indices[0]
        spread_scale = max(0.0, progress - 0.35)
        drift[lead] += (0.9 * progress + 0.8 * late_progress)
        drift += spread_scale * np.linspace(0.05, 0.4, num_sensors)

    return drift


def generate_fd004_synthetic_dataset(
    num_units: int = 20,
    num_steps: int = 180,
    num_sensors: int = 24,
    num_regimes: int = 4,
    seed: int = 7,
    site_id: str = "fd004-sim",
) -> tuple[list[dict[str, Any]], dict[str, UnitConfig]]:
    rng = np.random.default_rng(seed)
    sensor_names = [f"s{i:02d}" for i in range(1, num_sensors + 1)]
    base_profiles = _regime_signal_profiles(num_sensors, num_regimes, rng)
    subsystems = _subsystem_map(num_sensors)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    fault_modes = ["coupling", "volatility", "imbalance", "local_spread"]

    frames: list[dict[str, Any]] = []
    unit_configs: dict[str, UnitConfig] = {}

    for unit_index in range(1, num_units + 1):
        asset_id = f"unit_{unit_index:03d}"
        stable_end = int(num_steps * rng.uniform(0.35, 0.45))
        degrade_end = int(num_steps * rng.uniform(0.78, 0.88))
        regime_points = sorted(
            rng.choice(np.arange(12, num_steps - 12), size=max(2, num_regimes - 1), replace=False)
            .tolist()
        )
        regime_schedule = [0, *regime_points, num_steps]
        fault_mode = fault_modes[(unit_index - 1) % len(fault_modes)]

        cfg = UnitConfig(
            unit_id=asset_id,
            fault_mode=fault_mode,
            total_steps=num_steps,
            stable_end=stable_end,
            degrade_end=degrade_end,
            regime_schedule=regime_schedule,
        )
        unit_configs[asset_id] = cfg

        unit_shift = rng.normal(0.0, 0.2, size=num_sensors)
        subsystem_latent = rng.normal(0.0, 0.25, size=len(subsystems))

        for step in range(num_steps):
            regime_idx = sum(step >= cp for cp in regime_schedule[1:-1])
            regime_profile = base_profiles[regime_idx]
            degradation = _degradation_vector(step, cfg, fault_mode, num_sensors, subsystems)

            if step < stable_end:
                sigma = 0.06
            elif step < degrade_end:
                sigma = 0.1 + 0.16 * ((step - stable_end) / max(1, degrade_end - stable_end))
            else:
                sigma = 0.35

            coupling_gain = 0.0
            if fault_mode == "coupling" and step >= stable_end:
                coupling_gain = min(0.9, 0.3 + 0.8 * ((step - stable_end) / max(1, num_steps - stable_end)))

            for s_index, subsystem in enumerate(subsystems):
                subsystem_latent[s_index] = 0.9 * subsystem_latent[s_index] + rng.normal(0.0, 0.05)
                if coupling_gain > 0 and s_index > 0:
                    subsystem_latent[s_index] += coupling_gain * subsystem_latent[s_index - 1]

            signal = regime_profile + unit_shift + degradation
            for s_index, subsystem in enumerate(subsystems):
                signal[subsystem] += subsystem_latent[s_index]

            noisy = signal + rng.normal(0.0, sigma, size=num_sensors)
            sensor_values = {
                name: round(float(noisy[idx]), 6)
                for idx, name in enumerate(sensor_names)
            }

            frames.append(
                {
                    "timestamp": (start + timedelta(minutes=step)).isoformat(),
                    "site_id": site_id,
                    "asset_id": asset_id,
                    "sensor_values": sensor_values,
                    "_meta": {
                        "step": step,
                        "fault_mode": fault_mode,
                        "regime_index": regime_idx,
                        "stable_end": stable_end,
                        "degrade_end": degrade_end,
                        "regime_changes": regime_schedule[1:-1],
                    },
                }
            )

    return frames, unit_configs


def run_fd004_evaluation(
    num_units: int = 20,
    num_steps: int = 180,
    seed: int = 7,
    output_dir: str = "fd004_outputs",
) -> dict[str, Any]:
    frames, unit_configs = generate_fd004_synthetic_dataset(
        num_units=num_units,
        num_steps=num_steps,
        seed=seed,
    )

    by_unit: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        by_unit.setdefault(frame["asset_id"], []).append(frame)

    unit_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    prefinal_increase_count = 0
    stable_low_count = 0
    regime_false_high = 0
    total_regime_change_checks = 0

    for asset_id, unit_frames in by_unit.items():
        service = StructuralMonitoringService()
        cfg = unit_configs[asset_id]
        transitions = {"MEDIUM": None, "HIGH": None}
        instabilities: list[float] = []
        drifts: list[float] = []
        risk_levels: list[str] = []

        for frame in unit_frames:
            payload = {
                "timestamp": frame["timestamp"],
                "site_id": frame["site_id"],
                "asset_id": frame["asset_id"],
                "sensor_values": frame["sensor_values"],
            }
            result = service.ingest_payload(payload)
            step = int(frame["_meta"]["step"])
            analytics = result.get("experimental_analytics", {})
            instability = float(analytics.get("composite_instability", 0.0))

            row = {
                "asset_id": asset_id,
                "step": step,
                "fault_mode": cfg.fault_mode,
                "regime_index": frame["_meta"]["regime_index"],
                "structural_drift_score": float(result.get("structural_drift_score", 0.0)),
                "composite_instability": instability,
                "trend": result.get("trend", "UNKNOWN"),
                "risk_level": result.get("risk_level", "LOW"),
                "operator_message": result.get("operator_message", ""),
                "structural_analysis_available": bool(
                    result.get("structural_analysis_available", False)
                ),
            }
            all_rows.append(row)

            risk = str(row["risk_level"])
            if risk in transitions and transitions[risk] is None:
                transitions[risk] = step

            instabilities.append(instability)
            drifts.append(float(row["structural_drift_score"]))
            risk_levels.append(risk)

        stable_slice = instabilities[: cfg.stable_end]
        degraded_slice = instabilities[cfg.stable_end : cfg.degrade_end]
        prefinal_slice = instabilities[cfg.stable_end :]

        stable_avg = mean(stable_slice) if stable_slice else 0.0
        degraded_avg = mean(degraded_slice) if degraded_slice else 0.0

        if prefinal_slice and mean(prefinal_slice) > stable_avg * 1.35:
            prefinal_increase_count += 1

        if risk_levels[: cfg.stable_end]:
            stable_low_ratio = sum(1 for x in risk_levels[: cfg.stable_end] if x == "LOW") / len(
                risk_levels[: cfg.stable_end]
            )
            if stable_low_ratio >= 0.8:
                stable_low_count += 1

        for cp in cfg.regime_schedule[1:-1]:
            total_regime_change_checks += 1
            window = risk_levels[max(0, cp - 2) : min(len(risk_levels), cp + 3)]
            if any(level == "HIGH" for level in window) and cp < cfg.stable_end:
                regime_false_high += 1

        unit_summaries.append(
            {
                "asset_id": asset_id,
                "fault_mode": cfg.fault_mode,
                "first_medium_step": transitions["MEDIUM"],
                "first_high_step": transitions["HIGH"],
                "peak_instability": round(max(instabilities) if instabilities else 0.0, 4),
                "average_instability_stable": round(stable_avg, 4),
                "average_instability_degraded": round(degraded_avg, 4),
            }
        )

    overall_summary = {
        "units_total": len(by_unit),
        "units_with_prefinal_instability_increase": prefinal_increase_count,
        "units_mostly_low_in_stable_segment": stable_low_count,
        "regime_change_false_high_alerts": regime_false_high,
        "regime_change_windows_checked": total_regime_change_checks,
    }

    output = {
        "config": {
            "num_units": num_units,
            "num_steps": num_steps,
            "seed": seed,
            "num_sensors": 24,
            "num_fault_modes": 4,
        },
        "unit_summaries": unit_summaries,
        "overall_summary": overall_summary,
        "timeseries": all_rows,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fd004_synthetic_report.json"
    csv_path = out_dir / "fd004_synthetic_timeseries.csv"

    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    _print_console_summary(unit_summaries, overall_summary, str(json_path), str(csv_path))
    return output


def _print_console_summary(
    unit_summaries: list[dict[str, Any]],
    overall: dict[str, Any],
    json_path: str,
    csv_path: str,
) -> None:
    print("\nFD004-style synthetic evaluation summary")
    print("=" * 78)
    print(
        f"{'asset_id':<12} {'fault_mode':<12} {'first_MED':<10} {'first_HIGH':<10} "
        f"{'peak_instab':<12} {'stable_avg':<11} {'degraded_avg':<12}"
    )
    for row in unit_summaries:
        print(
            f"{row['asset_id']:<12} {row['fault_mode']:<12} {str(row['first_medium_step']):<10} "
            f"{str(row['first_high_step']):<10} {row['peak_instability']:<12} "
            f"{row['average_instability_stable']:<11} {row['average_instability_degraded']:<12}"
        )

    print("\nOverall:")
    for key, value in overall.items():
        print(f"- {key}: {value}")

    print(f"\nSaved report JSON: {json_path}")
    print(f"Saved timeseries CSV: {csv_path}")


def main() -> None:
    run_fd004_evaluation()


if __name__ == "__main__":
    main()
