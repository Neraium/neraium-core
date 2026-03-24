from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from neraium_core.alignment import StructuralEngine


@dataclass
class Scenario:
    name: str
    length: int
    transition_start: int | None
    generator: Callable[[int], dict[str, float]]


def _frame(t: int, sensors: dict[str, float]) -> dict:
    return {
        "timestamp": str(t),
        "site_id": "eval-site",
        "asset_id": "eval-asset",
        "sensor_values": sensors,
    }


def steady_bad_generator(t: int) -> dict[str, float]:
    base = math.sin(0.045 * t)
    if t < 90:
        return {"s1": base, "s2": 0.95 * base, "s3": 1.02 * base}
    degraded = 1.75 + 0.05 * math.sin(0.03 * t)
    return {"s1": degraded, "s2": 0.2 * degraded + 0.35, "s3": -0.1 * degraded + 0.25}


def active_break_generator(t: int) -> dict[str, float]:
    if t < 90:
        base = math.sin(0.04 * t)
        return {"s1": base, "s2": 0.94 * base, "s3": 1.01 * base}
    phase = 0.11 * (t - 90)
    base = math.sin(0.09 * t + phase) + 0.2 * math.sin(0.17 * t)
    return {
        "s1": base + 0.02 * (t - 90),
        "s2": 0.55 * base + 0.45 * math.sin(0.14 * t + phase),
        "s3": 0.25 * base - 0.55 * math.sin(0.19 * t),
    }


def regime_shift_generator(t: int) -> dict[str, float]:
    base_a = math.sin(0.05 * t)
    base_b = math.sin(0.08 * t + 1.2)
    if t < 80:
        mix = 0.0
    elif t < 125:
        mix = (t - 80) / 45.0
    else:
        mix = 1.0
    s1 = (1.0 - mix) * base_a + mix * (1.4 + 0.25 * base_b)
    s2 = (1.0 - mix) * (0.92 * base_a) + mix * (-0.2 * base_b + 0.55)
    s3 = (1.0 - mix) * (1.03 * base_a) + mix * (0.4 * base_b - 0.45)
    return {"s1": s1, "s2": s2, "s3": s3}


def noisy_oscillatory_generator(t: int) -> dict[str, float]:
    base = math.sin(0.11 * t) + 0.45 * math.sin(0.23 * t)
    noise = 0.05 * math.sin(0.93 * t) + 0.03 * math.cos(1.17 * t)
    if t < 110:
        return {"s1": base + noise, "s2": 0.9 * base - 0.4 * noise, "s3": 1.05 * base + 0.2 * noise}
    burst = 0.35 * math.sin(0.41 * t)
    return {
        "s1": base + burst + noise,
        "s2": 0.9 * base - 0.4 * noise + 0.2 * burst,
        "s3": 1.05 * base + 0.2 * noise - 0.3 * burst,
    }


def sharp_break_generator(t: int) -> dict[str, float]:
    if t < 95:
        base = math.sin(0.05 * t)
        return {"s1": base, "s2": 0.95 * base, "s3": 1.01 * base}
    if t == 95:
        return {"s1": 2.25, "s2": -1.85, "s3": 1.42}
    base = 1.75 + 0.03 * math.sin(0.04 * t)
    return {"s1": base, "s2": -0.3 * base + 0.4, "s3": 0.2 * base - 0.36}


def _first_timestep(states: list[str], target: str) -> int | None:
    for idx, state in enumerate(states):
        if state == target:
            return idx
    return None


def _first_watch_timestep(states: list[str]) -> int | None:
    for idx, state in enumerate(states):
        if state in {"WATCH", "ALERT"}:
            return idx
    return None


def _alert_flips(states: list[str]) -> int:
    flips = 0
    prev_alert = None
    for state in states:
        current_alert = state == "ALERT"
        if prev_alert is not None and current_alert != prev_alert:
            flips += 1
        prev_alert = current_alert
    return flips


def run_scenario(scenario: Scenario, *, transition_aware: bool) -> dict[str, float | int | None]:
    old_env = os.environ.get("NERAIUM_TRANSITION_AWARE")
    with tempfile.TemporaryDirectory() as td:
        os.environ["NERAIUM_TRANSITION_AWARE"] = "1" if transition_aware else "0"
        try:
            engine = StructuralEngine(
                baseline_window=40,
                recent_window=12,
                regime_store_path=str(Path(td) / "regime_eval.json"),
            )
        finally:
            if old_env is None:
                os.environ.pop("NERAIUM_TRANSITION_AWARE", None)
            else:
                os.environ["NERAIUM_TRANSITION_AWARE"] = old_env

        states: list[str] = []
        pressures: list[float] = []

        for t in range(scenario.length):
            out = engine.process_frame(_frame(t, scenario.generator(t)))
            states.append(str(out.get("state", "STABLE")))
            pressures.append(float(out.get("transition_pressure", 0.0)))

        first_watch = _first_watch_timestep(states)
        first_alert = _first_timestep(states, "ALERT")
        watch_alert_time = sum(1 for s in states if s in {"WATCH", "ALERT"})
        alert_time = sum(1 for s in states if s == "ALERT")

        onset_spike = None
        if scenario.transition_start is not None and scenario.transition_start >= 12:
            pre = pressures[max(0, scenario.transition_start - 12): scenario.transition_start]
            onset = pressures[scenario.transition_start: min(len(pressures), scenario.transition_start + 12)]
            if pre and onset:
                onset_spike = float(max(onset) - float(np.mean(pre)))

        chronic_pressure = float(np.mean(pressures[-24:])) if pressures else 0.0

        return {
            "first_watch": first_watch,
            "first_alert": first_alert,
            "alert_flips": _alert_flips(states),
            "time_watch_or_alert": watch_alert_time,
            "time_alert": alert_time,
            "onset_pressure_spike": onset_spike,
            "late_chronic_pressure": chronic_pressure,
        }


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report(output_path: Path, data: dict[str, dict[str, dict[str, float | int | None]]]) -> None:
    lines: list[str] = []
    lines.append("# Transition-aware Failure Detection Evaluation")
    lines.append("")
    lines.append("## Scenarios")
    lines.append("- steady_bad")
    lines.append("- active_break")
    lines.append("- regime_shift")
    lines.append("- noisy_oscillatory")
    lines.append("- sharp_break")
    lines.append("")
    lines.append("## Comparison Table")
    lines.append("")
    lines.append("| Scenario | Variant | first WATCH | first ALERT | alert flips | WATCH/ALERT time | ALERT time | onset pressure spike | late chronic pressure |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for scenario_name, variants in data.items():
        for variant_name, metrics in variants.items():
            lines.append(
                "| {scenario} | {variant} | {fw} | {fa} | {flips} | {wtime} | {atime} | {spike} | {latep} |".format(
                    scenario=scenario_name,
                    variant=variant_name,
                    fw=_fmt(metrics["first_watch"]),
                    fa=_fmt(metrics["first_alert"]),
                    flips=_fmt(metrics["alert_flips"]),
                    wtime=_fmt(metrics["time_watch_or_alert"]),
                    atime=_fmt(metrics["time_alert"]),
                    spike=_fmt(metrics["onset_pressure_spike"]),
                    latep=_fmt(metrics["late_chronic_pressure"]),
                )
            )

    lines.append("")
    lines.append("## Recommendation")
    active = data["active_break"]
    regime = data["regime_shift"]
    noisy = data["noisy_oscillatory"]
    steady = data["steady_bad"]
    sharp = data["sharp_break"]

    improved_transition = (
        active["transition_aware"]["first_watch"] is not None
        and (
            active["legacy"]["first_watch"] is None
            or int(active["transition_aware"]["first_watch"]) <= int(active["legacy"]["first_watch"])
        )
    )
    added_noise = int(noisy["transition_aware"]["alert_flips"]) > int(noisy["legacy"]["alert_flips"])
    chronic_suppressed = float(steady["transition_aware"]["late_chronic_pressure"]) < float(
        steady["legacy"]["late_chronic_pressure"]
    )
    regime_spike = float(regime["transition_aware"]["onset_pressure_spike"] or 0.0) > 0.08

    sharp_spike = float(sharp["transition_aware"]["onset_pressure_spike"] or 0.0) > 0.25

    if improved_transition and chronic_suppressed and regime_spike and sharp_spike and not added_noise:
        recommendation = "keep"
    elif improved_transition and (added_noise or not chronic_suppressed):
        recommendation = "revise"
    else:
        recommendation = "revert"

    lines.append(f"**Recommendation: {recommendation.upper()}**")
    lines.append("")
    lines.append(
        "Automated rationale: improved_transition={0}, chronic_suppressed={1}, regime_spike={2}, sharp_spike={3}, added_noise={4}.".format(
            improved_transition,
            chronic_suppressed,
            regime_spike,
            sharp_spike,
            added_noise,
        )
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    scenarios = [
        Scenario("steady_bad", 220, 90, steady_bad_generator),
        Scenario("active_break", 190, 90, active_break_generator),
        Scenario("regime_shift", 190, 80, regime_shift_generator),
        Scenario("noisy_oscillatory", 200, 110, noisy_oscillatory_generator),
        Scenario("sharp_break", 190, 95, sharp_break_generator),
    ]

    results: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for scenario in scenarios:
        legacy = run_scenario(scenario, transition_aware=False)
        transition = run_scenario(scenario, transition_aware=True)
        results[scenario.name] = {"legacy": legacy, "transition_aware": transition}

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_report(reports_dir / "transition_aware_evaluation.md", results)
    (reports_dir / "transition_aware_evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
