from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

import neraium_core.alignment as alignment_module
from neraium_core.alignment import StructuralEngine


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 7
    n_sensors: int = 8
    stable_steps: int = 40
    drift_steps: int = 40
    unstable_steps: int = 30
    site_id: str = "demo-site"
    asset_id: str = "pump-A"


def phase_for_step(step: int, cfg: SimulationConfig) -> str:
    if step < cfg.stable_steps:
        return "stable"
    if step < cfg.stable_steps + cfg.drift_steps:
        return "drift"
    return "unstable"


def _mix_matrix(size: int, coupling: float) -> np.ndarray:
    base = np.eye(size)
    for i in range(size):
        for j in range(i):
            base[i, j] = coupling / (1 + i - j)
    return base


def _display_instability(composite_instability: float, drift: float) -> float:
    drift_floor = 0.65 * max(drift - 0.25, 0.0)
    blended = composite_instability + 0.20 * drift
    return max(composite_instability, drift_floor, blended)


def _trend_arrow(history: deque[float], lookback: int = 8) -> str:
    if len(history) < 4:
        return "→"

    recent = list(history)[-lookback:]
    half = max(len(recent) // 2, 1)
    early_avg = float(np.mean(recent[:half]))
    late_avg = float(np.mean(recent[half:]))
    delta = late_avg - early_avg

    if delta > 0.03:
        return "↑"
    if delta < -0.03:
        return "↓"
    return "→"


def _risk_and_message(drift: float, instability: float, trend: str) -> tuple[str, str]:
    if drift >= 1.8 and instability >= 1.0 and trend == "↑":
        return "HIGH", "System entering unstable regime"
    if drift >= 0.9 or instability >= 0.65 or trend == "↑":
        return "MEDIUM", "Structural relationships shifting"
    return "LOW", "System stable"


def _patch_engine_compatibility() -> None:
    if getattr(alignment_module, "_demo_compat_patched", False):
        return

    original_directional = alignment_module.directional_metrics
    original_subsystems = alignment_module.subsystem_spectral_measures

    def _directional_compat(matrix: np.ndarray) -> dict[str, float]:
        metrics = dict(original_directional(matrix))
        if "divergence" not in metrics and "causal_divergence" in metrics:
            metrics["divergence"] = float(metrics["causal_divergence"])
        return metrics

    def _subsystem_compat(matrix: np.ndarray) -> dict[str, float]:
        metrics = dict(original_subsystems(matrix))
        if "max_instability" not in metrics and "subsystem_instability" in metrics:
            metrics["max_instability"] = float(metrics["subsystem_instability"])
        return metrics

    alignment_module.directional_metrics = _directional_compat
    alignment_module.subsystem_spectral_measures = _subsystem_compat
    alignment_module._demo_compat_patched = True


def generate_sensor_stream(cfg: SimulationConfig) -> Iterable[dict]:
    rng = np.random.default_rng(cfg.seed)
    total_steps = cfg.stable_steps + cfg.drift_steps + cfg.unstable_steps

    latent = np.zeros(2, dtype=float)
    sensor_names = [f"sensor_{i + 1}" for i in range(cfg.n_sensors)]

    for t in range(total_steps):
        phase = phase_for_step(t, cfg)

        if phase == "stable":
            coupling = 0.12
            noise_scale = 0.01
            latent_persistence = 0.95
            volatility = 0.003
        elif phase == "drift":
            progress = (t - cfg.stable_steps) / max(cfg.drift_steps - 1, 1)
            coupling = 0.14 + 0.36 * progress
            noise_scale = 0.02 + 0.10 * progress
            latent_persistence = 0.95 - 0.20 * progress
            volatility = 0.006 + 0.08 * progress
        else:
            progress = (t - cfg.stable_steps - cfg.drift_steps) / max(cfg.unstable_steps - 1, 1)
            coupling = 0.55 + 0.30 * progress
            noise_scale = 0.15 + 0.20 * progress
            latent_persistence = 0.70 - 0.25 * progress
            volatility = 0.09 + 0.10 * progress

        latent = latent_persistence * latent + rng.normal(0.0, volatility, size=2)

        sensor_noise = rng.normal(0.0, noise_scale, size=cfg.n_sensors)
        sensor_base = np.empty(cfg.n_sensors, dtype=float)

        split = cfg.n_sensors // 2
        sensor_base[:split] = latent[0]
        sensor_base[split:] = latent[1]

        mix = _mix_matrix(cfg.n_sensors, coupling=coupling)
        values = mix @ (sensor_base + sensor_noise)

        yield {
            "timestamp": t,
            "site_id": cfg.site_id,
            "asset_id": cfg.asset_id,
            "sensor_values": {name: float(v) for name, v in zip(sensor_names, values, strict=False)},
            "phase": phase,
        }


def run_demo(cfg: SimulationConfig | None = None) -> None:
    cfg = cfg or SimulationConfig()
    _patch_engine_compatibility()
    engine = StructuralEngine(baseline_window=24, recent_window=8)

    instability_history: deque[float] = deque(maxlen=16)

    for frame in generate_sensor_stream(cfg):
        phase = frame.pop("phase")
        result = engine.process_frame(frame)

        drift = float(result.get("structural_drift_score", 0.0))
        analytics = result.get("experimental_analytics") or {}
        raw_instability = float(analytics.get("composite_instability", 0.0))
        instability = _display_instability(raw_instability, drift)

        instability_history.append(instability)
        trend = _trend_arrow(instability_history)

        risk_level, message = _risk_and_message(drift, instability, trend)

        print(f"[time={frame['timestamp']}] phase={phase}")
        print(f"drift={drift:.2f} | instability={instability:.2f} | trend={trend}")
        print(f"risk={risk_level}")
        print(f"message={message}")
        print("-" * 56)


if __name__ == "__main__":
    run_demo()
