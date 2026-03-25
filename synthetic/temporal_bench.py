from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScenarioConfig:
    name: str
    entities: int = 6
    length: int = 80


def _base_noise(rng: np.random.Generator, scale: float = 0.02) -> float:
    return float(rng.normal(0.0, scale))


def generate_temporal_bench(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scenarios = [
        ScenarioConfig("same_static_context_different_degradation"),
        ScenarioConfig("different_static_context_same_degradation"),
        ScenarioConfig("degradation_with_temporary_recovery"),
        ScenarioConfig("two_competing_failure_paths"),
        ScenarioConfig("no_degradation_high_static_offset"),
        ScenarioConfig("mild_noise_only_variation"),
    ]

    rows: list[dict[str, float | int | str]] = []

    for sc in scenarios:
        for i in range(sc.entities):
            entity = f"{sc.name}_e{i+1}"
            static_offset = (i - sc.entities / 2) * 0.3
            degradation_rate = 0.01 + (0.012 * (i / max(1, sc.entities - 1)))

            for t in range(sc.length):
                base = math.sin(0.07 * t)
                s1 = base
                s2 = 0.8 * base
                s3 = 1.1 * base
                s4 = -0.6 * base

                if sc.name == "same_static_context_different_degradation":
                    s1 += degradation_rate * t
                    s2 += 0.7 * degradation_rate * t
                    s3 += 0.4 * degradation_rate * t
                elif sc.name == "different_static_context_same_degradation":
                    deg = 0.016 * t
                    s1 += static_offset + deg
                    s2 += 0.5 * static_offset + 0.7 * deg
                    s3 += -0.2 * static_offset + 0.5 * deg
                elif sc.name == "degradation_with_temporary_recovery":
                    deg = 0.018 * t
                    recovery = 0.0
                    if 32 <= t <= 48:
                        recovery = -0.22 * (1 - abs(t - 40) / 8)
                    s1 += deg + recovery
                    s2 += 0.8 * deg + 0.6 * recovery
                    s3 += 0.5 * deg + 0.3 * recovery
                elif sc.name == "two_competing_failure_paths":
                    if i % 2 == 0:
                        s1 += 0.02 * t
                        s2 += 0.015 * t
                        s4 += 0.002 * t * t / sc.length
                    else:
                        s3 -= 0.018 * t
                        s4 += 0.02 * t
                        s2 -= 0.006 * t
                elif sc.name == "no_degradation_high_static_offset":
                    s1 += static_offset
                    s2 += static_offset * 1.2
                    s3 -= static_offset * 0.8
                    s4 += static_offset * 0.5
                elif sc.name == "mild_noise_only_variation":
                    s1 += 0.02 * math.sin(0.27 * t)
                    s2 += 0.015 * math.cos(0.17 * t)
                    s3 += 0.02 * math.sin(0.21 * t + i)

                s1 += _base_noise(rng)
                s2 += _base_noise(rng)
                s3 += _base_noise(rng)
                s4 += _base_noise(rng)

                rows.append(
                    {
                        "scenario": sc.name,
                        "entity_id": entity,
                        "time": t,
                        "f1": s1,
                        "f2": s2,
                        "f3": s3,
                        "f4": s4,
                    }
                )

    return pd.DataFrame(rows)
