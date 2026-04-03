"""
Platform-level timing for core hot paths (temporal representation, graph metrics, engine loop).

Run:  python -m neraium_core.benchmarks.platform_hotpath

Not a regression gate; use to compare before/after optimization runs on the same machine.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from neraium_core.context_invariant_representation import TemporalRepresentationConfig, build_temporal_representation
from neraium_core.graph import graph_metrics


def _repeat(
    fn: Callable[[], None],
    *,
    warmup: int = 2,
    repeats: int = 30,
) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def run_platform_benchmark(
    *,
    history_rows: int = 128,
    n_sensors: int = 16,
    engine_frames: int = 55,
    repeats: int = 15,
    include_engine: bool = False,
) -> dict[str, object]:
    """Return mean seconds per call for each benchmark segment.

    ``include_engine`` runs a multi-frame ``StructuralEngine`` segment (slow); default
    is off so the benchmark finishes quickly for CI/local smoke checks.
    """
    rng = np.random.default_rng(42)
    hist = rng.standard_normal(size=(history_rows, n_sensors))
    cfg = TemporalRepresentationConfig(mode="combined", reference_strategy="robust")

    def bench_repr() -> None:
        build_temporal_representation(hist, cfg, timestamps=np.arange(history_rows, dtype=float))

    adj = rng.integers(0, 2, size=(n_sensors, n_sensors))
    adj = np.maximum(adj, adj.T).astype(float)
    np.fill_diagonal(adj, 0)

    corr = rng.standard_normal(size=(n_sensors, n_sensors))
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)

    def bench_graph() -> None:
        graph_metrics(adj, corr=corr)

    out: dict[str, object] = {
        "build_temporal_representation_s": _repeat(bench_repr, repeats=repeats),
        "graph_metrics_s": _repeat(bench_graph, repeats=repeats),
        "config": {
            "history_rows": float(history_rows),
            "n_sensors": float(n_sensors),
            "engine_frames": float(engine_frames),
            "include_engine": include_engine,
        },
    }
    if include_engine:
        from neraium_core.alignment import StructuralEngine

        def bench_engine() -> None:
            eng = StructuralEngine(
                baseline_window=int(min(50, history_rows)),
                recent_window=12,
                representation_mode="combined",
            )
            for t in range(engine_frames):
                vals = {f"s{i}": float(rng.standard_normal()) for i in range(1, n_sensors + 1)}
                eng.process_frame(
                    {
                        "timestamp": str(t),
                        "site_id": "b",
                        "asset_id": "a1",
                        "sensor_values": vals,
                    }
                )

        out["structural_engine_segment_s"] = _repeat(bench_engine, repeats=2, warmup=1)
    return out


def main() -> None:
    import os

    include_engine = os.environ.get("NERAIUM_BENCH_ENGINE", "").strip().lower() in {"1", "true", "yes"}
    r = run_platform_benchmark(include_engine=include_engine)
    print("neraium_core platform_hotpath (mean seconds per call)")
    for k, v in r.items():
        if k == "config":
            print(f"  {k}: {v}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
