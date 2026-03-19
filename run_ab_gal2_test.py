from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

try:
    # Optional but preferred for clean, numerically stable entropy.
    from scipy.stats import entropy as scipy_entropy  # type: ignore
except Exception:  # pragma: no cover
    scipy_entropy = None

from neraium_core.alignment import StructuralEngine


NODES: List[str] = ["A", "B", "C", "D"]


def get_gal2_time(base_url: str = "https://api-v2.gal-2.com/time") -> float:
    """
    Return current time from GAL-2 API (requires GAL2_API_KEY in environment).
    On any failure (missing key, network error, etc.) fall back to local monotonic time.
    No API keys are hardcoded; key must be supplied via GAL2_API_KEY.
    """
    import time

    api_key = os.getenv("GAL2_API_KEY")
    if not api_key:
        raise ValueError("GAL2_API_KEY not set")

    try:
        import requests

        headers = {"x-api-key": api_key}
        r = requests.get(base_url, headers=headers, timeout=3)
        r.raise_for_status()
        data = r.json()
        return float(data.get("time") or data)
    except Exception:
        return time.time()


def _use_gal2_from_config_and_env(config: Dict[str, Any]) -> bool:
    """Resolve USE_GAL2: env USE_GAL2 overrides config use_gal2. No API keys read here."""
    env_val = os.getenv("USE_GAL2", "").strip().lower()
    if env_val in ("1", "true", "yes"):
        return True
    if env_val in ("0", "false", "no"):
        return False
    return bool(config.get("use_gal2", False))


def _parse_iso_to_epoch_seconds(ts: str) -> float:
    # Accept common RFC3339 forms like "...Z" and timezone-aware ISO strings.
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def _epoch_seconds_to_iso_z(epoch_seconds: float) -> str:
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    # isoformat gives +00:00; normalize to Z for readability.
    return dt.isoformat().replace("+00:00", "Z")


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))


def _safe_max(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmax(arr))


def _distribution_entropy_from_probs(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    if scipy_entropy is not None:
        # Shannon entropy (natural log by default from scipy).
        return float(scipy_entropy(probs, base=math.e))
    return float(-np.sum(probs * np.log(probs)))


def _normalize_distribution(counter: Counter[str], eps: float = 1e-12) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return {}
    # Do not add eps here; for KL/JS we do smoothing at compute time to preserve semantics.
    return {k: float(v) / total for k, v in counter.items() if v > 0}


def _vectorize_distributions(
    dists: List[Dict[str, float]],
    labels: List[str],
    eps: float = 0.0,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for d in dists:
        vec = np.array([float(d.get(lbl, 0.0)) for lbl in labels], dtype=float)
        if eps > 0:
            vec = vec + eps
        s = float(vec.sum())
        if s > 0:
            vec = vec / s
        out.append(vec)
    return out


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence in bits (bounded in [0, 1] for base-2 logs).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p + eps
    q = q + eps
    p = p / float(p.sum())
    q = q / float(q.sum())
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        a = a + eps
        a = a / float(a.sum())
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def build_base_frames(
    config: Dict[str, Any],
    seed: int,
    use_gal2: bool = False,
    gal2_base_url: str = "https://api-v2.gal-2.com/time",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build identical underlying base telemetry for all conditions.

    Each node has:
      - baseline sinusoidal signals
      - Node B: sustained load asymmetry after t=60
      - Node C: partition-style jitter/noise after t=60
      - Node D: dropout/rejoin between t=60–80 (implemented as dropped frames)

    When use_gal2 is True, frame timestamps come from get_gal2_time(gal2_base_url).
    When False, timestamps are simulated from config (start_timestamp, dt_seconds).
    """
    rng = np.random.default_rng(seed)

    time_steps = int(config.get("time_steps", 120))
    dt_seconds = float(config.get("dt_seconds", 1.0))
    start_timestamp = str(config.get("start_timestamp", "2026-03-01T00:00:00Z"))

    base_epoch = _parse_iso_to_epoch_seconds(start_timestamp)

    sensor_names: List[str] = list(config.get("sensor_names", ["temp", "pressure", "vibration"]))
    # Per-sensor baseline parameters.
    # Frequencies are chosen to create evolving correlation geometry (not just static sinusoids).
    default_amps = [1.0 + 0.2 * i for i in range(len(sensor_names))]
    default_freqs = [0.015 + 0.007 * i for i in range(len(sensor_names))]
    default_phases = [float(rng.uniform(0.0, 2.0 * math.pi)) for _ in sensor_names]
    default_offsets = [0.0 for _ in sensor_names]

    amps = list(config.get("sensor_amplitudes", default_amps))
    freqs = list(config.get("sensor_frequencies", default_freqs))
    phases = list(config.get("sensor_phases", default_phases))
    offsets = list(config.get("sensor_offsets", default_offsets))

    expected_len = len(sensor_names)
    for name, arr in [
        ("sensor_amplitudes", amps),
        ("sensor_frequencies", freqs),
        ("sensor_phases", phases),
        ("sensor_offsets", offsets),
    ]:
        if len(arr) != expected_len:
            raise ValueError(f"Config key `{name}` must have length {expected_len} (got {len(arr)}).")

    # Node behavior controls.
    t_asym = int(config.get("node_b_asymmetry_after_t", 60))
    asym_factor = float(config.get("node_b_asymmetry_factor", 1.25))
    # Affect only a subset of sensors for asymmetry, so correlations shift.
    asym_sensor_fraction = float(config.get("node_b_asymmetry_sensor_fraction", 0.5))

    t_jitter = int(config.get("node_c_jitter_after_t", 60))
    jitter_std = float(config.get("node_c_jitter_std", 0.18))
    # Partition-style behavior often shows stronger noise on a subset of sensors.
    jitter_sensor_fraction = float(config.get("node_c_jitter_sensor_fraction", 0.5))

    dropout_start = int(config.get("node_d_dropout_start_t", 60))
    dropout_end = int(config.get("node_d_dropout_end_t", 80))
    dropout_rejoin_offset = float(config.get("node_d_rejoin_offset", 0.6))

    # Stable sensor subsets for node B/C behavior.
    sensor_indices = list(range(len(sensor_names)))
    rng_for_subsets = np.random.default_rng(seed + 999)
    rng_for_subsets.shuffle(sensor_indices)
    asym_count = max(1, int(round(len(sensor_indices) * asym_sensor_fraction)))
    jitter_count = max(1, int(round(len(sensor_indices) * jitter_sensor_fraction)))
    asym_idxs = set(sensor_indices[:asym_count])
    jitter_idxs = set(sensor_indices[:jitter_count])

    node_frames: Dict[str, List[Dict[str, Any]]] = {n: [] for n in NODES}

    for t in range(time_steps):
        sim_time = t * dt_seconds
        base_values: List[float] = []
        for amp, freq, phase, offset in zip(amps, freqs, phases, offsets):
            base_values.append(offset + float(amp) * math.sin(2.0 * math.pi * float(freq) * sim_time + float(phase)))

        # Node A: baseline only.
        node_values: Dict[str, List[float]] = {"A": list(base_values)}

        # Node B: sustained load asymmetry after t=60.
        b_vals = list(base_values)
        if t >= t_asym:
            for i in range(len(b_vals)):
                if i in asym_idxs:
                    # Scale amplitude and add a small steady bias to shift correlation geometry.
                    b_vals[i] = float(b_vals[i]) * asym_factor + 0.15 * asym_factor
        node_values["B"] = b_vals

        # Node C: partition-style jitter/noise after t=60.
        c_vals = list(base_values)
        if t >= t_jitter:
            for i in range(len(c_vals)):
                if i in jitter_idxs:
                    c_vals[i] = float(c_vals[i]) + float(rng.normal(0.0, jitter_std))
                else:
                    # Keep the rest mostly coherent, so correlation geometry degrades in a structured way.
                    c_vals[i] = float(c_vals[i]) + float(rng.normal(0.0, jitter_std * 0.25))
        node_values["C"] = c_vals

        # Node D: dropout/rejoin between t=60–80.
        if dropout_start <= t < dropout_end:
            # Simulate loss of telemetry frames for Node D.
            # No frame is produced in this interval.
            pass
        else:
            d_vals = list(base_values)
            if t >= dropout_end:
                # Rejoin: perturb phase/offset to model "coming back" into a different coordination state.
                for i in range(len(d_vals)):
                    if i in asym_idxs:
                        d_vals[i] = float(d_vals[i]) + dropout_rejoin_offset
                    else:
                        d_vals[i] = float(d_vals[i]) + dropout_rejoin_offset * 0.4
            node_values["D"] = d_vals

        # Materialize frames for nodes produced at this t.
        if use_gal2:
            epoch_seconds = get_gal2_time(gal2_base_url)
        else:
            epoch_seconds = base_epoch + t * dt_seconds
        ts_iso = _epoch_seconds_to_iso_z(epoch_seconds)

        for node, values in node_values.items():
            sensor_values = {name: float(v) for name, v in zip(sensor_names, values)}
            node_frames[node].append(
                {
                    "timestamp": ts_iso,
                    "site_id": f"node-{node}",
                    "asset_id": f"asset-{node}",
                    "sensor_values": sensor_values,
                    # Internal metadata to enable deterministic time transformations.
                    "_base_t": t,
                    "_ts_seconds": float(epoch_seconds),
                }
            )

    return node_frames


def apply_time_distortion(frames: List[Dict[str, Any]], config: Dict[str, Any], rng: np.random.Generator) -> List[Dict[str, Any]]:
    """
    Standard-time A/B condition:
      - timestamp jitter
      - frame drops
      - duplicates
      - reordering
    """
    jitter_std = float(config.get("timestamp_jitter_std_seconds", 0.08))
    drop_prob = float(config.get("frame_drop_prob", 0.06))
    duplicate_prob = float(config.get("frame_duplicate_prob", 0.04))
    reorder_window = int(config.get("reorder_window", 8))
    reorder_prob = float(config.get("reorder_prob", 0.85))

    distorted: List[Dict[str, Any]] = []

    # Step 1: drop and duplicate based on Bernoulli sampling.
    for frame in frames:
        if rng.random() < drop_prob:
            continue

        # Apply timestamp jitter (may create non-monotonic ordering in real systems).
        f1 = copy.deepcopy(frame)
        jitter = float(rng.normal(0.0, jitter_std))
        f1["_ts_seconds"] = float(f1["_ts_seconds"]) + jitter
        f1["timestamp"] = _epoch_seconds_to_iso_z(float(f1["_ts_seconds"]))
        distorted.append(f1)

        # Possibly emit a duplicate (same sensor values, perturbed timestamp).
        if rng.random() < duplicate_prob:
            f2 = copy.deepcopy(frame)
            jitter2 = float(rng.normal(0.0, jitter_std))
            f2["_ts_seconds"] = float(f2["_ts_seconds"]) + jitter2
            f2["timestamp"] = _epoch_seconds_to_iso_z(float(f2["_ts_seconds"]))
            distorted.append(f2)

    # Step 2: reordering within local windows (structured, not fully random permutation).
    if len(distorted) <= 1 or reorder_window <= 1:
        return distorted

    if rng.random() >= reorder_prob:
        return distorted

    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(distorted):
        chunk = distorted[i : i + reorder_window]
        if len(chunk) > 1:
            # Shuffle the chunk to simulate out-of-order arrival.
            perm = rng.permutation(len(chunk))
            chunk = [chunk[j] for j in perm]
        out.extend(chunk)
        i += reorder_window
    return out


def enforce_coherent_time(frames: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Coherent-time B condition:
      - strict monotonic ordering
      - no jitter or reordering
      - simulate ideal temporal substrate (GAL-2)
    """
    dt_seconds = float(config.get("dt_seconds", 1.0))
    start_timestamp = str(config.get("start_timestamp", "2026-03-01T00:00:00Z"))
    base_epoch = _parse_iso_to_epoch_seconds(start_timestamp)

    # Start from the base sequence and enforce monotonic timestamps based on `_base_t`.
    ordered = sorted(frames, key=lambda f: int(f["_base_t"]))
    out: List[Dict[str, Any]] = []
    last_ts = -float("inf")
    for idx, frame in enumerate(ordered):
        f = copy.deepcopy(frame)
        base_t = int(f["_base_t"])
        new_ts = base_epoch + base_t * dt_seconds
        # Ensure strict monotonic ordering even if base inputs contain accidental ties.
        if new_ts <= last_ts:
            new_ts = last_ts + max(1e-6, 0.001 * dt_seconds)
        last_ts = new_ts
        f["_ts_seconds"] = float(new_ts)
        f["timestamp"] = _epoch_seconds_to_iso_z(float(new_ts))
        out.append(f)
    return out


def _process_interpreted_state(result: Dict[str, Any]) -> str:
    # alignment.StructuralEngine injects interpreted_state via decision_output()
    # once enough history exists. Early frames may not have it.
    v = result.get("interpreted_state")
    if v is None:
        return "MISSING_INTERPRETED_STATE"
    return str(v)


def run_condition(
    config: Dict[str, Any],
    condition_name: str,
    base_node_frames: Dict[str, List[Dict[str, Any]]],
    seed: int,
    engine_params: Dict[str, Any],
    tmp_dir: str,
) -> Dict[str, Any]:
    """
    For a single condition and a single seed/run:
      - process all frames per node using a fresh Neraium engine instance
      - collect outputs and compute per-node metrics
    """
    # Condition RNG controls all time-substrate distortions (not base signal generation).
    condition_id = 0 if condition_name == "standard_time" else 1
    rng_master = np.random.default_rng(seed + 1000 + condition_id * 10_000)

    node_runs: Dict[str, Any] = {}
    for node_idx, node in enumerate(NODES):
        node_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(node_seed)

        node_frames = copy.deepcopy(base_node_frames[node])
        if condition_name == "standard_time":
            node_frames = apply_time_distortion(node_frames, config, rng)
        elif condition_name == "coherent_time":
            node_frames = enforce_coherent_time(node_frames, config)
        else:
            raise ValueError(f"Unknown condition_name: {condition_name}")

        regime_store_path = os.path.join(tmp_dir, f"regimes_{condition_name}_{node}_seed{seed}.json")
        engine = StructuralEngine(
            baseline_window=int(engine_params.get("baseline_window", 50)),
            recent_window=int(engine_params.get("recent_window", 12)),
            window_stride=int(engine_params.get("window_stride", 1)),
            regime_store_path=regime_store_path,
        )

        interpreted_counts: Counter[str] = Counter()
        state_counts: Counter[str] = Counter()
        latest_instabilities: List[float] = []
        state_sequence: List[str] = []

        # Required wrapper signature.
        def process_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
            return engine.process_frame(frame)

        outputs_count = 0
        for frame in node_frames:
            result = process_frame(frame)
            outputs_count += 1
            state = str(result.get("state", "MISSING_STATE"))
            state_counts[state] += 1
            state_sequence.append(state)

            interpreted = _process_interpreted_state(result)
            interpreted_counts[interpreted] += 1

            latest_instabilities.append(float(result.get("latest_instability", 0.0)))

        # Metrics requested in the prompt.
        state_transition_count = int(sum(1 for i in range(1, len(state_sequence)) if state_sequence[i] != state_sequence[i - 1]))

        interpreted_distribution = _normalize_distribution(interpreted_counts)
        labels = sorted(interpreted_distribution.keys())
        probs = np.array([interpreted_distribution[lbl] for lbl in labels], dtype=float) if labels else np.array([], dtype=float)
        classification_entropy = _distribution_entropy_from_probs(probs)

        node_runs[node] = {
            "frames_processed": outputs_count,
            "state_counts": dict(state_counts),
            "interpreted_state_counts": dict(interpreted_counts),
            "interpreted_state_distribution": interpreted_distribution,
            "mean_score": _safe_mean(latest_instabilities),
            "max_score": _safe_max(latest_instabilities),
            "state_transition_count": state_transition_count,
            "classification_entropy": classification_entropy,
        }

    return {"seed": seed, "node_metrics": node_runs}


def compute_stability(condition_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cross-run stability:
    how consistent each node’s interpreted_state distribution is across runs.
    """
    per_node: Dict[str, Any] = {}

    for node in NODES:
        dists: List[Dict[str, float]] = [run["node_metrics"][node]["interpreted_state_distribution"] for run in condition_runs]
        labels = sorted(set().union(*(d.keys() for d in dists)))
        if not labels:
            per_node[node] = {
                "avg_js_divergence": 0.0,
                "stability_score": 1.0,
            }
            continue

        vectors = _vectorize_distributions(dists, labels=labels, eps=1e-12)

        js_values: List[float] = []
        for i, j in combinations(range(len(vectors)), 2):
            js_values.append(_js_divergence(vectors[i], vectors[j], eps=1e-12))

        avg_js = float(np.mean(js_values)) if js_values else 0.0
        stability_score = float(max(0.0, 1.0 - avg_js))  # higher is better (lower divergence => closer)
        per_node[node] = {
            "avg_js_divergence": avg_js,
            "stability_score": stability_score,
        }

    overall_stability = float(np.mean([per_node[n]["stability_score"] for n in NODES])) if NODES else 0.0
    return {"per_node": per_node, "overall_mean_stability_score": overall_stability}


def compute_separation(condition_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Separation metric:
    how distinct node behavior is from other nodes (via distribution distance).
    """
    pair_separations: List[float] = []
    per_node_to_others: Dict[str, List[float]] = {n: [] for n in NODES}

    for run in condition_runs:
        node_dists: Dict[str, Dict[str, float]] = {
            node: run["node_metrics"][node]["interpreted_state_distribution"] for node in NODES
        }
        labels = sorted(set().union(*(d.keys() for d in node_dists.values())))
        if not labels:
            continue

        node_vecs = {
            node: _vectorize_distributions([dist], labels=labels, eps=1e-12)[0] for node, dist in node_dists.items()
        }

        for n1, n2 in combinations(NODES, 2):
            js = _js_divergence(node_vecs[n1], node_vecs[n2], eps=1e-12)
            pair_separations.append(js)
            per_node_to_others[n1].append(js)
            per_node_to_others[n2].append(js)

    overall_mean_pairwise_js = float(np.mean(pair_separations)) if pair_separations else 0.0
    per_node_avg_from_others: Dict[str, float] = {
        node: float(np.mean(vals)) if vals else 0.0 for node, vals in per_node_to_others.items()
    }
    return {
        "overall_mean_pairwise_js_divergence": overall_mean_pairwise_js,
        "per_node_avg_js_from_others": per_node_avg_from_others,
    }


def summarize_results(condition_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    stability = compute_stability(condition_runs)
    separation = compute_separation(condition_runs)
    return {"stability": stability, "separation": separation}


def _print_comparison(standard_summary: Dict[str, Any], coherent_summary: Dict[str, Any]) -> None:
    std_stability = float(standard_summary["stability"]["overall_mean_stability_score"])
    coh_stability = float(coherent_summary["stability"]["overall_mean_stability_score"])

    std_sep = float(standard_summary["separation"]["overall_mean_pairwise_js_divergence"])
    coh_sep = float(coherent_summary["separation"]["overall_mean_pairwise_js_divergence"])

    print("A/B comparison summary (interpreted_state distributions):")
    print(f"- standard_time: avg stability={std_stability:.4f}, avg separation(JS)={std_sep:.4f}")
    print(f"- coherent_time: avg stability={coh_stability:.4f}, avg separation(JS)={coh_sep:.4f}")

    # "Better" primary criterion: higher stability. Secondary: higher separation.
    if coh_stability > std_stability + 1e-9:
        better = "coherent_time"
    elif std_stability > coh_stability + 1e-9:
        better = "standard_time"
    else:
        better = "coherent_time" if coh_sep > std_sep else "standard_time"

    print(f"Best (by stability, tie-break by separation): {better}")


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object at the top level.")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled A/B test for Neraium SII time substrate (GAL-2).")
    parser.add_argument("config_path", type=str, help="Path to JSON config (e.g. test_config.json).")
    args = parser.parse_args()

    config = _load_json_file(args.config_path)
    iterations = int(config.get("iterations", config.get("runs", 4)))
    base_seed = int(config.get("base_seed", 1337))

    seeds: List[int]
    if "seeds" in config and isinstance(config["seeds"], list) and config["seeds"]:
        seeds = [int(s) for s in config["seeds"]]
        if iterations != len(seeds):
            # Prefer explicit seeds length; still respect config.iterations if provided.
            iterations = min(iterations, len(seeds))
            seeds = seeds[:iterations]
    else:
        seeds = [base_seed + i for i in range(iterations)]

    engine_params = {
        "baseline_window": config.get("baseline_window", 50),
        "recent_window": config.get("recent_window", 12),
        "window_stride": config.get("window_stride", 1),
    }

    out_path = Path("ab_gal2_comparison.json")

    USE_GAL2 = _use_gal2_from_config_and_env(config)
    gal2_base_url = str(config.get("gal2_time_url", "https://api-v2.gal-2.com/time"))

    results: Dict[str, Any] = {"standard_time": {"runs": []}, "coherent_time": {"runs": []}}

    # Use a single temporary directory per entire experiment so regime files don't collide.
    with tempfile.TemporaryDirectory(prefix="neraium_ab_gal2_") as tmp_dir:
        for run_i, seed in enumerate(seeds):
            base_node_frames = build_base_frames(
                config, seed=seed, use_gal2=USE_GAL2, gal2_base_url=gal2_base_url
            )

            std_run = run_condition(
                config=config,
                condition_name="standard_time",
                base_node_frames=base_node_frames,
                seed=seed,
                engine_params=engine_params,
                tmp_dir=tmp_dir,
            )
            coh_run = run_condition(
                config=config,
                condition_name="coherent_time",
                base_node_frames=base_node_frames,
                seed=seed,
                engine_params=engine_params,
                tmp_dir=tmp_dir,
            )

            results["standard_time"]["runs"].append(std_run)
            results["coherent_time"]["runs"].append(coh_run)

            print(f"Run {run_i + 1}/{len(seeds)} complete (seed={seed}).")

        results["standard_time"]["summary"] = summarize_results(results["standard_time"]["runs"])
        results["coherent_time"]["summary"] = summarize_results(results["coherent_time"]["runs"])

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    _print_comparison(
        standard_summary=results["standard_time"]["summary"],
        coherent_summary=results["coherent_time"]["summary"],
    )

    print(f"Saved comparison output to: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

