#!/usr/bin/env python3
"""
Credible 4-node SII structural perturbation benchmark with:
- JSON calibration config
- Multi-seed robustness aggregation (mean/std)
- CSV + JSON + Markdown report artifacts
- Hard node-role and quality assertions
- GAL-2 disturbed-time support
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    # Colab has pandas; local env may not.
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas"])
    import pandas as pd

from neraium_core.alignment import StructuralEngine


CONFIG_PATH = "benchmark_calibration.json"
OUTPUT_JSON = "upgraded_multinode_test_results.json"
OUTPUT_TIMESERIES_CSV = "upgraded_multinode_test_timeseries.csv"
OUTPUT_NODE_SUMMARY_CSV = "upgraded_multinode_test_node_summary.csv"
OUTPUT_METRICS_CSV = "upgraded_multinode_test_metrics.csv"
OUTPUT_REPORT_MD = "upgraded_multinode_quality_report.md"
OUTPUT_PHASE_CONFUSION_CSV = "upgraded_multinode_phase_confusion.csv"

INTERPRETED_STATE_ALLOWED = {
    "NOMINAL_STRUCTURE",
    "REGIME_SHIFT_OBSERVED",
    "COUPLING_INSTABILITY_OBSERVED",
    "STRUCTURAL_INSTABILITY_OBSERVED",
    "COHERENCE_UNDER_CONSTRAINT",
}
STATE_ALLOWED = {"STABLE", "WATCH", "ALERT"}
CONFIDENCE_ALLOWED = {"low", "medium", "high"}


@dataclass(frozen=True)
class PhaseWindows:
    baseline_end: int
    perturbation_start: int
    perturbation_end: int
    recovery_start: int


def load_config(path: str = CONFIG_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing benchmark calibration config: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _get_gal2_time() -> float | None:
    api_key = os.getenv("GAL2_API_KEY")
    url = os.getenv("GAL2_TIME_URL", "https://api-v2.gal-2.com/time")
    if not api_key:
        return None
    try:
        req = urllib.request.Request(url, headers={"x-api-key": api_key})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            t = data.get("time") or data.get("gal2_time")
            return float(t) if t is not None else None
    except Exception:
        return None


def phase_for_step(step: int, pw: PhaseWindows) -> str:
    if step <= pw.baseline_end:
        return "baseline"
    if pw.perturbation_start <= step <= pw.perturbation_end:
        return "perturbation"
    if step >= pw.recovery_start:
        return "recovery"
    return "unknown"


def coherent_timestamps(n_steps: int) -> list[str]:
    return [str(float(i)) for i in range(n_steps)]


def disturbed_timestamps(
    n_steps: int,
    jitter_small: float,
    jitter_gap_low: float,
    jitter_gap_high: float,
    jitter_gap_count: int,
    rng: np.random.Generator,
) -> tuple[list[str], bool]:
    gal2_base = _get_gal2_time()
    if gal2_base is not None:
        base = np.array([gal2_base + i for i in range(n_steps)], dtype=float)
        used = True
    else:
        base = np.linspace(0, n_steps - 1, n_steps, dtype=float)
        used = False

    jitter = rng.uniform(-jitter_small, jitter_small, size=n_steps)
    gap_count = min(max(1, jitter_gap_count), n_steps)
    gap_idx = rng.choice(n_steps, size=gap_count, replace=False)
    for i in gap_idx:
        jitter[i] += rng.uniform(jitter_gap_low, jitter_gap_high)

    ts = np.maximum(0.0, np.sort(base + jitter))
    return [str(round(float(v), 4)) for v in ts], used


def parse_ts_float(ts: str) -> float:
    try:
        return float(ts)
    except (TypeError, ValueError):
        return 0.0


def validate_out(out: dict[str, Any], ctx: str) -> tuple[str, str, str]:
    state = str(out.get("state", "STABLE"))
    interpreted = str(out.get("interpreted_state", "NOMINAL_STRUCTURE"))
    confidence = str(out.get("confidence", "low"))
    if state not in STATE_ALLOWED:
        raise AssertionError(f"Invalid state {state!r} ({ctx})")
    if interpreted not in INTERPRETED_STATE_ALLOWED:
        raise AssertionError(f"Invalid interpreted_state {interpreted!r} ({ctx})")
    if confidence not in CONFIDENCE_ALLOWED:
        raise AssertionError(f"Invalid confidence {confidence!r} ({ctx})")
    return state, interpreted, confidence


def latent_signal(t: float, amp: float, w1: float, w2: float, phi2: float) -> float:
    return amp * (math.sin(w1 * t) + 0.5 * math.sin(w2 * t + phi2))


def sensor_values_for_node(
    node: str,
    condition: str,
    step: int,
    t_real: float,
    phase: str,
    calib: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, float]:
    latent = calib["latent"]
    z = latent_signal(
        t_real,
        float(latent["amplitude"]),
        float(latent["w1"]),
        float(latent["w2"]),
        float(latent["phi2"]),
    )

    nominal = calib["nominal"]
    a1 = float(nominal["a1"])
    a2 = float(nominal["a2"])
    a3 = float(nominal["a3"])
    s1 = a1 * z
    s2 = a2 * z
    s3 = a3 * z

    if node == "A":
        return {"s1": s1, "s2": s2, "s3": s3}

    if node == "B":
        if phase == "perturbation":
            b = calib["node_b"]
            s2 = float(b["s2_scale"]) * s2 + float(b["s2_offset"])
        return {"s1": s1, "s2": s2, "s3": s3}

    if node == "C":
        if phase == "perturbation":
            c = calib["node_c"]
            disturbed_mult = float(c.get("disturbed_time_noise_multiplier", 1.0)) if condition == "disturbed_time" else 1.0
            s2 = s2 + float(c["s2_noise_amplitude"]) * float(rng.normal(0.0, 1.0))
            s3 = float(c["s3_residual"]) * s3 + disturbed_mult * float(c["s3_noise_amplitude"]) * float(rng.normal(0.0, 1.0))
        return {"s1": s1, "s2": s2, "s3": s3}

    if node == "D":
        if phase == "perturbation":
            d = calib["node_d"]
            s3 = s3 + float(d["s3_noise_amplitude"]) * float(rng.normal(0.0, 1.0))
            if float(rng.uniform(0.0, 1.0)) < float(d["missing_prob"]):
                s2 = float("nan")
        return {"s1": s1, "s2": s2, "s3": s3}

    raise ValueError(f"Unknown node: {node}")


def run_single_seed(
    seed: int,
    cfg: dict[str, Any],
    tmp_dir: str,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    nodes: list[str] = list(cfg["nodes"])
    conditions: list[str] = list(cfg["conditions"])
    t_start = int(cfg["time"]["start"])
    t_end = int(cfg["time"]["end"])
    n_steps = t_end - t_start

    phase_cfg = cfg["phases"]
    pw = PhaseWindows(
        baseline_end=int(phase_cfg["baseline_end"]),
        perturbation_start=int(phase_cfg["perturbation_start"]),
        perturbation_end=int(phase_cfg["perturbation_end"]),
        recovery_start=int(phase_cfg["recovery_start"]),
    )

    # deterministic RNG streams
    rng_time = np.random.default_rng(seed + 101)
    rng_sensor_base = seed + 1000

    gal2_used_for_condition: dict[str, bool] = {}
    timestamps_by_condition: dict[str, list[str]] = {}
    for condition in conditions:
        if condition == "coherent_time":
            timestamps_by_condition[condition] = coherent_timestamps(n_steps)
            gal2_used_for_condition[condition] = False
        elif condition == "disturbed_time":
            dt = cfg["disturbed_time"]
            ts, used = disturbed_timestamps(
                n_steps=n_steps,
                jitter_small=float(dt["jitter_small"]),
                jitter_gap_low=float(dt["jitter_gap_low"]),
                jitter_gap_high=float(dt["jitter_gap_high"]),
                jitter_gap_count=int(dt["jitter_gap_count"]),
                rng=rng_time,
            )
            timestamps_by_condition[condition] = ts
            gal2_used_for_condition[condition] = used
        else:
            raise ValueError(f"Unknown condition: {condition}")

    out_rows: list[dict[str, Any]] = []
    for condition in conditions:
        ts_list = timestamps_by_condition[condition]
        for node in nodes:
            regime_path = os.path.join(tmp_dir, f"regime_{seed}_{node}_{condition}.json")
            engine = StructuralEngine(
                baseline_window=int(cfg["engine"]["baseline_window"]),
                recent_window=int(cfg["engine"]["recent_window"]),
                regime_store_path=regime_path,
            )

            sensor_rng = np.random.default_rng(rng_sensor_base + (ord(node) - ord("A")) * 997 + (0 if condition == "coherent_time" else 50000))

            for step in range(n_steps):
                ts = ts_list[step]
                t_real = parse_ts_float(ts)
                phase = phase_for_step(step, pw)
                sensors = sensor_values_for_node(node, condition, step, t_real, phase, cfg["calibration"], sensor_rng)
                frame = {
                    "timestamp": ts,
                    "site_id": f"node-{node}",
                    "asset_id": f"asset-{node}",
                    "sensor_values": sensors,
                }

                out = engine.process_frame(frame)
                state, interpreted, conf = validate_out(out, f"seed={seed} node={node} cond={condition} step={step}")
                out_rows.append(
                    {
                        "seed": seed,
                        "node": node,
                        "condition": condition,
                        "step": step,
                        "phase": phase,
                        "timestamp": ts,
                        "state": state,
                        "interpreted_state": interpreted,
                        "confidence": conf,
                        "latest_instability": float(out.get("latest_instability", 0.0)),
                        "structural_drift_score": float(out.get("structural_drift_score", 0.0)),
                        "drift_alert": bool(out.get("drift_alert", False)),
                        "signal_emitted": bool(out.get("signal_emitted", False)),
                        "data_quality_gate_passed": bool(out.get("data_quality_summary", {}).get("gate_passed", True)),
                        "missing_sensor_count": int(out.get("missing_sensor_count", 0) or 0),
                    }
                )
    return out_rows, gal2_used_for_condition


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def compute_seed_metrics(rows_seed: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    target_node = str(cfg["metrics"]["target_node"])
    conditions: list[str] = list(cfg["conditions"])
    out: list[dict[str, Any]] = []

    for cond in conditions:
        r = [x for x in rows_seed if x["condition"] == cond]
        positives = [x for x in r if x["node"] == target_node and x["phase"] == "perturbation"]
        negatives = [x for x in r if not (x["node"] == target_node and x["phase"] == "perturbation")]

        tp = sum(1 for x in positives if x["state"] in {"WATCH", "ALERT"})
        fn = len(positives) - tp
        fp = sum(1 for x in negatives if x["state"] in {"WATCH", "ALERT"})
        precision = _safe_rate(tp, tp + fp)
        recall = _safe_rate(tp, tp + fn)

        ab_baseline_recovery = [
            x for x in r if x["node"] in {"A", "B"} and x["phase"] in {"baseline", "recovery"}
        ]
        ab_stable_nominal = sum(
            1
            for x in ab_baseline_recovery
            if x["state"] == "STABLE" and x["interpreted_state"] == "NOMINAL_STRUCTURE"
        )
        baseline_stability_rate = _safe_rate(ab_stable_nominal, len(ab_baseline_recovery))

        non_target_pert = [x for x in r if x["phase"] == "perturbation" and x["node"] != target_node]
        false_positive_rate = _safe_rate(sum(1 for x in non_target_pert if x["state"] in {"WATCH", "ALERT"}), len(non_target_pert))

        c_recovery = [x for x in r if x["node"] == target_node and x["phase"] == "recovery"]
        recovery_success_rate = _safe_rate(
            sum(1 for x in c_recovery if x["state"] == "STABLE" and x["interpreted_state"] == "NOMINAL_STRUCTURE"),
            len(c_recovery),
        )

        c_pert = [x for x in r if x["node"] == target_node and x["phase"] == "perturbation"]
        others_pert = [x for x in r if x["node"] != target_node and x["phase"] == "perturbation"]
        c_peak = max((float(x["latest_instability"]) for x in c_pert), default=0.0)
        others_peak = max((float(x["latest_instability"]) for x in others_pert), default=0.0)
        peak_separation = c_peak - others_peak
        c_perturbation_alert_rate = _safe_rate(sum(1 for x in c_pert if x["state"] in {"WATCH", "ALERT"}), len(c_pert))

        # Node-role rates during perturbation
        node_rates: dict[str, float] = {}
        for node in cfg["nodes"]:
            node_pert = [x for x in r if x["phase"] == "perturbation" and x["node"] == node]
            node_rates[node] = _safe_rate(sum(1 for x in node_pert if x["state"] in {"WATCH", "ALERT"}), len(node_pert))

        out.append(
            {
                "seed": int(rows_seed[0]["seed"]) if rows_seed else None,
                "condition": cond,
                "alert_precision": precision,
                "alert_recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "baseline_stability_rate": baseline_stability_rate,
                "false_positive_rate": false_positive_rate,
                "recovery_success_rate": recovery_success_rate,
                "peak_separation": peak_separation,
                "c_perturbation_alert_rate": c_perturbation_alert_rate,
                "node_A_pert_alert_rate": node_rates.get("A", 0.0),
                "node_B_pert_alert_rate": node_rates.get("B", 0.0),
                "node_C_pert_alert_rate": node_rates.get("C", 0.0),
                "node_D_pert_alert_rate": node_rates.get("D", 0.0),
            }
        )

    return out


def aggregate_metrics(df_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in df_metrics.columns
        if c not in {"seed", "condition"} and pd.api.types.is_numeric_dtype(df_metrics[c])
    ]
    grouped = df_metrics.groupby("condition", as_index=False)[numeric_cols].agg(["mean", "std"])
    grouped.columns = [
        "condition" if c[0] == "condition" else f"{c[0]}_{c[1]}"
        for c in grouped.columns.to_flat_index()
    ]
    return grouped


def assert_quality_gates(df_agg: pd.DataFrame, cfg: dict[str, Any]) -> None:
    floors = cfg["assertion_floors"]
    role_bounds = cfg["node_role_bounds"]

    by_cond = {row["condition"]: row for _, row in df_agg.iterrows()}
    for condition in cfg["conditions"]:
        row = by_cond[condition]
        assert float(row["alert_precision_mean"]) >= float(floors["alert_precision"][condition]), (
            f"{condition} alert_precision_mean below floor: {row['alert_precision_mean']}"
        )
        assert float(row["alert_recall_mean"]) >= float(floors["alert_recall"][condition]), (
            f"{condition} alert_recall_mean below floor: {row['alert_recall_mean']}"
        )
        assert float(row["baseline_stability_rate_mean"]) >= float(floors["baseline_stability_rate"][condition]), (
            f"{condition} baseline_stability_rate_mean below floor: {row['baseline_stability_rate_mean']}"
        )
        assert float(row["recovery_success_rate_mean"]) >= float(floors["recovery_success_rate"][condition]), (
            f"{condition} recovery_success_rate_mean below floor: {row['recovery_success_rate_mean']}"
        )
        assert float(row["false_positive_rate_mean"]) <= float(floors["false_positive_rate_max"][condition]), (
            f"{condition} false_positive_rate_mean above max: {row['false_positive_rate_mean']}"
        )
        assert float(row["peak_separation_mean"]) >= float(floors["peak_separation_min"][condition]), (
            f"{condition} peak_separation_mean below min: {row['peak_separation_mean']}"
        )

        # Explicit node-role guarantees
        assert float(row["node_A_pert_alert_rate_mean"]) <= float(role_bounds["A_pert_alert_rate_max"][condition])
        assert float(row["node_B_pert_alert_rate_mean"]) <= float(role_bounds["B_pert_alert_rate_max"][condition])
        assert float(row["node_D_pert_alert_rate_mean"]) <= float(role_bounds["D_pert_alert_rate_max"][condition])
        assert float(row["node_C_pert_alert_rate_mean"]) >= float(role_bounds["C_pert_alert_rate_min"][condition])


def write_markdown_report(
    cfg: dict[str, Any],
    df_agg: pd.DataFrame,
    gal2_configured: bool,
    gal2_used_any: bool,
    path: str = OUTPUT_REPORT_MD,
) -> None:
    lines: list[str] = []
    lines.append("# Upgraded Multinode SII Benchmark Report")
    lines.append("")
    lines.append("## Run Context")
    lines.append(f"- Seeds: {cfg['seeds']}")
    lines.append(f"- Conditions: {cfg['conditions']}")
    lines.append(f"- Nodes: {cfg['nodes']}")
    lines.append(f"- GAL-2 configured: {gal2_configured}")
    lines.append(f"- GAL-2 used for disturbed_time: {gal2_used_any}")
    lines.append("")
    lines.append("## Aggregate Metrics (mean ± std)")
    lines.append("")
    for _, row in df_agg.iterrows():
        c = row["condition"]
        lines.append(f"### {c}")
        lines.append(f"- alert_precision: {row['alert_precision_mean']:.6f} ± {row['alert_precision_std']:.6f}")
        lines.append(f"- alert_recall: {row['alert_recall_mean']:.6f} ± {row['alert_recall_std']:.6f}")
        lines.append(
            f"- baseline_stability_rate: {row['baseline_stability_rate_mean']:.6f} ± {row['baseline_stability_rate_std']:.6f}"
        )
        lines.append(f"- false_positive_rate: {row['false_positive_rate_mean']:.6f} ± {row['false_positive_rate_std']:.6f}")
        lines.append(
            f"- recovery_success_rate: {row['recovery_success_rate_mean']:.6f} ± {row['recovery_success_rate_std']:.6f}"
        )
        lines.append(f"- peak_separation: {row['peak_separation_mean']:.6f} ± {row['peak_separation_std']:.6f}")
        lines.append("")

    lines.append("## Node Role Checks (perturbation alert rate means)")
    lines.append("")
    for _, row in df_agg.iterrows():
        lines.append(
            f"- {row['condition']}: "
            f"A={row['node_A_pert_alert_rate_mean']:.4f}, "
            f"B={row['node_B_pert_alert_rate_mean']:.4f}, "
            f"C={row['node_C_pert_alert_rate_mean']:.4f}, "
            f"D={row['node_D_pert_alert_rate_mean']:.4f}"
        )
    lines.append("")
    lines.append("## Phase-Aware Confusion Artifact")
    lines.append("")
    lines.append(f"- CSV: `{OUTPUT_PHASE_CONFUSION_CSV}`")
    lines.append("- Contains TP/FP/TN/FN and derived rates for each (condition, node, phase).")
    lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_executive_tables(df_agg: pd.DataFrame, df_seed_metrics: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Benchmark metrics table (multi-seed aggregate)")
    print("=" * 70)
    keep_cols = [
        "condition",
        "alert_precision_mean",
        "alert_recall_mean",
        "baseline_stability_rate_mean",
        "false_positive_rate_mean",
        "recovery_success_rate_mean",
        "peak_separation_mean",
        "node_A_pert_alert_rate_mean",
        "node_B_pert_alert_rate_mean",
        "node_C_pert_alert_rate_mean",
        "node_D_pert_alert_rate_mean",
    ]
    print(df_agg[keep_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("Per-seed metrics (head)")
    print("=" * 70)
    print(df_seed_metrics.head(12).to_string(index=False))


def summarize_node_counts(df_timeseries: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for condition in cfg["conditions"]:
        for node in cfg["nodes"]:
            sub = df_timeseries[(df_timeseries["condition"] == condition) & (df_timeseries["node"] == node)]
            counts = sub["state"].value_counts().to_dict()
            out.append(
                {
                    "condition": condition,
                    "node": node,
                    "alert_count": int(counts.get("ALERT", 0)),
                    "watch_count": int(counts.get("WATCH", 0)),
                    "stable_count": int(counts.get("STABLE", 0)),
                }
            )
    return out


def build_phase_confusion_rows(df_timeseries: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build phase-aware confusion rows per (condition, node, phase).

    Ground truth:
      - positive only for target node C during perturbation.
      - all other node/phase slices are negatives.
    Prediction:
      - state in {WATCH, ALERT}.
    """
    target_node = str(cfg["metrics"]["target_node"])
    rows: list[dict[str, Any]] = []

    for condition in cfg["conditions"]:
        for node in cfg["nodes"]:
            for phase in ["baseline", "perturbation", "recovery"]:
                sub = df_timeseries[
                    (df_timeseries["condition"] == condition)
                    & (df_timeseries["node"] == node)
                    & (df_timeseries["phase"] == phase)
                ]
                if sub.empty:
                    continue

                is_positive_slice = node == target_node and phase == "perturbation"
                pred_pos = sub["state"].isin(["WATCH", "ALERT"])

                if is_positive_slice:
                    tp = int(pred_pos.sum())
                    fn = int(len(sub) - tp)
                    fp = 0
                    tn = 0
                else:
                    fp = int(pred_pos.sum())
                    tn = int(len(sub) - fp)
                    tp = 0
                    fn = 0

                precision = _safe_rate(tp, tp + fp)
                recall = _safe_rate(tp, tp + fn)
                specificity = _safe_rate(tn, tn + fp) if (tn + fp) > 0 else 0.0
                fpr = _safe_rate(fp, fp + tn) if (fp + tn) > 0 else 0.0
                accuracy = _safe_rate(tp + tn, len(sub))

                rows.append(
                    {
                        "condition": condition,
                        "node": node,
                        "phase": phase,
                        "support": int(len(sub)),
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "precision": precision,
                        "recall": recall,
                        "specificity": specificity,
                        "false_positive_rate": fpr,
                        "accuracy": accuracy,
                    }
                )
    return rows


def main() -> int:
    cfg = load_config(CONFIG_PATH)
    seeds: list[int] = [int(s) for s in cfg["seeds"]]

    gal2_configured = bool(os.getenv("GAL2_API_KEY"))
    gal2_used_any = False

    all_rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="neraium_multiseed_credible_") as tmp_dir:
        for seed in seeds:
            rows_seed, gal2_by_cond = run_single_seed(seed, cfg, tmp_dir)
            all_rows.extend(rows_seed)
            all_metrics.extend(compute_seed_metrics(rows_seed, cfg))
            if bool(gal2_by_cond.get("disturbed_time", False)):
                gal2_used_any = True

    df_timeseries = pd.DataFrame(all_rows)
    df_seed_metrics = pd.DataFrame(all_metrics)
    df_agg = aggregate_metrics(df_seed_metrics)

    # hard assertions
    assert_quality_gates(df_agg, cfg)

    # write artifacts
    df_timeseries.to_csv(OUTPUT_TIMESERIES_CSV, index=False)
    node_summary_rows = summarize_node_counts(df_timeseries, cfg)
    pd.DataFrame(node_summary_rows).to_csv(OUTPUT_NODE_SUMMARY_CSV, index=False)
    df_seed_metrics.to_csv(OUTPUT_METRICS_CSV, index=False)
    phase_confusion_rows = build_phase_confusion_rows(df_timeseries, cfg)
    pd.DataFrame(phase_confusion_rows).to_csv(OUTPUT_PHASE_CONFUSION_CSV, index=False)

    # JSON summary
    out_json = {
        "description": "Credible 4-node SII structural perturbation benchmark with multi-seed calibration.",
        "config_path": CONFIG_PATH,
        "seeds": seeds,
        "nodes": cfg["nodes"],
        "conditions": cfg["conditions"],
        "phase_windows": cfg["phases"],
        "engine": cfg["engine"],
        "gal2_api_configured": gal2_configured,
        "gal2_time_url": os.getenv("GAL2_TIME_URL", "https://api-v2.gal-2.com/time") if gal2_configured else None,
        "gal2_used_for_disturbed_time": gal2_used_any,
        "aggregate_metrics": json.loads(df_agg.to_json(orient="records")),
        "artifacts": {
            "timeseries_csv": OUTPUT_TIMESERIES_CSV,
            "node_summary_csv": OUTPUT_NODE_SUMMARY_CSV,
            "metrics_csv": OUTPUT_METRICS_CSV,
            "report_md": OUTPUT_REPORT_MD,
            "phase_confusion_csv": OUTPUT_PHASE_CONFUSION_CSV,
        },
    }
    Path(OUTPUT_JSON).write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    write_markdown_report(
        cfg=cfg,
        df_agg=df_agg,
        gal2_configured=gal2_configured,
        gal2_used_any=gal2_used_any,
        path=OUTPUT_REPORT_MD,
    )
    print_executive_tables(df_agg, df_seed_metrics)

    print("\n" + "=" * 70)
    print("Outputs written")
    print("=" * 70)
    print(f"- {OUTPUT_JSON}")
    print(f"- {OUTPUT_TIMESERIES_CSV}")
    print(f"- {OUTPUT_NODE_SUMMARY_CSV}")
    print(f"- {OUTPUT_METRICS_CSV}")
    print(f"- {OUTPUT_REPORT_MD}")
    print(f"- {OUTPUT_PHASE_CONFUSION_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

