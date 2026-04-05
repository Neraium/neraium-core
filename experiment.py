from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import eigvals, lstsq


DATA_PATH = Path(__file__).resolve().parent / "train_FD004.txt"

SIGNAL_COLUMNS = ["s_2", "s_3", "s_4", "s_7"]

WINDOW = 20
DEGRADATION_FRACTION = 0.8
EMA_ALPHA = 0.2
CUMULATIVE_N = 30

HEALTHY_FRACTION = 0.2
DETECTOR_PRESETS = {
    # Prioritises low false positives (default runtime behavior).
    "conservative": {
        "threshold_std": 2.5,
        "persistence": 5,
        "threshold_mode": "robust_mad",
        "threshold_percentile": 97.5,
    },
    # Compromise profile.
    "balanced": {
        "threshold_std": 2.0,
        "persistence": 4,
        "threshold_mode": "robust_mad",
        "threshold_percentile": 97.5,
    },
    # Prioritises early lead time at the cost of more false positives.
    "aggressive": {
        "threshold_std": 1.5,
        "persistence": 3,
        "threshold_mode": "mean_std",
        "threshold_percentile": 97.5,
    },
}
DEFAULT_PRESET = "conservative"
RUN_GRID_SEARCH = False
ENABLE_PLOTS = True
NUM_UNITS = 249


def load_fd004(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None)
    base_cols = ["unit", "cycle", "op_1", "op_2", "op_3"]
    sensor_cols = [f"s_{i}" for i in range(1, 22)]
    df.columns = base_cols + sensor_cols
    return df


def fit_var1(x: np.ndarray) -> np.ndarray:
    x_t = x[:-1]
    x_next = x[1:]
    a, _, _, _ = lstsq(x_t, x_next, rcond=None)
    return a


def normalize_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    if np.isclose(max_v, min_v):
        return np.zeros_like(x)
    return 2.0 * (x - min_v) / (max_v - min_v) - 1.0


def first_persistent_warning(signal: np.ndarray, threshold: float, persistence: int) -> int | None:
    return first_persistent_warning_with_gates(
        signal,
        threshold,
        persistence,
        require_upward_trend=True,
        slope_window=3,
        min_slope=0.0,
    )


def first_persistent_warning_with_gates(
    signal: np.ndarray,
    threshold: float,
    persistence: int,
    require_upward_trend: bool = True,
    slope_window: int = 3,
    min_slope: float = 0.0,
) -> int | None:
    run = 0
    for i in range(len(signal)):
        value = float(signal[i])
        above = value > threshold
        instant_slope = value - float(signal[i - 1]) if i > 0 else 0.0
        trending = (not require_upward_trend) or (i > 0 and instant_slope > 0.0)

        if min_slope <= 0.0:
            strong_trend = True
        else:
            if i >= slope_window:
                recent_slope = value - float(signal[i - slope_window])
            else:
                recent_slope = 0.0
            strong_trend = recent_slope > min_slope

        if above:
            if run == 0:
                if trending and strong_trend:
                    run = 1
            else:
                # Preserve persistence while the anomaly score stays above threshold,
                # even through local plateaus/wobbles.
                run += 1
            if run >= persistence:
                return i
        else:
            run = 0
    return None


def _informative_healthy_ema(healthy_reference_ema: np.ndarray) -> np.ndarray:
    """
    Remove non-informative warmup prefix values before robust thresholding.

    The VAR/warmup path can yield a leading run of zeros in drift EMA.  Those
    values are implementation artifacts (not healthy-behaviour variability) and
    can collapse robust median/MAD toward zero on shorter units.
    """
    values = np.asarray(healthy_reference_ema, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values

    magnitude = float(np.max(np.abs(values)))
    eps = max(1e-10, magnitude * 1e-6)
    informative_idx = np.flatnonzero(np.abs(values) > eps)
    if informative_idx.size == 0:
        return values
    return values[informative_idx[0]:]


def compute_healthy_threshold(
    healthy_reference_ema: np.ndarray,
    threshold_std: float,
    mode: str,
    threshold_percentile: float = 97.5,
) -> float:
    """
    Compute a leak-free warning threshold from healthy-reference EMA values only.

    Thresholding
    ------------
    threshold = percentile(informative_healthy_ema, p)
    where p is configured by ``threshold_percentile`` (default 97.5).
    """
    healthy_reference_ema = np.asarray(healthy_reference_ema, dtype=float)
    if healthy_reference_ema.size == 0:
        return float("inf")
    informative = _informative_healthy_ema(healthy_reference_ema)
    if informative.size == 0:
        return float("inf")
    return float(np.percentile(informative, threshold_percentile))


def compute_unit_timeseries(signals: pd.DataFrame) -> dict:
    ordered = signals.sort_values("cycle").reset_index(drop=True)
    x = ordered[SIGNAL_COLUMNS].to_numpy(dtype=float)
    t_total = len(x)

    baseline_scores = np.zeros(t_total)
    drift_arr = np.zeros(t_total)
    spectral_radius_scores = np.zeros(t_total)

    prev_a = None
    for t in range(WINDOW, t_total):
        segment = x[t - WINDOW:t]
        mu = segment.mean(axis=0)
        sigma = segment.std(axis=0) + 1e-8

        z = np.abs((x[t] - mu) / sigma)
        baseline_scores[t] = float(z.mean())

        a_t = fit_var1(segment)
        spectral_radius_scores[t] = float(np.max(np.abs(eigvals(a_t))))

        if prev_a is not None:
            drift_arr[t] = float(np.linalg.norm(a_t - prev_a, ord="fro"))
        prev_a = a_t

    drift_ema = np.zeros(t_total)
    if t_total > 0:
        drift_ema[0] = drift_arr[0]
    for t in range(1, t_total):
        drift_ema[t] = EMA_ALPHA * drift_arr[t] + (1 - EMA_ALPHA) * drift_ema[t - 1]

    cumulative_drift = np.zeros(t_total)
    for t in range(t_total):
        start = max(0, t - CUMULATIVE_N + 1)
        cumulative_drift[t] = drift_arr[start:t + 1].sum()

    drift_trend = np.zeros(t_total)
    for t in range(t_total):
        start = max(0, t - CUMULATIVE_N + 1)
        segment = drift_arr[start:t + 1]
        if len(segment) >= 2:
            xx = np.arange(len(segment))
            slope, _ = np.polyfit(xx, segment, 1)
            drift_trend[t] = float(slope)

    normalized_signals = ordered.copy()
    for col in SIGNAL_COLUMNS:
        normalized_signals[col] = normalize_series(normalized_signals[col].to_numpy(dtype=float))

    return {
        "signals": normalized_signals,
        "baseline_scores": baseline_scores,
        "drift_arr": drift_arr,
        "drift_ema": drift_ema,
        "cumulative_drift": cumulative_drift,
        "drift_trend": drift_trend,
        "spectral_radius_scores": spectral_radius_scores,
    }


def analyze_unit(
    signals: pd.DataFrame,
    threshold_std: float,
    persistence: int,
    threshold_mode: str,
    threshold_percentile: float = 97.5,
) -> tuple[dict, dict]:
    unit_id = int(signals["unit"].iloc[0])
    metrics = compute_unit_timeseries(signals)

    t_total = len(metrics["signals"])
    if t_total < WINDOW + 2:
        result = {
            "unit": unit_id,
            "cycles": t_total,
            "baseline_peak": np.nan,
            "operator_drift_peak": np.nan,
            "spectral_radius_peak": np.nan,
            "degradation_start": np.nan,
            "ema_warning": np.nan,
            "lead_vs_degradation": np.nan,
            "lead_vs_baseline": np.nan,
            "false_positive": False,
            "warning_before_degradation": False,
            "warning_before_baseline_peak": False,
            "max_ema_before_degradation": np.nan,
        }
        metrics["ema_threshold"] = np.nan
        return result, metrics

    baseline_peak_idx = int(np.argmax(metrics["baseline_scores"]))
    operator_drift_peak_idx = int(np.argmax(metrics["drift_arr"]))
    spectral_peak_idx = int(np.argmax(metrics["spectral_radius_scores"]))

    degradation_start = int(DEGRADATION_FRACTION * t_total)

    healthy_end = max(1, int(HEALTHY_FRACTION * t_total))
    healthy_reference = metrics["drift_ema"][:healthy_end]
    threshold = compute_healthy_threshold(
        healthy_reference,
        threshold_std,
        threshold_mode,
        threshold_percentile=threshold_percentile,
    )
    ema_warning = first_persistent_warning(metrics["drift_ema"], threshold, persistence)

    if ema_warning is None:
        lead_vs_degradation = np.nan
        lead_vs_baseline = np.nan
        false_positive = False
        warning_before_degradation = False
        warning_before_baseline_peak = False
        max_ema_before_deg = float(np.max(metrics["drift_ema"][:degradation_start])) if degradation_start > 0 else np.nan
    else:
        lead_vs_degradation = float(degradation_start - ema_warning)
        lead_vs_baseline = float(baseline_peak_idx - ema_warning)
        false_positive = bool(ema_warning < healthy_end)
        warning_before_degradation = bool(ema_warning < degradation_start)
        warning_before_baseline_peak = bool(ema_warning < baseline_peak_idx)
        max_ema_before_deg = float(np.max(metrics["drift_ema"][:degradation_start])) if degradation_start > 0 else np.nan

    result = {
        "unit": unit_id,
        "cycles": t_total,
        "baseline_peak": baseline_peak_idx,
        "operator_drift_peak": operator_drift_peak_idx,
        "spectral_radius_peak": spectral_peak_idx,
        "degradation_start": degradation_start,
        "ema_warning": ema_warning,
        "lead_vs_degradation": lead_vs_degradation,
        "lead_vs_baseline": lead_vs_baseline,
        "false_positive": false_positive,
        "warning_before_degradation": warning_before_degradation,
        "warning_before_baseline_peak": warning_before_baseline_peak,
        "max_ema_before_degradation": max_ema_before_deg,
    }
    metrics["ema_threshold"] = threshold
    return result, metrics


def run_multi_unit_experiment(
    threshold_std: float,
    persistence: int,
    threshold_mode: str,
    threshold_percentile: float = 97.5,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    df = load_fd004(DATA_PATH)

    units = sorted(df["unit"].unique())[:NUM_UNITS]
    results = []
    unit_data: dict[int, dict] = {}

    for unit in units:
        unit_df = df[df["unit"] == unit].copy()
        result, data = analyze_unit(
            unit_df,
            threshold_std,
            persistence,
            threshold_mode,
            threshold_percentile=threshold_percentile,
        )
        results.append(result)
        unit_data[int(unit)] = data

    return pd.DataFrame(results), unit_data


def print_summary(results_df: pd.DataFrame, preset_name: str) -> None:
    valid_deg = results_df["lead_vs_degradation"].dropna()
    valid_base = results_df["lead_vs_baseline"].dropna()

    false_positive_count = int(results_df["false_positive"].sum())
    units_with_warning = int(results_df["ema_warning"].notna().sum())
    warnings_before_degradation = int(results_df["warning_before_degradation"].sum())
    warnings_before_baseline_peak = int(results_df["warning_before_baseline_peak"].sum())

    print(f"\n=== AGGREGATE SUMMARY ({preset_name}) ===")
    print("units analyzed:", len(results_df))
    print("units with ema_warning:", units_with_warning)
    print("units where ema_warning < degradation_start:", warnings_before_degradation)
    print("units where ema_warning < baseline_peak:", warnings_before_baseline_peak)
    print("false positives (warning in healthy region):", false_positive_count)
    print("false positive rate:", false_positive_count / len(results_df) if len(results_df) else np.nan)

    if not valid_deg.empty:
        print("mean lead_vs_degradation:", valid_deg.mean())
        print("median lead_vs_degradation:", valid_deg.median())
        print("min lead_vs_degradation:", valid_deg.min())
        print("max lead_vs_degradation:", valid_deg.max())

    if not valid_base.empty:
        print("mean lead_vs_baseline:", valid_base.mean())
        print("median lead_vs_baseline:", valid_base.median())
        print("min lead_vs_baseline:", valid_base.min())
        print("max lead_vs_baseline:", valid_base.max())


def run_experiment_with_params(
    threshold_std: float,
    persistence: int,
    threshold_mode: str = "mean_std",
    threshold_percentile: float = 97.5,
) -> dict:
    results_df, _ = run_multi_unit_experiment(
        threshold_std=threshold_std,
        persistence=persistence,
        threshold_mode=threshold_mode,
        threshold_percentile=threshold_percentile,
    )

    false_positive_rate = float(results_df["false_positive"].mean())
    coverage = float(results_df["ema_warning"].notna().mean())

    valid_deg = results_df["lead_vs_degradation"].dropna()
    mean_lead = float(valid_deg.mean()) if not valid_deg.empty else np.nan

    return {
        "threshold": threshold_std,
        "persistence": persistence,
        "threshold_percentile": threshold_percentile,
        "false_positive_rate": false_positive_rate,
        "coverage": coverage,
        "mean_lead": mean_lead,
    }


def grid_search() -> pd.DataFrame:
    thresholds = [1.5, 2.0, 2.5, 3.0]
    persistences = [3, 5, 8]

    rows = []
    for threshold in thresholds:
        for persistence in persistences:
            result = run_experiment_with_params(threshold, persistence)
            rows.append(result)
            print(
                f"threshold={threshold}, persistence={persistence} -> "
                f"FP={result['false_positive_rate']:.2f}, "
                f"coverage={result['coverage']:.2f}, "
                f"lead={result['mean_lead']:.1f}"
            )

    return pd.DataFrame(rows)


def score_row(row: pd.Series) -> float:
    return (
        row["mean_lead"] * 1.0
        + row["coverage"] * 40
        - row["false_positive_rate"] * 400
    )


def evaluate_presets() -> pd.DataFrame:
    rows = []
    for preset_name, preset_cfg in DETECTOR_PRESETS.items():
        results_df, _ = run_multi_unit_experiment(
            threshold_std=float(preset_cfg["threshold_std"]),
            persistence=int(preset_cfg["persistence"]),
            threshold_mode=str(preset_cfg["threshold_mode"]),
            threshold_percentile=float(preset_cfg.get("threshold_percentile", 97.5)),
        )
        false_positive_rate = float(results_df["false_positive"].mean())
        coverage = float(results_df["ema_warning"].notna().mean())
        valid_deg = results_df["lead_vs_degradation"].dropna()
        mean_lead = float(valid_deg.mean()) if not valid_deg.empty else np.nan
        rows.append({
            "preset": preset_name,
            "threshold_std": float(preset_cfg["threshold_std"]),
            "persistence": int(preset_cfg["persistence"]),
            "threshold_mode": str(preset_cfg["threshold_mode"]),
            "threshold_percentile": float(preset_cfg.get("threshold_percentile", 97.5)),
            "false_positive_rate": false_positive_rate,
            "coverage": coverage,
            "mean_lead": mean_lead,
        })

    df = pd.DataFrame(rows)
    df["score"] = df.apply(score_row, axis=1)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def print_preset_comparison_summary(preset_df: pd.DataFrame) -> None:
    if preset_df.empty:
        return
    safest = preset_df.sort_values(["false_positive_rate", "coverage"], ascending=[True, False]).iloc[0]
    most_aggressive = preset_df.sort_values(["mean_lead", "false_positive_rate"], ascending=[False, False]).iloc[0]

    print("\n=== DETECTOR CALIBRATION SUMMARY ===")
    print(preset_df[[
        "preset",
        "threshold_std",
        "persistence",
        "threshold_mode",
        "threshold_percentile",
        "false_positive_rate",
        "coverage",
        "mean_lead",
        "score",
    ]].to_string(index=False))
    print(f"\nSafest preset (lowest FPR): {safest['preset']}")
    print(f"Most aggressive preset (max lead tendency): {most_aggressive['preset']}")


def plot_unit_result(unit_id: int, data: dict, result: pd.Series) -> plt.Figure:
    signals = data["signals"]
    drift_ema = data["drift_ema"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for col in SIGNAL_COLUMNS:
        ax1.plot(signals[col].values, label=col, alpha=0.85)
    ax1.set_ylabel("Normalized value")
    ax1.set_title(f"Unit {unit_id} — Normalized signals")
    ax1.legend(loc="upper left", fontsize=8)

    ax2.plot(data["drift_arr"], color="steelblue", alpha=0.6, label="Raw drift")
    ax2.plot(drift_ema, color="darkorange", label="Anomaly score (drift EMA)")

    threshold = data.get("ema_threshold", np.nan)
    if pd.notna(threshold):
        ax2.axhline(float(threshold), color="gray", linestyle=":", linewidth=1.2, label="EMA threshold")

    ax2.set_ylabel("Score")
    ax2.set_xlabel("Timestep")

    degradation_start = result.get("degradation_start", np.nan)
    ema_warning = result.get("ema_warning", np.nan)

    for ax in (ax1, ax2):
        if pd.notna(degradation_start):
            ax.axvline(int(degradation_start), color="red", linestyle="--", linewidth=1.4, label="Degradation start")
        if pd.notna(ema_warning):
            ax.axvline(int(ema_warning), color="green", linestyle="--", linewidth=1.4, label="EMA warning")

    for ax in (ax1, ax2):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=8)

    lead = result.get("lead_vs_degradation", np.nan)
    lead_str = f"{int(lead)}" if pd.notna(lead) else "N/A"
    fp_suffix = " [FALSE POSITIVE]" if bool(result.get("false_positive", False)) else ""
    fig.suptitle(f"Unit {unit_id} — lead_vs_degradation={lead_str}{fp_suffix}", fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


def run_grid_search_workflow() -> None:
    df = grid_search()
    df["score"] = df.apply(score_row, axis=1)
    ranked = df.sort_values("score", ascending=False).reset_index(drop=True)

    print("\n=== BEST CONFIG ===")
    print(ranked.iloc[0])

    print("\n=== RANKED CONFIGS ===")
    print(ranked)

    ranked.to_csv("fd004_grid_search_results.csv", index=False)
    print("\nSaved grid search results to fd004_grid_search_results.csv")


def main() -> None:
    default_cfg = DETECTOR_PRESETS[DEFAULT_PRESET]
    print(
        "Running default preset:",
        DEFAULT_PRESET,
        "| threshold_std=",
        default_cfg["threshold_std"],
        "| persistence=",
        default_cfg["persistence"],
        "| threshold_mode=",
        default_cfg["threshold_mode"],
        "| threshold_percentile=",
        default_cfg.get("threshold_percentile", 97.5),
    )
    results_df, unit_data = run_multi_unit_experiment(
        threshold_std=float(default_cfg["threshold_std"]),
        persistence=int(default_cfg["persistence"]),
        threshold_mode=str(default_cfg["threshold_mode"]),
        threshold_percentile=float(default_cfg.get("threshold_percentile", 97.5)),
    )

    results_df.to_csv("fd004_final_results.csv", index=False)
    print("Saved results to fd004_final_results.csv")

    print_summary(results_df, preset_name=DEFAULT_PRESET)

    preset_df = evaluate_presets()
    preset_df.to_csv("fd004_preset_comparison.csv", index=False)
    print("Saved preset comparison to fd004_preset_comparison.csv")
    print_preset_comparison_summary(preset_df)

    plots_dir = Path("./plots")
    plots_dir.mkdir(exist_ok=True)

    detected = results_df[
        results_df["ema_warning"].notna() &
        (results_df["ema_warning"] < results_df["degradation_start"])
    ].copy()

    false_positives = results_df[results_df["false_positive"]].copy()

    best_units = detected.nlargest(3, "lead_vs_degradation")["unit"].tolist() if not detected.empty else []
    worst_units = detected.nsmallest(3, "lead_vs_degradation")["unit"].tolist() if not detected.empty else []
    fp_units = false_positives["unit"].tolist()[:2]

    units_to_plot = list(dict.fromkeys(best_units + worst_units + fp_units))

    if ENABLE_PLOTS:
        print(f"\nPlotting units: best={best_units}, worst={worst_units}, FP={fp_units}")
        for uid in units_to_plot:
            uid = int(uid)
            if uid not in unit_data:
                continue
            row = results_df[results_df["unit"] == uid].iloc[0]
            fig = plot_unit_result(uid, unit_data[uid], row)
            out_path = plots_dir / f"unit_{uid}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_path}")

    print(f"\nDone. {len(units_to_plot) if ENABLE_PLOTS else 0} plots saved to {plots_dir}/")


if __name__ == "__main__":
    if RUN_GRID_SEARCH:
        run_grid_search_workflow()
    else:
        main()
