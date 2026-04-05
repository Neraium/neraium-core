from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, eigvals


DATA_PATH = Path(__file__).resolve().parent / "train_FD004.txt"

SIGNAL_COLUMNS = ["s_2", "s_3", "s_4", "s_7"]

WINDOW = 20
DEGRADATION_FRACTION = 0.8

EMA_ALPHA = 0.2
CUMULATIVE_N = 30
NUM_UNITS = 249

# false-positive / stability controls
HEALTHY_FRACTION = 0.2
PERSISTENCE = 3
THRESHOLD_STD = 1.5

RUN_GRID_SEARCH = False
ENABLE_PLOTS = True


def load_fd004(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None)

    # CMAPSS format: unit, cycle, 3 operational settings, 21 sensors
    base_cols = ["unit", "cycle", "op_1", "op_2", "op_3"]
    sensor_cols = [f"s_{i}" for i in range(1, 22)]
    df.columns = base_cols + sensor_cols

    return df


def fit_var1(x: np.ndarray) -> np.ndarray:
    """
    Fit x[t+1] = x[t] @ A
    x shape: (T, D)
    returns A shape: (D, D)
    """
    x_t = x[:-1]
    x_next = x[1:]
    a, _, _, _ = lstsq(x_t, x_next, rcond=None)
    return a


def normalize_for_plot(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.allclose(x.max(), x.min()):
        return np.zeros_like(x)
    x_min = x.min()
    x_max = x.max()
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


def first_persistent_warning(signal: np.ndarray, threshold: float, persistence: int) -> int | None:
    run = 0
    for i, value in enumerate(signal):
        if value > threshold:
            run += 1
            if run >= persistence:
                return i
        else:
            run = 0
    return None


def analyze_unit(signals: pd.DataFrame) -> tuple[dict, dict]:
    signals = signals.sort_values("cycle").reset_index(drop=True)

    x = signals[SIGNAL_COLUMNS].to_numpy(dtype=float)
    t_total = len(x)

    empty_plot_data = {
        "x": x,
        "baseline_scores": np.zeros(t_total),
        "drift_arr": np.zeros(t_total),
        "drift_ema": np.zeros(t_total),
        "spectral_radius_scores": np.zeros(t_total),
        "cumulative_drift": np.zeros(t_total),
        "drift_trend": np.zeros(t_total),
        "t_total": t_total,
    }

    if t_total < WINDOW + 2:
        result = {
            "unit": int(signals["unit"].iloc[0]),
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
        }
        return result, empty_plot_data

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

    # EMA of drift
    drift_ema = np.zeros(t_total)
    drift_ema[0] = drift_arr[0]
    for t in range(1, t_total):
        drift_ema[t] = EMA_ALPHA * drift_arr[t] + (1 - EMA_ALPHA) * drift_ema[t - 1]

    # cumulative drift
    cumulative_drift = np.zeros(t_total)
    for t in range(t_total):
        start = max(0, t - CUMULATIVE_N + 1)
        cumulative_drift[t] = drift_arr[start:t + 1].sum()

    # drift trend
    drift_trend = np.zeros(t_total)
    for t in range(t_total):
        start = max(0, t - CUMULATIVE_N + 1)
        segment = drift_arr[start:t + 1]
        if len(segment) >= 2:
            xx = np.arange(len(segment))
            slope, _ = np.polyfit(xx, segment, 1)
            drift_trend[t] = float(slope)

    baseline_peak_idx = int(np.argmax(baseline_scores))
    operator_drift_peak_idx = int(np.argmax(drift_arr))
    spectral_peak_idx = int(np.argmax(spectral_radius_scores))

    degradation_start = int(DEGRADATION_FRACTION * t_total)

    healthy_end = max(1, int(HEALTHY_FRACTION * t_total))
    healthy_reference = drift_ema[:healthy_end]
    threshold = float(healthy_reference.mean() + THRESHOLD_STD * healthy_reference.std())

    ema_warning = first_persistent_warning(drift_ema, threshold, PERSISTENCE)

    if ema_warning is None:
        lead_vs_degradation = np.nan
        lead_vs_baseline = np.nan
        false_positive = False
        warning_before_degradation = False
        warning_before_baseline_peak = False
    else:
        lead_vs_degradation = float(degradation_start - ema_warning)
        lead_vs_baseline = float(baseline_peak_idx - ema_warning)
        false_positive = bool(ema_warning < healthy_end)
        warning_before_degradation = bool(ema_warning < degradation_start)
        warning_before_baseline_peak = bool(ema_warning < baseline_peak_idx)

    result = {
        "unit": int(signals["unit"].iloc[0]),
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
    }

    plot_data = {
        "x": x,
        "baseline_scores": baseline_scores,
        "drift_arr": drift_arr,
        "drift_ema": drift_ema,
        "spectral_radius_scores": spectral_radius_scores,
        "cumulative_drift": cumulative_drift,
        "drift_trend": drift_trend,
        "t_total": t_total,
    }

    return result, plot_data


def plot_unit_result(unit_id: int, data: dict, result: dict) -> plt.Figure:
    t_total = data["t_total"]
    baseline_scores = data["baseline_scores"]
    drift_arr = data["drift_arr"]
    drift_ema = data["drift_ema"]
    spectral_radius_scores = data["spectral_radius_scores"]
    cumulative_drift = data["cumulative_drift"]
    drift_trend = data["drift_trend"]
    x = data["x"]

    cumulative_plot = normalize_for_plot(cumulative_drift)
    trend_plot = normalize_for_plot(drift_trend)

    # normalized signals
    sig_normalized = np.apply_along_axis(normalize_for_plot, 0, x)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # top panel: normalized signals
    ax1 = axes[0]
    for i, col in enumerate(SIGNAL_COLUMNS):
        ax1.plot(sig_normalized[:, i], label=col, alpha=0.7)
    ax1.set_ylabel("Normalized signal")
    ax1.set_title(
        f"FD004 Unit {unit_id}"
        + (" [FALSE POSITIVE]" if result.get("false_positive") else "")
    )
    ax1.legend(loc="upper left", fontsize=8)

    # bottom panel: anomaly score / drift EMA + other signals
    ax2 = axes[1]
    ax2.plot(baseline_scores, label="Baseline (mean abs z-score)", alpha=0.7)
    ax2.plot(drift_ema, label=f"Drift EMA (alpha={EMA_ALPHA})", linewidth=1.5)
    ax2.plot(drift_arr, label="Raw drift ||A_t - A_(t-1)||", alpha=0.5)
    ax2.plot(spectral_radius_scores, label="Spectral radius", alpha=0.6)
    ax2.plot(trend_plot, label=f"Drift trend (N={CUMULATIVE_N}, norm)", alpha=0.5)
    ax2.plot(cumulative_plot, label=f"Cumulative drift (N={CUMULATIVE_N}, norm)", alpha=0.5)
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Timestep")

    if pd.notna(result.get("degradation_start")):
        ax1.axvline(int(result["degradation_start"]), linestyle="--", color="red", alpha=0.7, label="Degradation start")
        ax2.axvline(int(result["degradation_start"]), linestyle="--", color="red", label="Degradation start (proxy)")

    if pd.notna(result.get("ema_warning")) and result["ema_warning"] is not None:
        ax1.axvline(int(result["ema_warning"]), linestyle="--", color="orange", alpha=0.7, label=f"EMA warning (t={int(result['ema_warning'])})")
        ax2.axvline(int(result["ema_warning"]), linestyle="--", color="orange", label=f"EMA warning (t={int(result['ema_warning'])})")

    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig


def run_multi_unit_experiment(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    df = load_fd004(DATA_PATH)

    units = sorted(df["unit"].unique())[:NUM_UNITS]
    results = []
    unit_data = {}

    if verbose:
        print("Loaded shape:", df.shape)
        print(f"Running experiment on {len(units)} units")
        print("Signals:", SIGNAL_COLUMNS)

    for unit in units:
        unit_df = df[df["unit"] == unit].copy()
        result, plot_data = analyze_unit(unit_df)
        results.append(result)
        unit_data[unit] = plot_data

    return pd.DataFrame(results), unit_data


def print_summary(results_df: pd.DataFrame) -> None:
    valid_deg = results_df["lead_vs_degradation"].dropna()
    valid_base = results_df["lead_vs_baseline"].dropna()

    false_positive_count = int(results_df["false_positive"].sum())
    units_with_warning = int(results_df["ema_warning"].notna().sum())
    warnings_before_degradation = int(results_df["warning_before_degradation"].sum())
    warnings_before_baseline_peak = int(results_df["warning_before_baseline_peak"].sum())

    print("\n=== AGGREGATE SUMMARY ===")
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


def run_experiment_with_params(threshold_std: float, persistence: int) -> dict:
    global THRESHOLD_STD, PERSISTENCE

    old_threshold = THRESHOLD_STD
    old_persistence = PERSISTENCE

    THRESHOLD_STD = threshold_std
    PERSISTENCE = persistence

    try:
        results_df, _ = run_multi_unit_experiment(verbose=False)

        false_positive_rate = float(results_df["false_positive"].mean())
        coverage = float(results_df["ema_warning"].notna().mean())

        valid_deg = results_df["lead_vs_degradation"].dropna()
        mean_lead = float(valid_deg.mean()) if not valid_deg.empty else np.nan

        return {
            "threshold": threshold_std,
            "persistence": persistence,
            "false_positive_rate": false_positive_rate,
            "coverage": coverage,
            "mean_lead": mean_lead,
        }
    finally:
        THRESHOLD_STD = old_threshold
        PERSISTENCE = old_persistence


def grid_search() -> pd.DataFrame:
    thresholds = [1.5, 2.0, 2.5, 3.0]
    persistences = [3, 5, 8]

    rows = []

    print("Loaded shape:", load_fd004(DATA_PATH).shape)
    print(f"Running experiment on {NUM_UNITS} units")
    print("Signals:", SIGNAL_COLUMNS)

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
        + row["coverage"] * 50
        - row["false_positive_rate"] * 200
    )


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
    results_df, unit_data = run_multi_unit_experiment(verbose=True)
    print_summary(results_df)
    results_df.to_csv("fd004_final_results.csv", index=False)
    print("\nSaved results to fd004_final_results.csv")

    if not ENABLE_PLOTS or results_df.empty:
        return

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # valid rows with non-nan lead_vs_degradation
    valid = results_df.dropna(subset=["lead_vs_degradation"]).copy()

    best_units: list[int] = []
    worst_units: list[int] = []
    fp_units: list[int] = []

    if not valid.empty:
        sorted_by_lead = valid.sort_values("lead_vs_degradation", ascending=False)
        best_units = list(sorted_by_lead["unit"].head(3))
        worst_units = list(sorted_by_lead["unit"].tail(3))

    fp_rows = results_df[results_df["false_positive"] == True]
    fp_units = list(fp_rows["unit"].head(2))

    plot_set: list[tuple[int, str]] = []
    for uid in best_units:
        plot_set.append((uid, "best"))
    for uid in worst_units:
        if uid not in best_units:
            plot_set.append((uid, "worst"))
    for uid in fp_units:
        if uid not in best_units and uid not in worst_units:
            plot_set.append((uid, "fp"))

    for uid, tag in plot_set:
        if uid not in unit_data:
            continue
        row = results_df[results_df["unit"] == uid]
        if row.empty:
            continue
        result = row.iloc[0].to_dict()
        fig = plot_unit_result(uid, unit_data[uid], result)
        out_path = plots_dir / f"unit_{uid:03d}_{tag}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    if RUN_GRID_SEARCH:
        run_grid_search_workflow()
    else:
        main()
