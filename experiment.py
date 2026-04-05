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

HEALTHY_FRACTION = 0.2
THRESHOLD_STD = 1.5
PERSISTENCE = 3
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
                return i - persistence + 1
        else:
            run = 0
    return None


def analyze_unit(signals: pd.DataFrame) -> dict:
    signals = signals.sort_values("cycle").reset_index(drop=True)

    x = signals[SIGNAL_COLUMNS].to_numpy(dtype=float)
    t_total = len(x)

    if t_total < WINDOW + 2:
        return {
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

    return {
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


def plot_unit(signals: pd.DataFrame, result: dict) -> None:
    signals = signals.sort_values("cycle").reset_index(drop=True)

    x = signals[SIGNAL_COLUMNS].to_numpy(dtype=float)
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

    cumulative_plot = normalize_for_plot(cumulative_drift)
    trend_plot = normalize_for_plot(drift_trend)

    plt.figure(figsize=(12, 7))
    plt.plot(baseline_scores, label="Baseline (mean abs z-score)")
    plt.plot(trend_plot, label=f"Drift trend/slope (N={CUMULATIVE_N}, normalized)")
    plt.plot(spectral_radius_scores, label="Spectral radius")
    plt.plot(drift_arr, label="Raw drift ||A_t - A_(t-1)||")
    plt.plot(drift_ema, label=f"Drift EMA (alpha={EMA_ALPHA})")
    plt.plot(cumulative_plot, label=f"Cumulative drift (N={CUMULATIVE_N}, normalized)")

    if pd.notna(result["degradation_start"]):
        plt.axvline(
            int(result["degradation_start"]),
            linestyle="--",
            color="red",
            label="Degradation start (proxy)",
        )

    if pd.notna(result["ema_warning"]):
        plt.axvline(
            int(result["ema_warning"]),
            linestyle="--",
            color="orange",
            label=f"EMA warning (t={int(result['ema_warning'])})",
        )

    unit_id = int(signals["unit"].iloc[0])
    plt.title(f"FD004 Unit {unit_id}: sustained structural drift detection")
    plt.xlabel("Timestep")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_multi_unit_experiment() -> tuple[pd.DataFrame, dict]:
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
        result = analyze_unit(unit_df)
        results.append(result)

        unit_data[int(unit)] = {
            "signals": signals,
            "drift_ema": drift_ema,
            "ema_threshold": ema_threshold,
        }

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
        results_df, _ = run_multi_unit_experiment()
    finally:
        THRESHOLD_STD = prev_threshold
        PERSISTENCE = prev_persistence

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


def plot_unit_result(unit_id: int, data: dict, result: pd.Series) -> plt.Figure:
    signals = data["signals"]
    drift_ema = data["drift_ema"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top panel: normalized signals
    for col in SIGNAL_COLUMNS:
        ax1.plot(signals[col].values, label=col, alpha=0.8)
    ax1.set_ylabel("Normalized Value")
    ax1.set_title(f"Unit {unit_id} — Signals and Anomaly Score")
    ax1.legend(loc="upper left", fontsize=8)

    # Bottom panel: anomaly score (drift EMA)
    ax2.plot(drift_ema, color="darkorange", label="Anomaly Score (drift EMA)")
    ax2.set_ylabel("Drift EMA")
    ax2.set_xlabel("Timestep")
    ax2.legend(loc="upper left", fontsize=8)

    # Vertical lines on both panels
    degradation_start = result["degradation_start"]
    ema_warning = result["ema_warning"]

    for ax in (ax1, ax2):
        ax.axvline(degradation_start, color="red", linestyle="--", linewidth=1.5,
                   label=f"Degradation start (t={degradation_start})")
        if ema_warning is not None and not pd.isna(ema_warning):
            ema_warning = int(ema_warning)
            ax.axvline(ema_warning, color="green", linestyle="--", linewidth=1.5,
                       label=f"EMA warning (t={ema_warning})")

    # Rebuild legends to include vlines
    for ax in (ax1, ax2):
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left", fontsize=8)

    lead = result["lead_vs_degradation"]
    lead_str = f"{int(lead)} steps" if lead is not None and not pd.isna(lead) else "N/A"
    fp_flag = ""
    if ema_warning is not None and not pd.isna(result["ema_warning"]):
        if int(result["ema_warning"]) >= degradation_start:
            fp_flag = " [FALSE POSITIVE]"
    fig.suptitle(
        f"Unit {unit_id} — Lead: {lead_str}{fp_flag}",
        fontsize=11, fontweight="bold"
    )

    plt.tight_layout()
    return fig


def main():
    results_df, unit_data = run_multi_unit_experiment()

    # Save results
    results_df.to_csv("fd004_final_results.csv", index=False)
    print("Saved results to fd004_final_results.csv")

    # Aggregate summary
    fp_count = int((results_df["ema_warning"] >= results_df["degradation_start"]).sum())
    coverage = results_df["ema_warning"].notna().mean()
    mean_lead = results_df["lead_vs_degradation"].dropna().mean()

    print("\n=== AGGREGATE SUMMARY ===")
    print(f"False positives:  {fp_count} / {len(results_df)}")
    print(f"Coverage:         {coverage:.3f}")
    print(f"Mean lead:        {mean_lead:.2f} steps")

    # Prepare plots folder
    plots_dir = Path("./plots")
    plots_dir.mkdir(exist_ok=True)

    # Select units to plot
    detected = results_df[
        results_df["ema_warning"].notna() &
        (results_df["ema_warning"] < results_df["degradation_start"])
    ].copy()

    false_positives = results_df[
        results_df["ema_warning"].notna() &
        (results_df["ema_warning"] >= results_df["degradation_start"])
    ].copy()

    best_units = (
        detected.nlargest(3, "lead_vs_degradation")["unit"].tolist()
        if not detected.empty else []
    )
    worst_units = (
        detected.nsmallest(3, "lead_vs_degradation")["unit"].tolist()
        if not detected.empty else []
    )
    fp_units = false_positives["unit"].tolist()[:2]

    units_to_plot = list(dict.fromkeys(best_units + worst_units + fp_units))

    print(f"\nPlotting units: best={best_units}, worst={worst_units}, FP={fp_units}")

    for uid in units_to_plot:
        if uid not in unit_data:
            continue
        row = results_df[results_df["unit"] == uid].iloc[0]
        fig = plot_unit_result(uid, unit_data[uid], row)
        out_path = plots_dir / f"unit_{uid}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    print(f"\nDone. {len(units_to_plot)} plots saved to {plots_dir}/")


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
    results_df = run_multi_unit_experiment(verbose=True)
    print_summary(results_df)
    results_df.to_csv("fd004_multi_unit_results.csv", index=False)
    print("\nSaved results to fd004_multi_unit_results.csv")

    if ENABLE_PLOTS and not results_df.empty:
        df = load_fd004(DATA_PATH)
        first_unit = int(results_df["unit"].iloc[0])
        unit_df = df[df["unit"] == first_unit].copy()
        unit_result = results_df[results_df["unit"] == first_unit].iloc[0].to_dict()
        plot_unit(unit_df, unit_result)


if __name__ == "__main__":
    if RUN_GRID_SEARCH:
        run_grid_search_workflow()
    else:
        main()
