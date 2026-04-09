import time
from pathlib import Path

import pandas as pd

from neraium_core.alignment import StructuralEngine

DATA_DIR = Path(r"C:\Users\Owner\Desktop\CMAPSSData")
DATASETS = ["FD001", "FD002", "FD003", "FD004"]

def load_train_dataset(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"train_{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1).copy()

    n_cols = df.shape[1]
    expected_sensor_count = n_cols - 5  # unit, cycle, os1, os2, os3
    df.columns = (
        ["unit", "cycle", "os1", "os2", "os3"]
        + [f"s{i}" for i in range(1, expected_sensor_count + 1)]
    )
    return df

def run_dataset(name: str) -> dict:
    print(f"\n{'='*70}")
    print(f"RUNNING {name}")
    print(f"{'='*70}")

    df = load_train_dataset(name)
    units = sorted(df["unit"].unique())
    total_units = len(units)
    total_rows = len(df)

    print(f"rows loaded: {total_rows}")
    print(f"units loaded: {total_units}")

    engine = StructuralEngine(
        baseline_window=24,
        recent_window=8,
        drift_smoothing_window=25,
        watch_quantile=0.65,
        alert_quantile=0.85,
        watch_persistence=5,
        alert_persistence=3,
        fast_trigger_multiplier=1.25,
        alert_latch_enabled=True,
        unlatch_ratio=0.75,
    )

    results = []
    t0 = time.time()

    for idx, unit in enumerate(units, start=1):
        unit_df = df[df["unit"] == unit].sort_values("cycle")

        first_alert_cycle = None
        first_watch_cycle = None
        last_cycle = int(unit_df["cycle"].max())

        for row in unit_df.itertuples(index=False):
            sensor_values = {
                "os1": float(row.os1),
                "os2": float(row.os2),
                "os3": float(row.os3),
            }

            for i in range(1, len(unit_df.columns) - 4):
                key = f"s{i}"
                if hasattr(row, key):
                    sensor_values[key] = float(getattr(row, key))

            frame = {
                "timestamp": float(row.cycle),
                "site_id": name,
                "asset_id": f"{name}_unit_{int(row.unit)}",
                "sensor_values": sensor_values,
            }

            out = engine.process_frame(frame)

            policy_watch = bool(out.get("policy_watch", False))
            policy_alert = bool(out.get("policy_alert", False))
            state = out.get("policy_state", out.get("state"))

            if first_watch_cycle is None and (policy_watch or state == "WATCH"):
                first_watch_cycle = int(row.cycle)

            if first_alert_cycle is None and (policy_alert or state == "ALERT"):
                first_alert_cycle = int(row.cycle)

        results.append({
            "dataset": name,
            "unit": int(unit),
            "last_cycle": last_cycle,
            "first_watch_cycle": first_watch_cycle,
            "first_alert_cycle": first_alert_cycle,
            "has_watch": first_watch_cycle is not None,
            "has_alert": first_alert_cycle is not None,
        })

        elapsed = time.time() - t0
        rate = idx / elapsed if elapsed > 0 else 0.0
        print(
            f"[{name}] processed unit {idx}/{total_units} | "
            f"alerts so far: {sum(r['has_alert'] for r in results)} | "
            f"{rate:.2f} units/sec | elapsed {elapsed:.1f}s",
            flush=True
        )

    res = pd.DataFrame(results)

    summary = {
        "dataset": name,
        "units": int(len(res)),
        "watch_coverage": float(res["has_watch"].mean()),
        "alert_coverage": float(res["has_alert"].mean()),
        "mean_first_watch_cycle": float(res.loc[res["has_watch"], "first_watch_cycle"].mean()) if res["has_watch"].any() else None,
        "mean_first_alert_cycle": float(res.loc[res["has_alert"], "first_alert_cycle"].mean()) if res["has_alert"].any() else None,
        "misses": int((~res["has_alert"]).sum()),
    }

    out_csv = Path(f"{name}_suite_unit_summary.csv")
    res.to_csv(out_csv, index=False)

    print(f"\n{name} SUMMARY")
    print(f"watch coverage: {summary['watch_coverage']:.4f}")
    print(f"alert coverage: {summary['alert_coverage']:.4f}")
    print(f"mean first watch cycle: {summary['mean_first_watch_cycle']}")
    print(f"mean first alert cycle: {summary['mean_first_alert_cycle']}")
    print(f"misses: {summary['misses']}")
    print(f"saved -> {out_csv}")

    return summary

all_summaries = []

for ds in DATASETS:
    try:
        all_summaries.append(run_dataset(ds))
    except Exception as e:
        print(f"\nERROR on {ds}: {e}")

suite_df = pd.DataFrame(all_summaries)
suite_df.to_csv("cmapss_suite_results_batched.csv", index=False)

print(f"\n{'='*70}")
print("CMAPSS SUITE RESULTS")
print(f"{'='*70}")
if not suite_df.empty:
    print(suite_df.to_string(index=False))
print("\nsaved -> cmapss_suite_results_batched.csv")
