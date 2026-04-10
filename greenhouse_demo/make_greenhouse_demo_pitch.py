import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "greenhouse_results_turbo.csv"
OUT_PATH = "greenhouse_demo_pitch.png"

df = pd.read_csv(CSV_PATH)

required = ["timestamp", "structural_drift_score", "relational_stability_score"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# remove warmup artifact
if len(df) > 20:
    df = df[df["timestamp"] > df["timestamp"].iloc[20]].reset_index(drop=True)

# smooth drift
df["drift_smooth"] = df["structural_drift_score"].rolling(10, min_periods=1).mean()

# make stability visible
df["stability_scaled"] = df["relational_stability_score"] * 80.0

# strongest shift point
diff = df["drift_smooth"].diff().abs().fillna(0.0)
transition_idx = int(diff.idxmax())
transition_time = float(df.loc[transition_idx, "timestamp"])

# reduce alert clutter
if "demo_alert" in df.columns:
    alerts = df[df["demo_alert"] == True].iloc[::8].copy()
else:
    alerts = df.iloc[0:0].copy()

plt.figure(figsize=(13, 6))

plt.plot(
    df["timestamp"],
    df["structural_drift_score"],
    alpha=0.22,
    linewidth=1.1,
    label="Raw Drift",
)

plt.plot(
    df["timestamp"],
    df["drift_smooth"],
    linewidth=2.8,
    label="Smoothed Drift",
)

plt.plot(
    df["timestamp"],
    df["stability_scaled"],
    linewidth=1.8,
    label="Stability (scaled)",
)

if len(alerts):
    plt.scatter(
        alerts["timestamp"],
        alerts["structural_drift_score"],
        s=24,
        label="Alerts",
        zorder=5,
    )

# tight transition zone
plt.axvspan(
    transition_time - 2,
    transition_time + 2,
    alpha=0.25,
)

# strong pitch label
plt.text(
    transition_time,
    float(df["structural_drift_score"].max()) * 0.95,
    "SYSTEM REORGANIZATION BEGINS",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
)

plt.title("System Evolution (Greenhouse)", fontsize=16)
plt.xlabel("Time")
plt.ylabel("Structural Change")
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PATH, dpi=180)
print(f"saved {OUT_PATH}")
