import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "greenhouse_results_turbo.csv"
OUT_PATH = "greenhouse_demo_clean_v2.png"

df = pd.read_csv(CSV_PATH)

# Basic safety
required = ["timestamp", "structural_drift_score", "relational_stability_score"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# Drop obvious warmup section so the left-side spike doesn't dominate
if len(df) > 20:
    df = df[df["timestamp"] > df["timestamp"].iloc[20]].reset_index(drop=True)

# Smooth drift for readability
df["drift_smooth"] = df["structural_drift_score"].rolling(10, min_periods=1).mean()

# Scale stability so it is visible on the same plot
df["stability_scaled"] = df["relational_stability_score"] * 50.0

# Find strongest transition point from smoothed drift movement
diff = df["drift_smooth"].diff().abs().fillna(0.0)
transition_idx = int(diff.idxmax())
transition_time = float(df.loc[transition_idx, "timestamp"])

# Reduce alert clutter
if "demo_alert" in df.columns:
    alerts = df[df["demo_alert"] == True].iloc[::5].copy()
else:
    alerts = df.iloc[0:0].copy()

plt.figure(figsize=(13, 6))

# Main lines
plt.plot(
    df["timestamp"],
    df["structural_drift_score"],
    alpha=0.25,
    linewidth=1.2,
    label="Raw Drift",
)

plt.plot(
    df["timestamp"],
    df["drift_smooth"],
    linewidth=2.5,
    label="Smoothed Drift",
)

plt.plot(
    df["timestamp"],
    df["stability_scaled"],
    linewidth=1.8,
    label="Stability (scaled)",
)

# Sparse alert markers
if len(alerts):
    plt.scatter(
        alerts["timestamp"],
        alerts["structural_drift_score"],
        s=22,
        label="Alerts",
        zorder=5,
    )

# Highlight transition as a zone, not just a thin line
plt.axvspan(
    transition_time - 5,
    transition_time + 5,
    alpha=0.15,
)

plt.text(
    transition_time,
    float(df["structural_drift_score"].max()) * 0.92,
    "STRUCTURAL SHIFT BEGINS",
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

plt.savefig(OUT_PATH, dpi=170)
print(f"saved {OUT_PATH}")
