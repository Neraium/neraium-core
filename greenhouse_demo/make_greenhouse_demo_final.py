import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "greenhouse_results_turbo.csv"
OUT_PATH = "neraium_system_evolution_example.png"

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

# scale stability so it is visible
df["stability_scaled"] = df["relational_stability_score"] * 80.0

# strongest transition point
diff = df["drift_smooth"].diff().abs().fillna(0.0)
transition_idx = int(diff.idxmax())
transition_time = float(df.loc[transition_idx, "timestamp"])
transition_y = float(df.loc[transition_idx, "drift_smooth"])

# reduce alert clutter
if "demo_alert" in df.columns:
    alerts = df[df["demo_alert"] == True].iloc[::8].copy()
else:
    alerts = df.iloc[0:0].copy()

fig, ax = plt.subplots(figsize=(14, 7))

# raw drift
ax.plot(
    df["timestamp"],
    df["structural_drift_score"],
    alpha=0.18,
    linewidth=1.0,
    label="Raw Drift",
)

# smoothed drift
ax.plot(
    df["timestamp"],
    df["drift_smooth"],
    linewidth=2.8,
    label="Smoothed Drift",
)

# scaled stability
ax.plot(
    df["timestamp"],
    df["stability_scaled"],
    linewidth=2.0,
    label="Stability (scaled)",
)

# sparse alerts
if len(alerts):
    ax.scatter(
        alerts["timestamp"],
        alerts["structural_drift_score"],
        s=28,
        label="Alerts",
        zorder=5,
    )

# tight transition zone
ax.axvspan(
    transition_time - 2,
    transition_time + 2,
    alpha=0.25,
)

# headline label
headline_x = transition_time + 3
headline_y = float(df["structural_drift_score"].max()) * 0.97

ax.text(
    headline_x,
    headline_y,
    "SYSTEM REORGANIZATION BEGINS",
    ha="left",
    va="bottom",
    fontsize=11,
    fontweight="bold",
)

# arrow pointing to the actual shift
ax.annotate(
    "",
    xy=(transition_time, transition_y),
    xytext=(headline_x, headline_y - 1.0),
    arrowprops=dict(arrowstyle="->", linewidth=1.5),
)

# supporting callout
ax.text(
    headline_x,
    headline_y - 2.2,
    "Structural drift rises before any obvious failure event",
    ha="left",
    va="top",
    fontsize=9,
)

ax.set_title("System Evolution (Greenhouse)", fontsize=18)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Structural Change", fontsize=12)
ax.legend()

# extra left breathing room
plt.xlim(df["timestamp"].min() - 5, df["timestamp"].max())

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"saved {OUT_PATH}")
