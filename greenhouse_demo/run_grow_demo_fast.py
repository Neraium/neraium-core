import pandas as pd
from tqdm import tqdm
from neraium_core.alignment import StructuralEngine

CSV_PATH = "Greenhouse_climate.csv"

# demo-speed settings
EVERY_N = 20        # keep every 20th row; use 10 if you want more detail
BATCH_SIZE = 100    # more frequent progress updates

df = pd.read_csv(CSV_PATH)

numeric_cols = []
for col in df.columns:
    if col == "GHtime":
        continue
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.notna().any():
        df[col] = converted
        numeric_cols.append(col)

if "GHtime" in df.columns:
    df["GHtime"] = pd.to_numeric(df["GHtime"], errors="coerce")
    df = df[df["GHtime"].notna()].sort_values("GHtime").reset_index(drop=True)
else:
    df["GHtime"] = range(len(df))

# stable schema + fill
df[numeric_cols] = df[numeric_cols].ffill().fillna(0.0)

# hard downsample for demo speed
df = df.iloc[::EVERY_N].reset_index(drop=True)

print(f"rows_after_downsample={len(df)}")
print(f"numeric_cols={len(numeric_cols)}")
print("first sensors:", numeric_cols[:12])

engine = StructuralEngine(
    baseline_window=30,
    recent_window=8,
)

# disable geometry update for speed during this demo run
try:
    if hasattr(engine, "geometry_layer") and engine.geometry_layer is not None:
        engine.geometry_layer.update = lambda *args, **kwargs: {}
except Exception:
    pass

results = []
alerts = 0

value_cols = ["GHtime"] + numeric_cols
work = df[value_cols]
total_rows = len(work)

pbar = tqdm(total=total_rows, desc="Processing greenhouse demo", unit="rows")

for start in range(0, total_rows, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total_rows)
    chunk = work.iloc[start:end]

    last_regime = None
    last_drift = None
    last_stability = None

    for row in chunk.itertuples(index=False, name=None):
        timestamp = float(row[0])
        sensor_values = {numeric_cols[i]: float(row[i + 1]) for i in range(len(numeric_cols))}

        frame = {
            "timestamp": timestamp,
            "site_id": "grow-house",
            "asset_id": "zone-A",
            "sensor_values": sensor_values,
        }

        result = engine.process_frame(frame)
        if result:
            drift = float(result.get("structural_drift_score") or 0.0)
            stability = float(result.get("relational_stability_score") or 0.0)

            # tighter demo alert logic
            demo_alert = bool(drift > 25.0 and stability < 0.05)
            result["demo_alert"] = demo_alert
            if demo_alert:
                alerts += 1

            results.append(result)
            last_regime = result.get("regime_name")
            last_drift = drift
            last_stability = stability

    pbar.update(len(chunk))
    pbar.set_postfix({
        "results": len(results),
        "alerts": alerts,
        "regime": last_regime,
        "drift": None if last_drift is None else round(last_drift, 3),
        "stability": None if last_stability is None else round(last_stability, 3),
    })

pbar.close()

results_df = pd.DataFrame(results)
results_df.to_csv("greenhouse_results.csv", index=False)

print("\nDONE")
print(f"total_rows_processed={total_rows}")
print(f"total_results={len(results)}")
print(f"total_demo_alerts={alerts}")
print("saved greenhouse_results.csv")
