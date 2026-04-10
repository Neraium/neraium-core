import pandas as pd
from tqdm import tqdm
from neraium_core.alignment import StructuralEngine

CSV_PATH = "Greenhouse_climate.csv"

# turbo demo settings
EVERY_N = 100
MAX_ROWS = 250
BATCH_SIZE = 25

KEEP_COLS = ["CO2air", "HumDef", "RHair", "Tair"]

df = pd.read_csv(CSV_PATH)

if "GHtime" in df.columns:
    df["GHtime"] = pd.to_numeric(df["GHtime"], errors="coerce")
    df = df[df["GHtime"].notna()].sort_values("GHtime").reset_index(drop=True)
else:
    df["GHtime"] = range(len(df))

usable_cols = []
for col in KEEP_COLS:
    if col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
            usable_cols.append(col)

if not usable_cols:
    raise RuntimeError("No usable demo columns found in Greenhouse_climate.csv")

df = df[["GHtime"] + usable_cols].copy()
df[usable_cols] = df[usable_cols].ffill().fillna(0.0)

df = df.iloc[::EVERY_N].reset_index(drop=True)
if len(df) > MAX_ROWS:
    step = max(1, len(df) // MAX_ROWS)
    df = df.iloc[::step].reset_index(drop=True)

print(f"rows_after_downsample={len(df)}")
print(f"usable_cols={usable_cols}")

engine = StructuralEngine(
    baseline_window=10,
    recent_window=4,
)

try:
    if hasattr(engine, "geometry_layer") and engine.geometry_layer is not None:
        engine.geometry_layer.update = lambda *args, **kwargs: {}
except Exception:
    pass

results = []
alerts = 0
total_rows = len(df)

pbar = tqdm(total=total_rows, desc="Turbo greenhouse demo", unit="rows")

for start in range(0, total_rows, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total_rows)
    chunk = df.iloc[start:end]

    last_regime = None
    last_drift = None
    last_stability = None

    for row in chunk.itertuples(index=False, name=None):
        frame = {
            "timestamp": float(row[0]),
            "site_id": "grow-house",
            "asset_id": "zone-A",
            "sensor_values": {usable_cols[i]: float(row[i + 1]) for i in range(len(usable_cols))},
        }

        result = engine.process_frame(frame)
        if result:
            drift = float(result.get("structural_drift_score") or 0.0)
            stability = float(result.get("relational_stability_score") or 0.0)
            demo_alert = bool(drift > 10.0 and stability < 0.20)
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

pd.DataFrame(results).to_csv("greenhouse_results_turbo.csv", index=False)

print("\nDONE")
print(f"total_rows_processed={total_rows}")
print(f"total_results={len(results)}")
print(f"total_demo_alerts={alerts}")
print("saved greenhouse_results_turbo.csv")
