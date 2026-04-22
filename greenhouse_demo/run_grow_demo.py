import pandas as pd
from neraium_core.alignment import StructuralEngine

CSV_PATH = "Greenhouse_climate.csv"
BATCH_SIZE = 250

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

df[numeric_cols] = df[numeric_cols].ffill().fillna(0.0)

print(f"rows={len(df)}")
print(f"numeric_cols={len(numeric_cols)}")
print("first sensors:", numeric_cols[:12])

engine = StructuralEngine(
    baseline_window=50,
    recent_window=12,
)

try:
    if hasattr(engine, "geometry_layer") and engine.geometry_layer is not None:
        engine.geometry_layer.update = lambda *args, **kwargs: {}
except Exception:
    pass

results = []
alerts = 0

for start in range(0, len(df), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(df))
    chunk = df.iloc[start:end]

    chunk_results = 0
    last_regime = None
    last_drift = None
    last_stability = None

    for _, row in chunk.iterrows():
        frame = {
            "timestamp": float(row["GHtime"]),
            "site_id": "grow-house",
            "asset_id": "zone-A",
            "sensor_values": {col: float(row[col]) for col in numeric_cols},
        }

        result = engine.process_frame(frame)
        if result:
            results.append(result)
            chunk_results += 1
            if result.get("drift_alert"):
                alerts += 1
            last_regime = result.get("regime_name")
            last_drift = result.get("structural_drift_score")
            last_stability = result.get("relational_stability_score")

    pct = (end / len(df)) * 100
    print(
        f"[{start:>6}-{end:>6}] "
        f"{pct:6.2f}% complete | "
        f"chunk_results={chunk_results:>4} | "
        f"total_results={len(results):>5} | "
        f"alerts={alerts:>4} | "
        f"last_regime={last_regime} | "
        f"last_drift={last_drift} | "
        f"last_stability={last_stability}"
    )

print("\nDONE")
print(f"total_rows={len(df)}")
print(f"total_results={len(results)}")
print(f"total_alerts={alerts}")

print("\nlast_10_results:")
for r in results[-10:]:
    print({
        "t": r.get("timestamp"),
        "drift": r.get("structural_drift_score"),
        "stability": r.get("relational_stability_score"),
        "regime": r.get("regime_name"),
        "alert": r.get("drift_alert"),
    })
