from pathlib import Path

import pandas as pd

from neraium_core.alignment import StructuralEngine

DATA_DIR = Path("/content/data")  # change to your folder
DATASET = "FD004"  # FD001, FD002, FD003, FD004

file_path = DATA_DIR / f"test_{DATASET}.txt"

df = pd.read_csv(file_path, sep=r"\s+", header=None)
df.columns = ["unit", "cycle"] + [f"os{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]

engine = StructuralEngine(
    baseline_window=50,
    recent_window=12,
)

results = []
global_t = 0.0

for _, row in df.iterrows():
    frame = {
        "timestamp": global_t,
        "site_id": "nasa_cmapss",
        "asset_id": f"engine_{int(row['unit'])}",
        "sensor_values": {
            f"os{i}": float(row[f"os{i}"]) for i in range(1, 4)
        }
        | {
            f"s{i}": float(row[f"s{i}"]) for i in range(1, 22)
        },
    }

    out = engine.process_frame(frame)
    out["unit"] = int(row["unit"])
    out["cycle"] = int(row["cycle"])
    results.append(out)

    global_t += 1.0

out_df = pd.DataFrame(results)
out_df.to_csv(f"{DATASET}_results.csv", index=False)

print("saved", f"{DATASET}_results.csv")
print(out_df.head())
print(out_df.columns.tolist())
