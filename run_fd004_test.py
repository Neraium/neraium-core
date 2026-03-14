import pandas as pd
from run_engine import StructuralEngine

# change this path to where your dataset is
file_path = r"C:\Users\Craig\Downloads\CMAPSSData\train_FD004.txt"

df = pd.read_csv(file_path, sep="\s+", header=None)

columns = (
    ["unit", "cycle"]
    + ["op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

df.columns = columns

results = []

for unit_id in df["unit"].unique():

    engine = StructuralEngine()

    unit_data = df[df["unit"] == unit_id]

    for _, row in unit_data.iterrows():

        sensors = {}

        for s in [f"s{i}" for i in range(1, 22)]:
            sensors[s] = row[s]

        frame = {
            "timestamp": str(row["cycle"]),
            "site_id": "nasa",
            "asset_id": f"engine_{unit_id}",
            "sensor_values": sensors
        }

        result = engine.process_frame(frame)

        results.append({
            "unit": unit_id,
            "cycle": row["cycle"],
            "drift": result["structural_drift_score"],
            "mahal": result["mahalanobis_score"],
            "cov": result["covariance_drift_score"],
            "velocity": result["drift_velocity"],
            "state": result["state"]
        })

results_df = pd.DataFrame(results)

results_df.to_csv("fd004_results.csv", index=False)

print("FD004 replay complete")