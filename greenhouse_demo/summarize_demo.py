import pandas as pd

df = pd.read_csv("greenhouse_results_turbo.csv")

print("---- DEMO SUMMARY ----")
print("rows:", len(df))
print("alerts:", df.get("demo_alert", []).sum())
print("avg drift:", df["structural_drift_score"].mean())
print("max drift:", df["structural_drift_score"].max())
print("----------------------")
