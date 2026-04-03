import pandas as pd
import matplotlib.pyplot as plt

# Load results from the replay script
df = pd.read_csv("fd004_results.csv")

engines = df["unit"].unique()

plt.figure(figsize=(12,6))

for engine in engines[:10]:  # plot first 10 engines for clarity
    engine_df = df[df["unit"] == engine]

    plt.plot(
        engine_df["cycle"],
        engine_df["drift"],
        label=f"engine_{engine}"
    )

plt.xlabel("Cycle")
plt.ylabel("Structural Drift Score")
plt.title("NASA CMAPSS FD004 Engine Degradation")
plt.legend()
plt.show()