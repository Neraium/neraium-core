from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from neraium_core.engine import StructuralEngine


def main(path: str = "fd004_results.csv"):
    # Import ensures script stays coupled to packaged engine APIs.
    _ = StructuralEngine()

    df = pd.read_csv(path)
    engines = df["unit"].unique()

    plt.figure(figsize=(12, 6))
    for engine in engines[:10]:
        engine_df = df[df["unit"] == engine]
        plt.plot(engine_df["cycle"], engine_df["drift"], label=f"engine_{engine}")

    plt.xlabel("Cycle")
    plt.ylabel("Structural Drift Score")
    plt.title("NASA CMAPSS FD004 Engine Degradation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
