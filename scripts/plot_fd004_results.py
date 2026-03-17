import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv("fd004_results.csv")
    plt.figure(figsize=(12, 6))
    for engine in df["unit"].unique()[:10]:
        engine_df = df[df["unit"] == engine]
        plt.plot(engine_df["cycle"], engine_df["drift"], label=f"engine_{engine}")
    plt.xlabel("Cycle")
    plt.ylabel("Structural Drift Score")
    plt.title("NASA CMAPSS FD004 Engine Degradation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
