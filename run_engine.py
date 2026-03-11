from neraium_core.alignment import AlignmentEngine
from neraium_core.baseline import BASELINE_SYSTEM


def main():
    engine = AlignmentEngine(BASELINE_SYSTEM)

    print("Neraium Alignment Engine initialized.")
    print("System ID:", BASELINE_SYSTEM.system_id)
    print("Inference window:", BASELINE_SYSTEM.inference_window_seconds)


if __name__ == "__main__":
    main()