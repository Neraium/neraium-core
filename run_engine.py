from neraium_core.baseline import BASELINE_SYSTEM, DEMO_BASELINE
from neraium_core.pipeline import NeraiumPipeline
from neraium_core.replay import load_replay_file


def main():
    print("Neraium Alignment Engine initialized.")
    print("System:", BASELINE_SYSTEM.system_id)

    pipeline = NeraiumPipeline(BASELINE_SYSTEM, DEMO_BASELINE)
    payloads = load_replay_file("data/sample_telemetry.json")

    for payload in payloads:
        results = pipeline.ingest(payload)
        for result in results:
            print("\n[window emitted]")
            print("window_end=", result.window_end)
            print("vector=", result.vector)
            print("accepted_for_scoring=", result.accepted_for_scoring)
            print("score=", result.score)
            print("reason=", result.reason)


if __name__ == "__main__":
    main()