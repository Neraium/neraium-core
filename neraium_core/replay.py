from neraium_core.replay import load_replay_file


def test_load_replay_file():
    payloads = load_replay_file("data/sample_telemetry.json")
    assert len(payloads) > 0
    assert payloads[0].system_id == "baseline"