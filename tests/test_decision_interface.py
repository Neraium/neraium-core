from neraium_core.decision_interface import attach_action_signal_to_result


def test_attach_action_signal_uses_prior_when_missing() -> None:
    result = {"instability": 0.1}
    prior = {"action": "watch", "confidence": 0.65}

    sig = attach_action_signal_to_result(result, prior)

    assert result["action_signal"]["action"] == "watch"
    assert sig["confidence"] == 0.65


def test_attach_action_signal_preserves_existing_signal() -> None:
    result = {"action_signal": {"action": "hedge", "confidence": 0.8}}

    sig = attach_action_signal_to_result(result, None)

    assert sig["action"] == "hedge"
    assert result["action_signal"]["confidence"] == 0.8
