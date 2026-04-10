from neraium_core.doctrine import refusal_reason_for_statement


def test_refusal_reason_enforces_corroboration_min_confidence() -> None:
    reason = refusal_reason_for_statement(
        "structural instability present",
        confidence=0.5,
        corroborating_signals=3,
    )

    assert reason == "low_confidence"

