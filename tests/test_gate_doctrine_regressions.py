from neraium_core.doctrine import assess_assertion
from neraium_core.explanation_layer import build_explanation_text
from neraium_core.gate.evaluator import evaluate_context_exclusions
from neraium_core.gate.schema import GateInput


def test_assess_assertion_does_not_treat_partial_words_as_prescriptive() -> None:
    assessment = assess_assertion("The shoulder bearing shows observable vibration.")

    assert assessment.allowed is True
    assert assessment.classification == "observational"


def test_evaluate_context_exclusions_handles_none_context_flags() -> None:
    gate_input = GateInput(timestamp=0.0, asset_id="asset-1", context_flags=None)  # type: ignore[arg-type]

    result = evaluate_context_exclusions(gate_input)

    assert result.status == "PASS"
    assert "no exclusionary context flags active" in result.reason


def test_build_explanation_text_keeps_gate_outcome_when_memory_and_action_present() -> None:
    text = build_explanation_text(
        current_decision="WATCH",
        attribution={"top_drivers": ["sensor_7"]},
        risk="MEDIUM",
        confidence=0.7,
        recommended_action="inspect vibration trend",
        memory_recall={
            "nearest_match": {
                "found": True,
                "similarity": 0.87,
                "asset_id": "unit-9",
            }
        },
        gate_decision={"decision": "SUPPRESS", "refusal_reason": "insufficient corroboration"},
    )

    assert "Aletheia's Gate suppressed this observation: insufficient corroboration" in text
