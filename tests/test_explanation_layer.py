from neraium_core.explanation_layer import build_explanation_text


def test_build_explanation_text_includes_required_elements() -> None:
    text = build_explanation_text(
        current_decision="STRUCTURAL_INSTABILITY_OBSERVED",
        attribution={"top_drivers": ["sensor_7", "sensor_2"]},
        risk="HIGH",
        confidence="medium",
        recommended_action="Inspect sensor_7 pathway",
    )

    assert "STRUCTURAL_INSTABILITY_OBSERVED" in text
    assert "sensor_7" in text
    assert "Confidence is medium" in text
    assert "Recommended action: Inspect sensor_7 pathway." in text
    assert 2 <= len([s for s in text.split(".") if s.strip()]) <= 4


def test_build_explanation_text_without_recommendation() -> None:
    text = build_explanation_text(
        current_decision="NOMINAL_STRUCTURE",
        attribution={"top_drivers": []},
        risk="LOW",
        confidence=0.82,
    )

    assert "NOMINAL_STRUCTURE" in text
    assert "no dominant driver" in text
    assert "Confidence is high" in text
    assert "Recommended action" not in text
