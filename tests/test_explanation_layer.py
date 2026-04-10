from neraium_core.explanation_layer import build_explanation_text


def test_build_explanation_text_includes_required_elements() -> None:
    text = build_explanation_text(
        current_decision="STRUCTURAL_INSTABILITY_OBSERVED",
        attribution={"top_drivers": ["sensor_7", "sensor_2"]},
        risk="HIGH",
        confidence="medium",
        recommended_action="structural instability present",
    )

    assert "STRUCTURAL_INSTABILITY_OBSERVED" in text
    assert "sensor_7" in text
    assert "Confidence is medium" in text
    assert "Recommended action: structural instability present." in text
    assert 2 <= len([s for s in text.split(".") if s.strip()]) <= 4


def test_build_explanation_text_reports_doctrine_refusal_reason() -> None:
    text = build_explanation_text(
        current_decision="WATCH",
        attribution={"top_drivers": ["sensor_7"]},
        risk="MEDIUM",
        confidence=0.7,
        recommended_action="inspect sensor_7 pathway",
    )

    assert "Doctrine refusal:" in text
    assert "insufficient_evidence" in text


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


def test_build_explanation_text_with_memory_context() -> None:
    text = build_explanation_text(
        current_decision="WATCH",
        attribution={"top_drivers": ["sensor_7"]},
        risk="MEDIUM",
        confidence=0.6,
        memory_recall={
            "nearest_match": {
                "found": True,
                "similarity": 0.84,
                "asset_id": "unit-14",
                "run_id": "run-x",
                "cycle": 12,
            },
            "novelty": {"is_novel": False, "reason": "resembles_known_structural_pattern"},
        },
    )
    assert "resembles a prior pattern" in text
    assert "unit-14" in text
