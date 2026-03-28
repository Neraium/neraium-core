from __future__ import annotations

from neraium_core.assistant_layer import build_assistant_context, render_assistant_report, render_assistant_response


def _state(cycle: int = 3) -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:03:00+00:00",
        "cycle": cycle,
        "run_id": "run-x",
        "customer_id": "cust-x",
        "session": {"run_id": "run-x", "site_id": "site-1", "asset_id": "asset-2"},
        "risk_assessment": {"risk_level": "MEDIUM", "trend": "RISING", "latest_instability": 0.88},
        "operational_recommendation": {
            "recommended_action": "inspect pressure train",
            "recommendation_confidence": 0.78,
            "rationale": "Instability and trend are rising.",
            "operator_note": "Advisory only.",
            "status": {"available": True, "advisory": True},
        },
        "explanation_text": "Pressure and temperature drifted together.",
        "events": ["recommendation_available", "deterioration_detected"],
        "memory_recall": {
            "novelty": {"is_novel": False, "reason": "similar historical profile"},
            "nearest_match": {"found": True, "similarity": 0.9, "summary": "Prior pressure drift incident"},
            "top_matches": [{"summary": "Prior pressure drift incident", "similarity": 0.9, "scope": "same_site"}],
            "pattern_family": {"label": "pressure-drift", "confidence": 0.81},
        },
    }


def test_context_builder_includes_expected_shape() -> None:
    current = _state()
    history = [current, {**_state(cycle=2), "risk_assessment": {"risk_level": "LOW"}, "events": ["recommendation_available"]}]

    context = build_assistant_context(current_state=current, recent_history=history)

    assert context["current_state"]["cycle"] == 3
    assert context["risk"]["risk_level"] == "MEDIUM"
    assert context["recommendation"]["recommended_action"] == "inspect pressure train"
    assert context["memory_recall"]["nearest_match"]["found"] is True
    assert isinstance(context["recent_timeline"], list)
    assert context["recent_changes"]["risk_level"]["previous"] == "LOW"


def test_grounding_sections_distinguish_observed_inferred_recommended() -> None:
    context = build_assistant_context(current_state=_state(), recent_history=[_state(), _state(cycle=2)])

    response = render_assistant_response(mode="summary", context=context)

    assert response["mode"] == "summary"
    assert "Observed:" in response["text"]
    assert "Inferred:" in response["text"]
    assert "Recommended:" in response["text"]
    assert response["grounding"]["recommended"]


def test_supported_modes_render_without_missing_sections() -> None:
    context = build_assistant_context(current_state=_state(), recent_history=[_state(), _state(cycle=2)])

    for mode in ("summary", "why_recommended", "what_changed", "pattern_similarity", "handoff"):
        response = render_assistant_response(mode=mode, context=context)
        assert response["mode"] == mode
        assert isinstance(response["text"], str)
        assert response["text"]
        assert "context" in response


def test_report_modes_include_required_sections() -> None:
    context = build_assistant_context(current_state=_state(), recent_history=[_state(), _state(cycle=2)])

    mode_expectations = {
        "client_report": [
            "Overview",
            "Current System State",
            "Risk Assessment",
            "Recommended Next Step (advisory)",
            "Supporting Evidence",
            "Pattern Context (memory recall if present)",
            "Confidence",
            "Operator Note",
        ],
        "technician_summary": [
            "Current state (concise)",
            "What changed",
            "Key drivers",
            "Recommended next step",
        ],
        "inspection_brief": [
            "Target system/component",
            "Why inspection is recommended",
            "What to check",
            "Risk context",
        ],
        "handoff_note": [
            "Overview",
            "Current System State",
            "Risk Assessment",
            "Recommended Next Step (advisory)",
            "Supporting Evidence",
            "Pattern Context (memory recall if present)",
            "Confidence",
            "Operator Note",
        ],
    }

    for mode, required_sections in mode_expectations.items():
        response = render_assistant_report(mode=mode, context=context)
        assert response["mode"] == mode
        assert response["report_text"]
        for section_name in required_sections:
            assert section_name in response["sections"]


def test_report_is_grounded_in_structured_fields() -> None:
    context = build_assistant_context(current_state=_state(), recent_history=[_state(), _state(cycle=2)])

    response = render_assistant_report(mode="client_report", context=context)

    text = response["report_text"]
    assert "MEDIUM" in text
    assert "inspect pressure train" in text
    assert "Pressure and temperature drifted together." in text
    assert "deterioration_detected" in text
    assert "Prior pressure drift incident" in text
