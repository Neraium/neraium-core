from __future__ import annotations

from ui.reasoning.context_builder import build_admitted_events


def test_event_admitted_parses_common_string_values() -> None:
    records = [
        {"timestamp": "t1", "event_admitted": "False"},
        {"timestamp": "t2", "event_admitted": "0"},
        {"timestamp": "t3", "event_admitted": "no"},
        {"timestamp": "t4", "event_admitted": "True"},
        {"timestamp": "t5", "event_admitted": "1"},
        {"timestamp": "t6", "event_admitted": "yes"},
    ]

    events = build_admitted_events(records)

    assert [event.event_admitted for event in events] == [False, False, False, True, True, True]


def test_event_admitted_uses_threshold_fallback_for_unparseable_values() -> None:
    records = [
        {
            "timestamp": "fallback_true",
            "event_admitted": "maybe",
            "structural_drift_score": 0.9,
            "relational_stability_score": 0.2,
        },
        {
            "timestamp": "fallback_false",
            "event_admitted": "unknown",
            "structural_drift_score": 0.2,
            "relational_stability_score": 0.9,
        },
    ]

    events = build_admitted_events(records, drift_threshold=0.55, stability_threshold=0.45)

    assert [event.event_admitted for event in events] == [True, False]
