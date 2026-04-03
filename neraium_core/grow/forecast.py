from __future__ import annotations


def estimate_time_to_impact_hours(
    *,
    drift_score: float,
    instability: float,
    slope: float,
    threshold_breach_detected: bool,
) -> float | None:
    if threshold_breach_detected:
        return 0.0
    pressure = max(drift_score, instability)
    if pressure < 0.16 and slope <= 0.0:
        return None
    urgency_factor = max(0.05, slope + pressure / 3.0)
    estimate = max(2.0, min(72.0, 18.0 / urgency_factor))
    return round(estimate, 1)


def humanize_impact_window(hours: float | None) -> str:
    if hours is None:
        return "No near-term impact window estimated"
    if hours <= 8:
        return f"Likely impact within {hours:.1f} hours"
    if hours <= 24:
        return f"Likely impact within the next shift ({hours:.1f} hours)"
    return f"Likely impact within the next {hours:.1f} hours"
