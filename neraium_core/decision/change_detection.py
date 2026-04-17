"""Detect and explain what changed between frames.

Identifies specific metric deltas and determines root cause drivers.
"""

from __future__ import annotations

from typing import Any


def detect_metric_deltas(
    current: dict[str, float],
    previous: dict[str, float] | None,
    keys: list[str],
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Identify which specific metrics changed vs prior frame.

    Returns list of changes with delta analysis.
    """
    if not previous:
        return []

    if thresholds is None:
        thresholds = {}

    changes = []
    for key in keys:
        curr = float(current.get(key, 0.0))
        prev = float(previous.get(key, 0.0))
        delta = curr - prev
        threshold = float(thresholds.get(key, 0.05))

        if abs(delta) > threshold:
            changes.append({
                "metric": key,
                "delta": round(delta, 4),
                "from": round(prev, 4),
                "to": round(curr, 4),
                "direction": "up" if delta > 0 else "down",
                "magnitude": round(abs(delta), 4),
            })

    return sorted(changes, key=lambda x: x["magnitude"], reverse=True)


def find_primary_driver(
    changes: list[dict[str, Any]],
    impact_weights: dict[str, float] | None = None,
) -> str | None:
    """Identify which change had the most impact.

    Args:
        changes: List from detect_metric_deltas
        impact_weights: Map of metric to importance in [0, 1]

    Returns:
        The metric name with highest weighted impact
    """
    if not changes:
        return None

    if impact_weights is None:
        impact_weights = {}

    scored = []
    for change in changes:
        metric = change["metric"]
        magnitude = change["magnitude"]
        weight = float(impact_weights.get(metric, 1.0))
        score = magnitude * weight
        scored.append((score, metric))

    if not scored:
        return None

    return max(scored, key=lambda x: x[0])[1]


def describe_drift_change(
    current_drift: float,
    previous_drift: float,
    delta_details: list[dict[str, Any]],
) -> str:
    """Create human-readable description of drift change.

    Prioritizes persistent relationship changes and subsystem instability.
    """
    delta = current_drift - previous_drift
    delta_text = f"({previous_drift:.2f} → {current_drift:.2f}, {delta:+.2f})"

    if not delta_details:
        return f"Structural alignment changed {delta_text}"

    primary = delta_details[0]
    primary_metric = primary["metric"].replace("_", " ")

    if delta > 0.1:
        return f"Structural alignment degraded {delta_text}; {primary_metric} is primary driver"

    if delta < -0.1:
        return f"Structural alignment improved {delta_text}; {primary_metric} is primary recovery factor"

    if len(delta_details) == 1:
        return f"Structural alignment shifted {delta_text} due to {primary_metric}"

    return f"Structural alignment changed {delta_text} from multiple factors ({', '.join(d['metric'] for d in delta_details[:2])})"


def categorize_change(
    metric_name: str,
    current_value: float,
    previous_value: float,
    thresholds: dict[str, tuple[float, float]] | None = None,
) -> str:
    """Categorize the type of change.

    Returns category: 'spike', 'sustained', 'threshold_crossing', 'correction', 'drift'
    """
    if thresholds is None:
        thresholds = {}

    delta = current_value - previous_value
    abs_delta = abs(delta)

    if abs_delta > 0.3:
        return "spike"

    if abs_delta < 0.02:
        return "drift"

    if metric_name in thresholds:
        low, high = thresholds[metric_name]
        prev_crossed = previous_value < low or previous_value > high
        curr_crossed = current_value < low or current_value > high
        if prev_crossed != curr_crossed:
            return "threshold_crossing"

    if delta < 0:
        return "correction"

    return "sustained"
