from __future__ import annotations

import numpy as np


def forecast_instability(history: list[float], threshold: float = 2.5) -> dict[str, float | None]:
    if len(history) < 2:
        return {"slope": 0.0, "recent_slope": 0.0, "acceleration": 0.0, "time_to_instability": None}
    x = np.arange(len(history), dtype=float)
    y = np.array(history, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    span = min(5, len(y) - 1)
    recent_slope = float((y[-1] - y[-(span + 1)]) / max(span, 1))
    acceleration = float(recent_slope - slope)
    tti = None
    if recent_slope > 0 and y[-1] < threshold:
        tti = float((threshold - y[-1]) / recent_slope)
    return {"slope": slope, "recent_slope": recent_slope, "acceleration": acceleration, "time_to_instability": tti}
