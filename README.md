# neraium-core

Systemic Infrastructure Intelligence (SII) engine for telemetry-based structural drift analysis.

## Architecture overview

- `neraium_core/engine.py`: Rolling-window structural intelligence engine.
- `neraium_core/scoring.py`: Drift-scoring primitives (Mahalanobis, covariance shift, z-score anomalies).
- `neraium_core/ingest.py`: Telemetry normalization/validation for incoming frames.
- `neraium_core/lead_time.py`: Lead-time estimator based on drift velocity.
- `neraium_core/telemetry.py`: Shared telemetry frame model.
- `scripts/fd004_plot.py`: Example plotting workflow for FD004 replay outputs.

## Quick start

```bash
pip install -e .[dev]
ruff check .
pytest
```

## Example usage

```python
from neraium_core.engine import StructuralEngine

engine = StructuralEngine()

frame = {
    "timestamp": "2026-01-01T00:00:00Z",
    "site_id": "site-a",
    "asset_id": "pump-1",
    "sensor_values": {
        "pressure": 118.2,
        "flow": 79.1,
        "vibration": 0.31,
    },
}

engine.ingest(frame)
score = engine.score()
print(score)
```
