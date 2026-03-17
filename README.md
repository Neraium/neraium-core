# neraium-core

`neraium-core` is a structural telemetry monitoring prototype focused on **drift detection from baseline behavior**.

## What this repository currently does

This repository currently implements a structural drift engine that:
- converts telemetry into ordered sensor vectors,
- forms an initial healthy baseline,
- computes Mahalanobis drift from that baseline,
- computes covariance/relational drift between baseline and recent windows,
- fuses those metrics into a structural drift score,
- reports operationally-oriented outputs such as health and lead-time estimates.

## Rigorous vs heuristic outputs

### Rigorous / stronger pieces
- baseline-relative multivariate state tracking,
- Mahalanobis distance,
- covariance/relational drift,
- structural drift fusion.

### Heuristic / approximate pieces
- alert thresholds (`STABLE`/`WATCH`/`ALERT`),
- health score conversion,
- lead-time estimate from drift velocity,
- textual driver and impact strings.

## Current limitations

This is **not yet** the full intended SII platform (graph/spectral/regime/subsystem layers).
It is a focused core prototype for structural drift only.

## Repository structure

```text
neraium-core/
├── neraium_core/
│   ├── __init__.py
│   ├── engine.py
│   ├── ingest.py
│   ├── models.py
│   ├── service.py
│   └── config.py
├── scripts/
│   └── run_engine.py
├── tests/
│   ├── test_engine.py
│   ├── test_ingest.py
│   └── test_service.py
├── .github/workflows/ci.yml
├── pyproject.toml
├── README.md
└── .gitignore
```

## Install

```bash
python -m pip install -e .[dev]
```

## Quick usage

```python
from neraium_core.service import StructuralIntelligenceService

service = StructuralIntelligenceService()
result = service.ingest_frame(
    {
        "timestamp": "2026-01-01T00:00:00Z",
        "site_id": "site-a",
        "asset_id": "asset-1",
        "sensor_values": {"temp": 42.0, "pressure": 99.1, "vibration": 0.6},
    }
)

print(result.to_dict())
```

## Compatibility notes

- `run_engine.py` remains as a backward-compatible wrapper exporting `StructuralEngine`.
- `ingest.py` remains as a backward-compatible wrapper exporting ingestion helpers.
- Core logic now lives under `neraium_core/` for package-centric testable architecture.

## Roadmap

Future work can incrementally add graph, spectral, regime, and subsystem layers while preserving this structural-drift core as a stable foundation.
