# Systemic Infrastructure Intelligence (SII)

SII estimates **evolving relational geometry** in multivariate telemetry. Instead of flagging isolated point anomalies, it tracks how correlation structure, directional proxy structure, graph topology, and spectral signatures change over time.

## Why this is not ordinary anomaly detection

Traditional anomaly detection often scores outlier points. SII focuses on **loss of structural stability** in the system by comparing sliding-window relational matrices to healthy baseline structure and regime-specific signatures.

## What is rigorous vs proxy-based

Rigorous statistical quantities include windowed standardization, correlation geometry, Frobenius drift, graph observables, and eigenspectrum of correlation matrices. Directional metrics are explicitly **proxy-based** from lagged correlations and are not formal causal inference.

## API

- `POST /telemetry` ingest timestamped signals
- `GET /windows/latest`
- `GET /observables/latest`
- `GET /drift/latest`
- `GET /regime/latest`
- `GET /forecast/latest`
- `GET /health`

Run locally:

```bash
pip install -e .[dev]
uvicorn api.server:app --reload
```

## Limitations

- Directional proxies are not Granger or intervention-based causality.
- Regime assignment is nearest-signature matching, not full hidden-state inference.
- Baseline quality depends on initial windows labeled healthy.
