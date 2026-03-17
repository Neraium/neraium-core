# neraium-core
Universal Infrastructure Telemetry Drift Detection Platform

## SII Analytical Modules (Experimental)

The `neraium_core` package now includes an experimental set of SII analytical modules that extend structural monitoring without introducing a new top-level package.

These modules are designed to complement `StructuralEngine`:
- **Rigorous modules** (geometry, spectral) provide direct matrix-based structural observables.
- **Proxy-oriented modules** (directional, forecasting) provide directional and trend signals that are useful in practice but should be interpreted as heuristics.

Included modules:
- `neraium_core.geometry`: correlation matrix + relational structure extraction
- `neraium_core.spectral`: eigendecomposition, spectral radius, spectral gap
- `neraium_core.graph`: thresholded adjacency + degree/density/clustering metrics
- `neraium_core.directional`: lagged directional correlations + causal metrics
- `neraium_core.entropy`: interaction entropy over structural matrices
- `neraium_core.early_warning`: variance + lag-1 autocorrelation indicators
- `neraium_core.subsystems`: subsystem discovery + subsystem spectral instability
- `neraium_core.forecasting`: instability trend and time-to-instability proxies
- `neraium_core.scoring`: composite instability score over core analytical channels
