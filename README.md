# neraium-core
Universal Infrastructure Telemetry Drift Detection Platform

## SII Analytical Modules (Experimental)

The canonical `neraium_core` package contains experimental SII analytical modules that reintroduce advanced analytical capabilities without adding any new top-level package.

These modules are lightly integrated with `StructuralEngine` through `experimental_analytics`, while keeping the runtime engine behavior and baseline drift workflow intact.

### Rigorous / structural modules
- `neraium_core.geometry`: correlation matrix computation and baseline-relative relational drift.
- `neraium_core.spectral`: eigendecomposition, spectral radius, spectral gap, dominant mode loadings.
- `neraium_core.graph`: thresholded adjacency construction and graph metrics (degree, density, clustering, connectivity).
- `neraium_core.entropy`: interaction entropy over structural matrices.
- `neraium_core.subsystems`: subsystem discovery from clustered relationships and local dominant spectral instability.

### Proxy-based / heuristic modules
- `neraium_core.directional`: lagged directional matrix, causal energy/asymmetry/divergence, likely failure-origin proxy.
- `neraium_core.early_warning`: per-signal variance, lag-1 autocorrelation, averaged critical-slowing indicators.
- `neraium_core.forecasting`: instability trend, drift-velocity smoothing, time-to-instability estimate.
- `neraium_core.scoring`: configurable composite instability score across structural and proxy channels.
