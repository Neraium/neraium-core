# Migration: research-heavy root to product-oriented layout

This migration reorganizes the repository so product runtime code is easier to find and research assets are isolated.

## Why this change

The repo had a mix of production entrypoints, API/runtime code, benchmark harnesses, Colab artifacts, and monolith exports all at the root level. That made ownership and release hygiene harder.

This reorganization keeps runtime behavior intact while moving experimental assets into an explicit `experiments/` area.

## What moved

### Moved into `experiments/`

- `analyze_results_v2.py` → `experiments/scripts/analyze_results_v2.py`
- `final_summary.py` → `experiments/scripts/final_summary.py`
- `_show_summary.py` logic → `experiments/scripts/show_summary.py`
- `simulator.py` → `experiments/scripts/simulator.py`
- `run_upgraded_multinode_test.py` → `experiments/benchmarks/run_upgraded_multinode_test.py`
- `lead_time_engine.py` → `experiments/benchmarks/lead_time_engine.py`
- `core_math_engine_monolith.py` → `experiments/monoliths/core_math_engine_monolith.py`
- `intelligence_layer_monolith.py` → `experiments/monoliths/intelligence_layer_monolith.py`
- `colab_intel_layer.py` → `experiments/notebooks/colab_intel_layer.py`

### Compatibility preserved

Root-level compatibility wrappers/shims remain for moved files so existing commands/imports continue to work.

Examples:
- `python run_upgraded_multinode_test.py` still works.
- `from lead_time_engine import HybridSIIDetector` still works.
- `from intelligence_layer_monolith import StructuralEngine` still works.

## Canonical module locations standardized

To avoid duplicated logic at repo root, these root modules are now compatibility shims that point to package implementations:

- `ingest.py` → `neraium_core.pipeline`
- `forecast_models.py` → `neraium_core.forecast_models`
- `regime_store.py` → `neraium_core.regime_store`

## Tooling updates

- `scripts/build_intelligence_monolith.py` now writes monolith outputs under `experiments/monoliths/`.

## Target structure (product-oriented)

- `apps/api/` → app/API serving layer
- `neraium_core/` → core engine and platform logic
- `neraium_core/integrations/`, `neraium_core/adapters/` → integrations
- `experiments/` → research scripts, benchmarks, monoliths, notebook assets
- `docs/` → product and operational docs
- `tests/` → automated validation

## Risk posture

- No intentional behavior change to production runtime paths.
- Entrypoints are preserved via wrappers when files were moved.
- Package imports now favor canonical `neraium_core` implementations.
