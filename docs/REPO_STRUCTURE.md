# Repository structure audit and target layout

## Current code classification

### Production product code

- `apps/api/` (FastAPI app, web/static serving, API routes)
- `neraium_core/` (core structural engine, product/service layers, platform logic)
- `tools/` (release/demo/validation orchestration used by product workflows)
- `validation/` (validation pipeline and release gate logic)

### Experiments / notebooks / reports / benchmarks

- `experiments/` (new home for scripts, benchmarks, monolith exports, notebook helper scripts)
- `notebooks/` (Jupyter notebooks)
- `evaluation/` (scenario/evaluation bench harnesses)
- `reports/`, `artifacts/`, `colab_outputs/` (generated analysis outputs)
- `synthetic/` (synthetic benchmark helpers)

### Legacy or redundant runner scripts (now wrapped)

Legacy root scripts were moved to `experiments/` and kept as compatibility wrappers at root so older commands remain valid.

## Target structure

```text
apps/
  api/                    # app/API surface
neraium_core/             # core engine + product logic
  adapters/
  integrations/
experiments/              # research and prototyping only
  scripts/
  benchmarks/
  monoliths/
  notebooks/
docs/                     # docs and runbooks
tests/                    # automated tests
tools/                    # operational utility CLIs
validation/               # release validation pipeline
```

## Migration policy used

1. Move research/prototype assets into `experiments/`.
2. Preserve existing entrypoints with thin wrappers.
3. Deduplicate root modules by forwarding to canonical `neraium_core` implementations.
4. Keep changes incremental and review-friendly.
