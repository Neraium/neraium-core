# neraium-core

Infrastructure telemetry drift detection engine.

## Project structure

```text
neraium-core
├── neraium_core/
│   ├── __init__.py
│   ├── engine.py
│   ├── lead_time.py
│   └── ingest.py
├── scripts/
│   └── fd004_plot.py
├── tests/
│   ├── test_engine.py
│   └── test_pipeline.py
├── pyproject.toml
├── README.md
└── .gitignore
```

## Installation

```bash
pip install -e .[dev]
```

## Run the engine

```bash
python run_engine.py
```

Or in Python:

```python
from neraium_core.engine import StructuralEngine

engine = StructuralEngine()
```
