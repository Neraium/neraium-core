"""Structural detection engine: core runtime layers.

The engine is organized into clear responsibility layers:
- core: Frame ingestion, normalization, data quality
- windows: Window extraction and baseline caching
- drift: Structural drift detection and state machine
- relational: Relational stability and graph metrics
- temporal: Temporal coherence and feature extraction
- transitions: Transition pressure and regime tracking
- orchestration: Coordinates core detections
- state: Baseline and regime persistence
- packaging: Output assembly and schema
- config: Shared constants

Core detections (drift, relational, temporal, transitions) form the
production-critical path. Auxiliary analytics are isolated in the
separate auxiliary/ package.

Primary Interface
=================
For most users, import the unified Engine:

    from neraium_core import Engine

    engine = Engine()
    result = engine.ingest_frame(timestamp=..., unit_id=..., sensors={...})

See unified.py for full API documentation.
"""

from neraium_core.engine.unified import Engine
from neraium_core.engine.production import ProductionEngine, InputFrame, EngineResult, BatchResult

__all__ = ["Engine", "ProductionEngine", "InputFrame", "EngineResult", "BatchResult"]
