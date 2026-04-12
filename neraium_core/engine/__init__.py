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
"""

__all__ = []
