"""Neraium: Structural drift detection engine for industrial equipment.

Primary entry point:

    from neraium_core import Engine

    engine = Engine()
    result = engine.ingest_frame(timestamp=..., unit_id=..., sensors={...})

For validation/benchmarking:

    engine = Engine(enable_shadow_mode=True)
    engine.replay("FD004.csv", dataset_type="fd004")
    metrics = engine.get_summary()
    evidence = engine.get_evidence_report()

See PHASE_A_CONTRACT_AND_ISOLATION.md for internal contracts.
See PHASE_B_UNIFY_SURFACE.md for unified surface design.
See PRODUCTION_READINESS.md for deployment guidance.
"""

from importlib.metadata import PackageNotFoundError, version

from neraium_core.engine import Engine

try:
    __version__ = version("neraium-core")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "Engine",  # Primary user-facing interface
    "alignment", "models", "pipeline", "service", "store", "sii", "doctrine", "gate",
    "__version__",
]
