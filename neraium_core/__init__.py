"""Neraium analytical core package."""

__all__ = [
    "StructuralEngine",
    "StructuralIntelligenceService",
    "normalize_rest_payload",
    "parse_csv_text",
    "DetectorConfig",
    "DetectionResult",
    "HybridSIIDetector",
]

try:
    from neraium_core.engine import StructuralEngine
    from neraium_core.service import StructuralIntelligenceService
except ModuleNotFoundError:
    StructuralEngine = None
    StructuralIntelligenceService = None

from neraium_core.ingest import normalize_rest_payload, parse_csv_text

try:
    from neraium_core.lead_time import DetectionResult, DetectorConfig, HybridSIIDetector
except ModuleNotFoundError:
    DetectionResult = None
    DetectorConfig = None
    HybridSIIDetector = None
