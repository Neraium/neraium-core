from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODULE_PATH = Path(__file__).resolve().parents[1] / "neraium_core" / "engine_stages" / "analytics_packaging.py"
SPEC = importlib.util.spec_from_file_location("analytics_packaging_stage", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


AnalyticsPackagingInput = MODULE.AnalyticsPackagingInput
AnalyticsPackagingResult = MODULE.AnalyticsPackagingResult
package_analytics_output = MODULE.package_analytics_output


def _fallback(reason: str) -> dict[str, dict[str, object]]:
    return {
        "trajectory_analysis": {"available": False, "reason": reason},
        "branching_analysis": {"available": False, "reason": reason},
        "constraint_analysis": {"available": False, "reason": reason},
        "hierarchy_analysis": {"available": False, "reason": reason},
        "horizon_analysis": {"available": False, "reason": reason},
        "counterfactual_simulation": {"available": False, "reason": reason},
    }


def test_analytics_packaging_contract_and_shape() -> None:
    analytics = {
        "trajectory_analysis": {"available": True, "label": "METASTABLE"},
        "branching_analysis": {"available": True, "branch_count_estimate": 2},
        "custom": {"k": 1},
    }
    result = package_analytics_output(
        AnalyticsPackagingInput(analytics=analytics, unavailable_payload=_fallback("missing inputs"))
    )

    assert isinstance(result, AnalyticsPackagingResult)
    assert result.experimental_analytics["trajectory_analysis"]["label"] == "METASTABLE"
    assert result.experimental_analytics["custom"] == {"k": 1}
    assert result.experimental_analytics["constraint_analysis"]["available"] is False


def test_analytics_packaging_preserves_existing_metadata_and_fallback_defaults() -> None:
    analytics = {
        "trajectory_analysis": {"available": True, "path": "DIVERGING"},
        "counterfactual_simulation": {"available": True, "scenario_spread": 0.3},
    }
    packaged = package_analytics_output(
        AnalyticsPackagingInput(analytics=analytics, unavailable_payload=_fallback("missing inputs"))
    )

    assert packaged.analytics["trajectory_analysis"]["path"] == "DIVERGING"
    assert packaged.analytics["counterfactual_simulation"]["scenario_spread"] == 0.3
    assert packaged.analytics["horizon_analysis"] == {"available": False, "reason": "missing inputs"}
