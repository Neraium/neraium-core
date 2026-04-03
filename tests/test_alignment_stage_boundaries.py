from __future__ import annotations

from neraium_core.alignment import StructuralEngine


def test_structural_engine_stage_groups_are_typed_and_include_low_risk_candidates() -> None:
    groups = StructuralEngine.stage_groups()
    assert isinstance(groups, list)
    assert groups
    first = groups[0]
    assert "stage" in first
    assert "inputs" in first
    assert "outputs" in first
    assert any(item.get("extraction_candidate") and item.get("risk") == "low" for item in groups)
