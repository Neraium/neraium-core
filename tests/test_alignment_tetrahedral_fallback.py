from __future__ import annotations

import tempfile

from neraium_core.alignment import StructuralEngine


def test_safe_default_tetrahedral_payload_except_path_includes_interpreted_label(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=10,
            recent_window=5,
            regime_store_path=f"{d}/r.json",
        )

        def _boom(**_: object) -> dict[str, object]:
            raise ValueError("forced failure")

        monkeypatch.setattr("neraium_core.alignment.compute_tetrahedral_state", _boom)

        payload = engine._safe_default_tetrahedral_payload()
        assert payload["nearest_vertex"] == "STRUCTURAL"
        assert payload["interpreted_label"] == "STRUCTURAL_STRESS_BUILDING"
