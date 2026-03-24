from __future__ import annotations

import tempfile

import numpy as np

from neraium_core.alignment import StructuralEngine


def _frame(t: int, values: dict[str, float]) -> dict[str, object]:
    return {
        "timestamp": str(t),
        "site_id": "site1",
        "asset_id": "asset1",
        "sensor_values": values,
    }


def _run_engine_sequence(engine: StructuralEngine, seq: list[dict[str, float]]) -> dict[str, object]:
    last: dict[str, object] | None = None
    for t, vals in enumerate(seq):
        last = engine.process_frame(_frame(t, vals))
    assert last is not None
    return last


def test_counterfactual_guidance_emits_ranked_contributors_and_directions() -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(baseline_window=14, recent_window=8, regime_store_path=f"{d}/r.json")

        seq: list[dict[str, float]] = []
        for t in range(42):
            base = float(np.sin(0.08 * t))
            if t < 28:
                s2 = 0.95 * base
                s3 = 0.8 * base
            else:
                # Inject strong structural break for s1<->s2 relationship.
                s2 = -0.85 * base + 0.12 * float(np.sin(0.5 * t))
                s3 = 0.35 * base + 0.10 * float(np.cos(0.37 * t))
            seq.append({"s1": base, "s2": s2, "s3": s3})

        last = _run_engine_sequence(engine, seq)
        guidance = (last.get("experimental_analytics") or {}).get("counterfactual_guidance") or {}
        contributors = guidance.get("top_structural_break_contributors") or []
        directions = guidance.get("top_stabilizing_directions") or []

        assert guidance.get("available") is True
        assert contributors
        assert directions
        assert "s1 <-> s2" in {str(item.get("relationship")) for item in contributors[:2]}
        assert float(contributors[0]["contributor_score"]) >= float(contributors[-1]["contributor_score"])
        assert "classification" in (guidance.get("reversibility") or {})


def test_counterfactual_guidance_reversibility_heuristic_synthetic_extremes() -> None:
    with tempfile.TemporaryDirectory() as d:
        locked_engine = StructuralEngine(baseline_window=14, recent_window=8, regime_store_path=f"{d}/locked.json")
        reversible_engine = StructuralEngine(
            baseline_window=14,
            recent_window=8,
            regime_store_path=f"{d}/reversible.json",
        )

        # Scenario A: persistent transition + fragmentation => tends to LOCKED_IN.
        seq_locked: list[dict[str, float]] = []
        for t in range(58):
            base = float(np.sin(0.07 * t))
            if t < 28:
                seq_locked.append({"s1": base, "s2": 0.9 * base, "s3": 0.85 * base, "s4": 0.8 * base})
            else:
                seq_locked.append(
                    {
                        "s1": base,
                        "s2": -0.9 * base + 0.25 * float(np.sin(0.5 * t)),
                        "s3": 0.35 * float(np.cos(0.37 * t)) + 0.15 * base,
                        "s4": -0.32 * float(np.sin(0.41 * t)) + 0.08 * float(np.cos(0.19 * t)),
                    }
                )
        last_locked = _run_engine_sequence(locked_engine, seq_locked)

        # Scenario B: perturbation that recovers quickly => should not be LOCKED_IN.
        seq_reversible: list[dict[str, float]] = []
        for t in range(58):
            base = float(np.sin(0.07 * t))
            if t < 28:
                s2 = 0.9 * base
                s3 = 0.88 * base
            elif t < 40:
                s2 = 0.5 * base + 0.18 * float(np.sin(0.42 * t))
                s3 = 0.45 * base + 0.11 * float(np.cos(0.31 * t))
            else:
                s2 = 0.92 * base
                s3 = 0.87 * base
            seq_reversible.append({"s1": base, "s2": s2, "s3": s3})
        last_reversible = _run_engine_sequence(reversible_engine, seq_reversible)

        rev_locked = (
            ((last_locked.get("experimental_analytics") or {}).get("counterfactual_guidance") or {})
            .get("reversibility", {})
            .get("classification")
        )
        rev_reversible = (
            ((last_reversible.get("experimental_analytics") or {}).get("counterfactual_guidance") or {})
            .get("reversibility", {})
            .get("classification")
        )

        assert rev_locked in {"LOCKED_IN", "METASTABLE"}
        assert rev_reversible in {"REVERSIBLE", "METASTABLE"}
        assert rev_locked != "REVERSIBLE"
