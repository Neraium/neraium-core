from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODULE_PATH = Path(__file__).resolve().parents[1] / "neraium_core" / "engine_stages" / "score_computation.py"
SPEC = importlib.util.spec_from_file_location("score_computation_stage", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


ScoreComputationInput = MODULE.ScoreComputationInput
ScoreComputationResult = MODULE.ScoreComputationResult
compute_scores = MODULE.compute_scores


def test_score_computation_contract_and_insufficient_data_behavior() -> None:
    result = compute_scores(
        ScoreComputationInput(
            base_components={"early_warning": 0.4, "relational_drift": 1.1, "spectral": 0.8, "regime_drift": 0.6},
            raw_components={},
            evidence_confidence=0.2,
            correlation_ready=False,
            regime_factor=0.0,
            transition_aware_enabled=True,
        )
    )
    assert isinstance(result, ScoreComputationResult)
    assert result.component_confidence["relational_drift"] == 0.0
    assert result.component_confidence["early_warning"] == 0.2
    assert result.composite_score >= 0.0


def test_score_computation_parity_for_deterministic_inputs() -> None:
    stage_input = ScoreComputationInput(
        base_components={
            "drift": 1.3,
            "relational_drift": 1.1,
            "regime_drift": 0.7,
            "spectral": 0.8,
            "early_warning": 0.5,
            "transition_pressure": 0.9,
        },
        raw_components={},
        evidence_confidence=0.9,
        correlation_ready=True,
        regime_factor=0.6,
        transition_aware_enabled=True,
    )
    first = compute_scores(stage_input)
    second = compute_scores(stage_input)
    assert first.components == second.components
    assert first.component_confidence == second.component_confidence
    assert first.weights_for_composite == second.weights_for_composite
    assert first.components_for_decision == second.components_for_decision
    assert first.composite_score == second.composite_score
