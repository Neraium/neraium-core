"""Engine stages for structural analytics and typed governance checks."""

from neraium_core.engine_stages.action_governance import (
    ActionGovernanceInput,
    ActionGovernanceResult,
    ActionValidityAssessment,
    evaluate_action_governance,
)
from neraium_core.engine_stages.analytics_packaging import (
    AnalyticsPackagingInput,
    AnalyticsPackagingResult,
    package_analytics_output,
)
from neraium_core.engine_stages.decision_projection import (
    DecisionProjectionInput,
    apply_transition_state_mapping,
    project_decision_and_explanation,
)
from neraium_core.engine_stages.decision_validity import (
    DecisionValidityInput,
    DecisionValidityResult,
    evaluate_decision_validity,
)
from neraium_core.engine_stages.fragility import FragilityInput, FragilityResult, evaluate_fragility
from neraium_core.engine_stages.ingress_history import (
    IngressAndHistoryBuffersInput,
    IngressAndHistoryBuffersResult,
    prepare_ingress_and_history_buffers,
)
from neraium_core.engine_stages.regime_mutation import (
    RegimeMutationInput,
    RegimeMutationResult,
    mutate_regime_state,
)
from neraium_core.engine_stages.representation_quality import (
    RepresentationAndQualityInput,
    RepresentationAndQualityResult,
    build_representation_and_quality,
)
from neraium_core.engine_stages.risk_guidance import (
    RiskGuidanceInput,
    RiskGuidanceResult,
    evaluate_risk_guidance,
)
from neraium_core.engine_stages.score_computation import (
    ScoreComputationInput,
    ScoreComputationResult,
    compute_scores,
)
from neraium_core.engine_stages.scoring_preparation import (
    ScoringPreparationInput,
    ScoringPreparationResult,
    prepare_scoring_inputs,
)
from neraium_core.engine_stages.stage_boundaries import EngineStageBoundary, structural_engine_stage_groups
from neraium_core.engine_stages.warmup_defaults import build_warmup_result_payload

__all__ = [
    "ActionGovernanceInput",
    "ActionGovernanceResult",
    "ActionValidityAssessment",
    "AnalyticsPackagingInput",
    "AnalyticsPackagingResult",
    "DecisionProjectionInput",
    "DecisionValidityInput",
    "DecisionValidityResult",
    "EngineStageBoundary",
    "FragilityInput",
    "FragilityResult",
    "IngressAndHistoryBuffersInput",
    "IngressAndHistoryBuffersResult",
    "RegimeMutationInput",
    "RegimeMutationResult",
    "RepresentationAndQualityInput",
    "RepresentationAndQualityResult",
    "RiskGuidanceInput",
    "RiskGuidanceResult",
    "ScoreComputationInput",
    "ScoreComputationResult",
    "ScoringPreparationInput",
    "ScoringPreparationResult",
    "apply_transition_state_mapping",
    "build_representation_and_quality",
    "build_warmup_result_payload",
    "compute_scores",
    "evaluate_action_governance",
    "evaluate_decision_validity",
    "evaluate_fragility",
    "evaluate_risk_guidance",
    "mutate_regime_state",
    "package_analytics_output",
    "prepare_ingress_and_history_buffers",
    "prepare_scoring_inputs",
    "project_decision_and_explanation",
    "structural_engine_stage_groups",
]
