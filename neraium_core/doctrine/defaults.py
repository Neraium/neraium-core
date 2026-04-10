from __future__ import annotations

from neraium_core.doctrine.schema import (
    CorroborationThresholds,
    DeteriorationCriteria,
    Doctrine,
    PersistenceThresholds,
    UncertaintyOverrideBehavior,
)


DEFAULT_DOCTRINE_V1 = Doctrine(
    doctrine_version="doctrine-v1",
    allowed_assertions=(
        "structural instability present",
        "persistent deviation from baseline",
        "system reorganization detected",
        "persistent system reorganization is present",
        "stability remains low",
        "drift is elevated",
        "evidence is insufficient for a defensible assertion",
    ),
    forbidden_outputs=(
        "shut down",
        "reduce load",
        "increase ventilation",
        "operator should",
        "must",
        "immediately",
        "recommend",
        "action:",
    ),
    deterioration_criteria=DeteriorationCriteria(
        instability_min=0.6,
        drift_min=0.5,
    ),
    persistence_thresholds=PersistenceThresholds(
        min_consecutive_cycles=3,
        min_window_fraction=0.6,
    ),
    corroboration_thresholds=CorroborationThresholds(
        min_confirming_signals=2,
        min_confidence=0.55,
    ),
    refusal_conditions=(
        "insufficient_evidence",
        "insufficient_corroboration",
        "low_confidence",
        "prescriptive_or_actuation_language",
    ),
    uncertainty_override=UncertaintyOverrideBehavior(
        refuse_on_low_confidence=True,
        min_confidence=0.45,
        refusal_message="Evidence is currently insufficient for a defensible system-state assertion.",
    ),
    metadata={
        "intent": "Constrain outputs to defensible observations, never prescriptions.",
        "scope": "Doctrine layer only; no Gate enforcement.",
    },
)
