"""Tests for stability_evaluation and staged_pipeline calibration helpers."""

import numpy as np
import pandas as pd

from neraium_core.stability_evaluation import compute_operational_stability_index
from neraium_core.staged_pipeline import (
    DecisionStage,
    MIN_BASELINE_SAMPLES_FOR_CALIBRATION,
    decide_state_with_calibration,
    decision_adjusted_score,
    state_from_node_quantiles,
)


def test_decision_adjusted_score_matches_state_from_score_internals():
    inst, conf, loc = 2.0, 0.8, 0.5
    adj = decision_adjusted_score(inst, conf, loc)
    loc_gate = 0.40 + 0.60 * loc
    conf_gate = 0.55 + 0.45 * conf
    assert adj == DecisionStage.adjusted_instability(inst, conf, loc)
    assert adj == inst * loc_gate * conf_gate


def test_quantile_state():
    assert state_from_node_quantiles(0.1, 0.5, 1.0) == "STABLE"
    assert state_from_node_quantiles(0.7, 0.5, 1.0) == "WATCH"
    assert state_from_node_quantiles(1.5, 0.5, 1.0) == "ALERT"


def test_decide_state_global_fallback_short_baseline():
    s, mode = decide_state_with_calibration(
        phase="baseline",
        adj=1.0,
        confidence=0.6,
        localization=0.3,
        dec_adj=decision_adjusted_score(1.0, 0.6, 0.3),
        baseline_dec_adj_prior=[0.1] * (MIN_BASELINE_SAMPLES_FOR_CALIBRATION - 1),
        frozen_watch_alert=None,
    )
    assert mode == "global_fallback"
    assert s in {"STABLE", "WATCH", "ALERT"}


def test_compute_operational_stability_index_basic():
    df = pd.DataFrame(
        {
            "state": ["STABLE"] * 10,
            "interpreted_state": ["NOMINAL_STRUCTURE"] * 10,
            "latest_instability": np.linspace(0.1, 0.2, 10),
            "temporal_coherence_score": [0.9] * 10,
            "confidence_score": [0.7] * 10,
            "regime_distance": [0.2] * 10,
            "nominal_consistency_score": [0.85] * 10,
        }
    )
    out = compute_operational_stability_index(df)
    assert out["operational_stability_index"] > 0.7
    assert out["nominal_false_positive_burden"] == 0.0
