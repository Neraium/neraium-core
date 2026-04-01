from __future__ import annotations

from neraium_core.system_intelligence.intervention_ranking.recommender import InterventionRecommendationRanker


def test_low_support_confidence_not_flat_and_responds_to_inputs() -> None:
    ranker = InterventionRecommendationRanker()

    weaker = ranker.compute_final_confidence(
        composite=0.18,
        support=0,
        context_match=0.25,
        harmful_strength=0.0,
        novelty=0.65,
        reliability=0.42,
        drift_warning=False,
    )
    stronger = ranker.compute_final_confidence(
        composite=0.72,
        support=0,
        context_match=0.25,
        harmful_strength=0.0,
        novelty=0.12,
        reliability=0.91,
        drift_warning=False,
    )

    assert weaker["confidence_regime"] == "uncertain"
    assert stronger["confidence_regime"] == "uncertain"
    assert stronger["confidence"] > weaker["confidence"]
    assert round(float(stronger["confidence"]), 6) != round(float(weaker["confidence"]), 6)


def test_low_support_confidence_is_still_bounded_and_conservative() -> None:
    ranker = InterventionRecommendationRanker()

    support_zero = ranker.compute_final_confidence(
        composite=0.95,
        support=0,
        context_match=1.0,
        harmful_strength=0.0,
        novelty=0.0,
        reliability=1.0,
        drift_warning=False,
    )
    support_one = ranker.compute_final_confidence(
        composite=0.95,
        support=1,
        context_match=1.0,
        harmful_strength=0.0,
        novelty=0.0,
        reliability=1.0,
        drift_warning=False,
    )

    assert support_zero["confidence_regime"] == "uncertain"
    assert support_one["confidence_regime"] == "uncertain"
    assert float(support_zero["confidence"]) <= 0.14
    assert float(support_one["confidence"]) <= 0.18
    assert float(support_one["confidence"]) > float(support_zero["confidence"])


def test_normal_regime_calibration_supports_higher_confidence_with_strong_evidence() -> None:
    ranker = InterventionRecommendationRanker()
    out = ranker.compute_final_confidence(
        composite=0.82,
        support=6,
        context_match=0.85,
        harmful_strength=0.02,
        novelty=0.1,
        reliability=0.95,
        drift_warning=False,
    )
    assert out["confidence_regime"] == "normal"
    assert float(out["confidence"]) >= 0.55
