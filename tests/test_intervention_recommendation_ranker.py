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
    assert float(support_zero["confidence"]) <= 0.10
    assert float(support_one["confidence"]) <= 0.12
    assert float(support_one["confidence"]) > float(support_zero["confidence"])
