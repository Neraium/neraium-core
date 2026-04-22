import warnings

import numpy as np

from neraium_core.engine_stages.scoring_preparation import ScoringPreparationInput, prepare_scoring_inputs
from neraium_core.features.window_feature_extractor import summarize_feature_delta


class _DummyContext:
    _rolling_baseline_corr = None


def test_summarize_feature_delta_no_warnings_on_single_row_windows() -> None:
    baseline = np.array([[1.0, 2.0]], dtype=float)
    recent = np.array([[1.5, 2.5]], dtype=float)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", category=RuntimeWarning)
        out = summarize_feature_delta(baseline, recent)

    assert isinstance(out, dict)
    assert len(captured) == 0


def test_prepare_scoring_inputs_requires_two_observations() -> None:
    z_baseline = np.zeros((1, 2), dtype=float)
    z_recent = np.zeros((1, 2), dtype=float)
    valid_mask = np.array([True, True], dtype=bool)

    result = prepare_scoring_inputs(
        _DummyContext(),
        ScoringPreparationInput(
            z_baseline=z_baseline,
            z_recent=z_recent,
            valid_mask=valid_mask,
            valid_signal_count=2,
        ),
    )

    assert not result.should_proceed_scoring
    assert result.minimum_sample_ready is False
