"""
Unit tests for the corrected persistence-based warning rule.

What is being tested
--------------------
``neraium_core.intelligence_stack.detector.find_warning_index`` implements the corrected semantics:

    A warning fires at the index where *persistence* CONSECUTIVE exceedances
    are CONFIRMED.  This is the index of the Nth exceedance — NOT backdated
    to the first step in the run.

Why this matters
----------------
The original experiment code suffered from a backdating bug where the reported
warning index was the FIRST step in the exceedance run rather than the step
where persistence was satisfied.  This inflated apparent lead times and made
the detector look earlier than it actually was.

Corrected rule (what these tests verify)
-----------------------------------------
  persistence=3, threshold=0.5, scores=[0.1, 0.9, 0.9, 0.9]
  Exceedances at indices:  1, 2, 3
  Confirmation at:         index 3  (the 3rd consecutive)
  NOT at:                  index 1  (the first in the run) ← backdating bug

Test coverage
-------------
1. No exceedance → None
2. Scores never reach threshold → None
3. Exact run of `persistence` → correct confirmation index
4. Single-step persistence (persistence=1) → first exceedance
5. Interrupted run → warning delayed until true persistence is satisfied
6. Multiple candidate runs → fires on the first qualifying run
7. No backdating — confirmation index ≠ first-in-run index
8. Trailing non-qualifying run → None (run is cut short at end of series)
9. persistence=0 raises ValueError
10. Empty scores array → None
11. All scores exceed → confirmation at index persistence-1
12. Persistence exactly satisfied at last index → returns last index
"""

import numpy as np
import pytest

from neraium_core.intelligence_stack.detector import compute_warning_state, find_warning_index


# ─────────────────────────────────────────────────────────────────────────────
# Basic correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_no_exceedance_returns_none():
    """Scores always below threshold → never triggers."""
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    result = find_warning_index(scores, threshold=0.5, persistence=2)
    assert result is None


def test_scores_never_reach_threshold_returns_none():
    """All scores strictly below threshold."""
    scores = np.zeros(50)
    assert find_warning_index(scores, threshold=0.01, persistence=1) is None


def test_exact_persistence_run_returns_confirmation_index():
    """
    Exactly `persistence` consecutive exceedances.
    Warning should fire at the LAST of those steps (index 3 for persistence=3,
    starting at index 1).
    """
    scores = np.array([0.1, 0.9, 0.9, 0.9])
    result = find_warning_index(scores, threshold=0.5, persistence=3)
    assert result == 3, f"Expected 3, got {result}"


def test_single_persistence_fires_immediately():
    """persistence=1 → any single exceedance triggers warning."""
    scores = np.array([0.0, 0.0, 0.8, 0.0])
    result = find_warning_index(scores, threshold=0.5, persistence=1)
    assert result == 2


def test_all_scores_exceed_fires_at_persistence_minus_one():
    """All scores above threshold → confirmation at index persistence-1."""
    scores = np.ones(10) * 0.9
    result = find_warning_index(scores, threshold=0.5, persistence=4)
    assert result == 3  # 0-indexed: indices 0,1,2,3 → confirmed at 3


# ─────────────────────────────────────────────────────────────────────────────
# Interrupted runs
# ─────────────────────────────────────────────────────────────────────────────


def test_interrupted_run_delays_warning():
    """
    An exceedance run is broken by a sub-threshold value.
    The counter resets; warning fires only when a complete run is achieved.

    Scores:  [0.9, 0.9, 0.1, 0.9, 0.9, 0.9]
    persistence = 3
    First run (indices 0–1): length 2 — too short, reset at index 2
    Second run (indices 3–5): length 3 — confirmed at index 5
    """
    scores = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9])
    result = find_warning_index(scores, threshold=0.5, persistence=3)
    assert result == 5, f"Expected 5, got {result}"


def test_multiple_interruptions_fires_on_first_complete_run():
    """
    Several partial runs, then one qualifying run.
    """
    scores = np.array([0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9])
    #                         ^         ^         run start
    # Runs: [0], [2,3], [5,6,7] → persistence=3 → confirmed at index 7
    result = find_warning_index(scores, threshold=0.5, persistence=3)
    assert result == 7


def test_interrupted_run_never_triggers():
    """
    No run ever reaches persistence=3.
    """
    scores = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9])
    result = find_warning_index(scores, threshold=0.5, persistence=3)
    assert result is None


def test_trailing_partial_run_does_not_trigger():
    """
    Series ends mid-run, never satisfying persistence.
    """
    scores = np.array([0.1, 0.9, 0.9])  # only 2 consecutive at end
    result = find_warning_index(scores, threshold=0.5, persistence=3)
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Anti-backdating (core correctness requirement)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_backdating_confirmation_index_not_run_start():
    """
    CRITICAL: The returned index must be the CONFIRMATION step,
    not the FIRST step in the exceedance run.

    With persistence=3:
      - Exceedance run starts at index 2
      - Confirmation is at index 4 (the 3rd consecutive step)

    The buggy version would return 2.
    The corrected version must return 4.
    """
    scores = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    result = find_warning_index(scores, threshold=0.5, persistence=3)

    # Confirm the confirmation index — NOT the start of the run
    assert result == 4, (
        f"Expected confirmation index 4, got {result}. "
        "If result==2, the backdating bug is still present."
    )
    assert result != 2, "Backdating bug: returned first-in-run index instead of confirmation index"


def test_no_backdating_longer_run():
    """
    Longer run: confirm the returned index is always the Nth step,
    never the 1st step of the run.
    """
    # Run starts at index 5, persistence=4 → confirmation at 5+3=8
    scores = np.zeros(15)
    scores[5:] = 0.9
    result = find_warning_index(scores, threshold=0.5, persistence=4)
    assert result == 8, f"Expected 8, got {result}"
    assert result != 5, "Backdating bug detected"


def test_confirmation_index_equals_run_start_plus_persistence_minus_one():
    """
    General property: for a clean run starting at index S,
    confirmation = S + persistence - 1.
    """
    for run_start in [0, 3, 7]:
        for persistence in [1, 2, 3, 5]:
            scores = np.zeros(20)
            scores[run_start:] = 1.0
            result = find_warning_index(scores, threshold=0.5, persistence=persistence)
            expected = run_start + persistence - 1
            assert result == expected, (
                f"run_start={run_start}, persistence={persistence}: "
                f"expected {expected}, got {result}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_scores_returns_none():
    """Empty input → None (no warning)."""
    result = find_warning_index(np.array([]), threshold=0.5, persistence=1)
    assert result is None


def test_persistence_zero_raises():
    """persistence < 1 is nonsensical and must raise ValueError."""
    with pytest.raises(ValueError, match="persistence"):
        find_warning_index(np.array([0.9, 0.9]), threshold=0.5, persistence=0)


def test_persistence_negative_raises():
    with pytest.raises(ValueError):
        find_warning_index(np.array([0.9]), threshold=0.5, persistence=-1)


def test_threshold_exactly_met_counts_as_exceedance():
    """Score == threshold should count (>= not >)."""
    scores = np.array([0.5, 0.5, 0.5])
    result = find_warning_index(scores, threshold=0.5, persistence=3)
    assert result == 2


def test_score_just_below_threshold_does_not_count():
    """Score strictly below threshold → not an exceedance."""
    scores = np.array([0.499, 0.499, 0.499])
    result = find_warning_index(scores, threshold=0.5, persistence=3)
    assert result is None


def test_fires_on_first_qualifying_run_not_later():
    """
    When multiple qualifying runs exist, the FIRST one triggers.
    """
    # First qualifying run at indices 1-3; second at 6-8
    scores = np.array([0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.9])
    result = find_warning_index(scores, threshold=0.5, persistence=3)
    assert result == 3, f"Expected 3 (first qualifying run), got {result}"


def test_confirmation_at_last_index():
    """
    Run completes exactly at the last timestep — must still return the index.
    """
    scores = np.array([0.0, 0.0, 0.9, 0.9, 0.9])
    result = find_warning_index(scores, threshold=0.5, persistence=3)
    assert result == 4


def test_single_element_series():
    """One-element series, persistence=1, exceeds → 0."""
    assert find_warning_index(np.array([1.0]), 0.5, 1) == 0


def test_single_element_series_no_exceed():
    """One-element series, persistence=1, does not exceed → None."""
    assert find_warning_index(np.array([0.0]), 0.5, 1) is None


# ─────────────────────────────────────────────────────────────────────────────
# Type robustness
# ─────────────────────────────────────────────────────────────────────────────


def test_accepts_python_list_via_numpy():
    """Should work when wrapping a Python list in numpy."""
    scores = np.array([0.1, 0.8, 0.8, 0.8])
    assert find_warning_index(scores, threshold=0.5, persistence=3) == 3


def test_float32_array():
    scores = np.array([0.0, 0.9, 0.9, 0.9], dtype=np.float32)
    assert find_warning_index(scores, threshold=0.5, persistence=3) == 3


def test_hysteresis_prevents_rapid_warning_oscillation():
    scores = np.array([0.1, 0.9, 0.9, 0.9, 0.45, 0.55, 0.45, 0.2], dtype=float)
    state = compute_warning_state(
        scores,
        threshold=0.6,
        persistence=3,
        exit_threshold_ratio=0.8,  # exit < 0.48
        exit_persistence=2,
        min_anomaly_duration=3,
    )
    # Enter at index 3 and stay in warning through index 6; clear at 7 after two lows.
    assert np.where(state)[0].tolist() == [3, 4, 5, 6]


def test_min_anomaly_duration_can_be_stricter_than_persistence():
    scores = np.array([0.0, 0.7, 0.7, 0.7, 0.2], dtype=float)
    # persistence=2 is satisfied by index 2, but duration=4 delays confirmation to index 3.
    idx = find_warning_index(
        scores,
        threshold=0.6,
        persistence=2,
        min_anomaly_duration=3,
    )
    assert idx == 3


def test_dynamic_threshold_array_is_supported():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9], dtype=float)
    thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.75, 0.85], dtype=float)
    idx = find_warning_index(scores, threshold=thresholds, persistence=2)
    assert idx == 5


def test_zero_dim_threshold_array_is_treated_as_scalar():
    scores = np.array([0.1, 0.8, 0.8], dtype=float)
    idx = find_warning_index(scores, threshold=np.array(0.5), persistence=2)
    assert idx == 2


def test_clear_is_not_gated_by_min_anomaly_duration():
    scores = np.array([0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.1], dtype=float)
    state = compute_warning_state(
        scores=scores,
        threshold=0.8,
        persistence=2,
        exit_threshold_ratio=0.8,  # exit < 0.64
        exit_persistence=2,
        min_anomaly_duration=5,
    )
    # Enter at index 5, then clear at index 7 after two below-exit values.
    assert np.where(state)[0].tolist() == [5, 6]
