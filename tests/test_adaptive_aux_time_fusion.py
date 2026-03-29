"""Adaptive AUX-TIME fusion coherence (Combined SII + AUX-TIME path)."""

from neraium_core.staged_pipeline import adaptive_aux_time_fusion_coherence


def test_adaptive_raises_effective_coherence_when_distortion_high_and_tc_moderate():
    tc = 0.55
    g = 0.85
    out = adaptive_aux_time_fusion_coherence(tc, g, enabled=True)
    assert out > tc
    assert out <= 1.0


def test_adaptive_disabled_passthrough():
    tc = 0.4
    assert adaptive_aux_time_fusion_coherence(tc, 1.0, enabled=False) == tc


def test_adaptive_no_change_when_already_coherent():
    tc = 0.98
    g = 0.9
    out = adaptive_aux_time_fusion_coherence(tc, g, enabled=True)
    assert abs(out - tc) < 0.02
