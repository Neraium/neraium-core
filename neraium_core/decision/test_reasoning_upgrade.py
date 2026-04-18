"""Tests for decision reasoning upgrade module.

Validates:
1. Causal abstraction correctness
2. Temporal phase classification
3. Uncertainty scenario handling
"""

from __future__ import annotations

import pytest

from neraium_core.decision import reasoning_upgrade as upgrade


class TestCausalAbstraction:
    """Test causal theme derivation."""

    def test_external_shock_detection(self) -> None:
        """Sudden high drift with unstable trajectory = external shock."""
        theme = upgrade.derive_causal_theme(
            drift=0.8,
            relational_stability=0.6,
            coherence=0.5,
            persistence_frames=1,
            trajectory="unstable",
        )
        assert theme == "external_shock"

    def test_relational_breakdown_detection(self) -> None:
        """Low relational stability + low coherence = relational breakdown."""
        theme = upgrade.derive_causal_theme(
            drift=0.4,
            relational_stability=0.3,
            coherence=0.4,
            persistence_frames=3,
            trajectory="degrading",
        )
        assert theme == "relational_breakdown"

    def test_multi_signal_instability(self) -> None:
        """Moderate-high drift + degrading + some persistence = multi-signal instability."""
        theme = upgrade.derive_causal_theme(
            drift=0.6,
            relational_stability=0.5,
            coherence=0.55,
            persistence_frames=4,
            trajectory="degrading",
        )
        assert theme == "multi_signal_instability"

    def test_persistent_structural_degradation(self) -> None:
        """High drift + low coherence + sustained persistence = persistent structural degradation."""
        theme = upgrade.derive_causal_theme(
            drift=0.75,
            relational_stability=0.4,
            coherence=0.45,
            persistence_frames=7,
            trajectory="degrading",
        )
        assert theme == "persistent_structural_degradation"

    def test_transient_anomaly(self) -> None:
        """Low relational stability with stable/recovering trajectory = transient anomaly."""
        theme = upgrade.derive_causal_theme(
            drift=0.2,
            relational_stability=0.3,
            coherence=0.6,
            persistence_frames=1,
            trajectory="stable",
        )
        assert theme == "transient_anomaly"

    def test_causal_theme_output_is_valid_enum(self) -> None:
        """All causal themes are valid enum values."""
        valid_themes = {
            "relational_breakdown",
            "persistent_structural_degradation",
            "multi_signal_instability",
            "transient_anomaly",
            "external_shock",
        }
        for drift in [0.2, 0.5, 0.8]:
            for rel_stability in [0.2, 0.5, 0.8]:
                for coherence in [0.3, 0.6, 0.9]:
                    theme = upgrade.derive_causal_theme(
                        drift=drift,
                        relational_stability=rel_stability,
                        coherence=coherence,
                        persistence_frames=3,
                        trajectory="degrading",
                    )
                    assert theme in valid_themes


class TestTemporalPhaseClassification:
    """Test temporal progression phase classification."""

    def test_onset_detection(self) -> None:
        """First 1-2 frames of degradation = onset."""
        phase = upgrade.classify_temporal_phase(
            persistence_frames=1,
            drift_velocity=0.0,
            trajectory="degrading",
            coherence_trend=0.0,
        )
        assert phase == "onset"

        phase = upgrade.classify_temporal_phase(
            persistence_frames=2,
            drift_velocity=0.05,
            trajectory="degrading",
            coherence_trend=0.0,
        )
        assert phase == "onset"

    def test_resolving_detection(self) -> None:
        """Improving trajectory or recovering coherence = resolving."""
        phase = upgrade.classify_temporal_phase(
            persistence_frames=3,
            drift_velocity=-0.1,
            trajectory="recovering",
            coherence_trend=0.0,
        )
        assert phase == "resolving"

        phase = upgrade.classify_temporal_phase(
            persistence_frames=5,
            drift_velocity=0.05,
            trajectory="degrading",
            coherence_trend=0.3,
        )
        assert phase == "resolving"

    def test_accelerating_detection(self) -> None:
        """Increasing drift velocity with degrading trajectory = accelerating."""
        phase = upgrade.classify_temporal_phase(
            persistence_frames=3,
            drift_velocity=0.15,
            trajectory="degrading",
            coherence_trend=0.0,
        )
        assert phase == "accelerating"

    def test_chronic_detection(self) -> None:
        """Long persistence (10+) with low acceleration = chronic."""
        phase = upgrade.classify_temporal_phase(
            persistence_frames=12,
            drift_velocity=0.02,
            trajectory="degrading",
            coherence_trend=-0.05,
        )
        assert phase == "chronic"

    def test_persistent_detection(self) -> None:
        """3-6 frames sustained at degrading = persistent."""
        phase = upgrade.classify_temporal_phase(
            persistence_frames=4,
            drift_velocity=0.08,
            trajectory="degrading",
            coherence_trend=0.0,
        )
        assert phase == "persistent"

    def test_temporal_phase_output_is_valid_enum(self) -> None:
        """All temporal phases are valid enum values."""
        valid_phases = {"onset", "persistent", "accelerating", "chronic", "resolving"}
        for frames in [1, 3, 6, 10, 15]:
            for velocity in [-0.2, -0.05, 0.0, 0.05, 0.15]:
                for traj in ["stable", "degrading", "recovering", "unstable"]:
                    phase = upgrade.classify_temporal_phase(
                        persistence_frames=frames,
                        drift_velocity=velocity,
                        trajectory=traj,
                        coherence_trend=0.0,
                    )
                    assert phase in valid_phases


class TestUncertaintyContextClassification:
    """Test uncertainty handling under partial evidence."""

    def test_high_confidence_scenario(self) -> None:
        """Strong persistence + degrading + good pattern match = high confidence."""
        context = upgrade.classify_uncertainty_context(
            finding_confidence=0.85,
            persistence_frames=4,
            trajectory="degrading",
            pattern_match_tier="strong",
            causal_chain_strength=0.8,
        )
        assert context == "high_confidence"

    def test_low_evidence_scenario(self) -> None:
        """Weak confidence or transient pattern = low evidence."""
        context = upgrade.classify_uncertainty_context(
            finding_confidence=0.3,
            persistence_frames=1,
            trajectory="stable",
            pattern_match_tier="weak",
            causal_chain_strength=0.3,
        )
        assert context == "low_evidence"

    def test_unstable_trajectory_scenario(self) -> None:
        """Unstable trajectory classification overrides others."""
        context = upgrade.classify_uncertainty_context(
            finding_confidence=0.9,
            persistence_frames=5,
            trajectory="unstable",
            pattern_match_tier="strong",
            causal_chain_strength=0.9,
        )
        assert context == "unstable_trajectory"

    def test_conflicting_signals_scenario(self) -> None:
        """Moderate confidence + low persistence + non-degrading = conflicting."""
        context = upgrade.classify_uncertainty_context(
            finding_confidence=0.6,
            persistence_frames=1,
            trajectory="stable",
            pattern_match_tier="moderate",
            causal_chain_strength=0.5,
        )
        assert context == "conflicting_signals"

    def test_early_noise_scenario(self) -> None:
        """Early noise: high confidence but only 1-2 frames, weak pattern."""
        context = upgrade.classify_uncertainty_context(
            finding_confidence=0.7,
            persistence_frames=1,
            trajectory="degrading",
            pattern_match_tier="weak",
            causal_chain_strength=0.4,
        )
        assert context == "low_evidence"

    def test_uncertainty_context_output_is_valid_enum(self) -> None:
        """All uncertainty contexts are valid enum values."""
        valid_contexts = {
            "low_evidence",
            "conflicting_signals",
            "unstable_trajectory",
            "high_confidence",
        }
        for conf in [0.2, 0.5, 0.8]:
            for frames in [1, 3, 5, 10]:
                for traj in ["stable", "degrading", "recovering", "unstable"]:
                    context = upgrade.classify_uncertainty_context(
                        finding_confidence=conf,
                        persistence_frames=frames,
                        trajectory=traj,
                        pattern_match_tier="moderate",
                        causal_chain_strength=0.5,
                    )
                    assert context in valid_contexts


class TestFormattingFunctions:
    """Test human-readable formatting of reasoning."""

    def test_causal_summary_formatting(self) -> None:
        """Causal themes format to descriptive summaries."""
        summary = upgrade.format_causal_summary("relational_breakdown")
        assert "Relational breakdown" in summary

        summary = upgrade.format_causal_summary("external_shock")
        assert "External shock" in summary

    def test_temporal_context_formatting(self) -> None:
        """Temporal phases format to descriptive context."""
        context = upgrade.format_temporal_context("onset", 1)
        assert "emerging" in context.lower()

        context = upgrade.format_temporal_context("persistent", 5)
        assert "persistent" in context.lower() and "5" in context

        context = upgrade.format_temporal_context("accelerating", 4)
        assert "accelerat" in context.lower()

    def test_uncertainty_rationale_formatting(self) -> None:
        """Uncertainty contexts format to rationale strings."""
        rationale = upgrade.format_uncertainty_rationale("high_confidence", 0.9, 5)
        assert "High confidence" in rationale

        rationale = upgrade.format_uncertainty_rationale("low_evidence", 0.3, 1)
        assert "pattern weak" in rationale.lower() or "live evidence" in rationale.lower()

        rationale = upgrade.format_uncertainty_rationale("conflicting_signals", 0.6, 2)
        assert "unstable" in rationale.lower() or "emerging" in rationale.lower()
