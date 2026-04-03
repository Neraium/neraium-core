"""
Runtime calibration profiles for StructuralEngine thresholds.

Defaults match the bundled ``balanced`` profile (historical run_engine behavior).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class CalibrationProfile:
    """Tunable thresholds for drift, transition, early warning, and onset refinement."""

    name: str

    # ------------------------------------------------------------------
    # _alert_state (quantile drift thresholds)
    # ------------------------------------------------------------------
    drift_watch_quantile: float = 0.86
    drift_alert_quantile: float = 0.945
    drift_watch_floor_min: float = 1.35
    drift_watch_band_add: float = 0.12
    drift_alert_gap_min: float = 0.22
    drift_alert_band_add: float = 0.15
    drift_cold_watch: float = 2.05
    drift_cold_alert: float = 3.45
    drift_alert_threshold: float = 1.5

    # ------------------------------------------------------------------
    # _compute_transition_metrics
    # ------------------------------------------------------------------
    trans_evidence_persist_min: float = 0.58
    trans_sustained_pressure: float = 0.82
    trans_sustained_evidence_persist: float = 0.52
    trans_emerging_pressure: float = 0.56
    trans_emerging_evidence_persist: float = 0.38
    trans_metastable_min: float = 0.38

    # ------------------------------------------------------------------
    # _compute_early_warning entity pool
    # ------------------------------------------------------------------
    ew_entity_drift_max: float = 1.3
    ew_entity_pressure_max: float = 0.4
    ew_entity_lock_max: float = 0.32

    # EW reason tags
    ew_reason_stability_erosion: float = 0.5
    ew_reason_coherence: float = 0.48
    ew_reason_pressure_persist: float = 0.55
    ew_reason_recovery_fall: float = 0.52
    ew_reason_directional_fall: float = 0.52
    ew_reason_branch_pressure: float = 0.48

    # EW state thresholds
    ew_alert_pre: float = 0.86
    ew_alert_commit: float = 0.68
    ew_emerging_pre: float = 0.64
    ew_emerging_mid: float = 0.48
    ew_emerging_short: float = 0.6
    ew_pre_pre: float = 0.46
    ew_pre_erosion: float = 0.4

    # ------------------------------------------------------------------
    # _state_from_onset_evidence (quantile tails when history long)
    # ------------------------------------------------------------------
    onset_hist_quantile_onset: float = 0.76
    onset_hist_add_onset: float = 0.025
    onset_hist_quantile_watch: float = 0.68
    onset_hist_add_watch: float = 0.02
    onset_cold_onset: float = 0.5
    onset_cold_watch: float = 0.44
    onset_pressure_tail_gate: float = 0.5

    onset_alert_drift_dev: float = 0.38
    onset_alert_drift_persist: float = 0.46
    onset_alert_pressure_persist: float = 0.14
    onset_alert_low_freq: float = 0.38
    onset_alert_directional: float = 0.58
    onset_alert_agreement: float = 0.48
    onset_alert_smoothness: float = 0.28

    onset_watch_evidence_drift_persist: float = 0.38
    onset_watch_evidence_directional: float = 0.5
    onset_watch_evidence_smoothness: float = 0.24

    state_watch_drift_persist: float = 0.4
    state_watch_directional: float = 0.54

    sustained_watch_min_frames: int = 4

    energy_directional_min: float = 0.5
    sensor_shift_drift_persist: float = 0.3
    sensor_shift_directional: float = 0.45

    alert_candidate_streak_min: int = 2
    alert_recent_candidates_min: int = 2
    sensor_shift_streak_min: int = 2

    # ------------------------------------------------------------------
    # process_frame refinement and pressure boost
    # ------------------------------------------------------------------
    pressure_boost_from_erosion_scale: float = 0.1

    ew_state_window_frames: int = 6
    ew_emerging_persist_min: int = 4
    ew_pre_instability_count_required: int = 4


def _field_names() -> set[str]:
    return {f.name for f in fields(CalibrationProfile)}


def get_calibration_profile(name: str) -> CalibrationProfile:
    """Return a named profile. Unknown names fall back to ``balanced``."""
    n = (name or "balanced").strip().lower()
    if n == "conservative":
        return CalibrationProfile(
            name="conservative",
            drift_watch_quantile=0.88,
            drift_alert_quantile=0.95,
            drift_watch_floor_min=1.55,
            drift_watch_band_add=0.14,
            drift_alert_gap_min=0.26,
            drift_alert_band_add=0.17,
            drift_cold_watch=2.35,
            drift_cold_alert=3.65,
            drift_alert_threshold=1.6,
            trans_evidence_persist_min=0.63,
            trans_sustained_pressure=0.87,
            trans_sustained_evidence_persist=0.58,
            trans_emerging_pressure=0.62,
            trans_emerging_evidence_persist=0.44,
            trans_metastable_min=0.46,
            ew_entity_drift_max=1.2,
            ew_entity_pressure_max=0.35,
            ew_entity_lock_max=0.28,
            ew_reason_stability_erosion=0.55,
            ew_reason_coherence=0.53,
            ew_reason_pressure_persist=0.6,
            ew_reason_recovery_fall=0.57,
            ew_reason_directional_fall=0.57,
            ew_reason_branch_pressure=0.53,
            ew_alert_pre=0.9,
            ew_alert_commit=0.73,
            ew_emerging_pre=0.69,
            ew_emerging_mid=0.53,
            ew_emerging_short=0.65,
            ew_pre_pre=0.51,
            ew_pre_erosion=0.45,
            onset_hist_quantile_onset=0.78,
            onset_hist_add_onset=0.03,
            onset_hist_quantile_watch=0.7,
            onset_hist_add_watch=0.025,
            onset_cold_onset=0.55,
            onset_cold_watch=0.48,
            onset_pressure_tail_gate=0.55,
            onset_alert_drift_dev=0.43,
            onset_alert_drift_persist=0.51,
            onset_alert_pressure_persist=0.18,
            onset_alert_low_freq=0.43,
            onset_alert_directional=0.62,
            onset_alert_agreement=0.52,
            onset_alert_smoothness=0.32,
            onset_watch_evidence_drift_persist=0.43,
            onset_watch_evidence_directional=0.54,
            state_watch_drift_persist=0.45,
            state_watch_directional=0.58,
            sustained_watch_min_frames=5,
            energy_directional_min=0.55,
            sensor_shift_drift_persist=0.35,
            sensor_shift_directional=0.5,
            alert_candidate_streak_min=3,
            alert_recent_candidates_min=3,
            sensor_shift_streak_min=3,
            pressure_boost_from_erosion_scale=0.06,
            ew_emerging_persist_min=5,
            ew_pre_instability_count_required=5,
        )
    if n == "early_warning":
        return CalibrationProfile(
            name="early_warning",
            drift_watch_quantile=0.82,
            drift_alert_quantile=0.93,
            drift_watch_floor_min=1.15,
            drift_watch_band_add=0.1,
            drift_alert_gap_min=0.2,
            drift_alert_band_add=0.14,
            drift_cold_watch=1.85,
            drift_cold_alert=3.1,
            drift_alert_threshold=1.4,
            trans_evidence_persist_min=0.54,
            trans_sustained_pressure=0.78,
            trans_sustained_evidence_persist=0.48,
            trans_emerging_pressure=0.52,
            trans_emerging_evidence_persist=0.34,
            trans_metastable_min=0.32,
            ew_entity_drift_max=1.45,
            ew_entity_pressure_max=0.45,
            ew_entity_lock_max=0.36,
            ew_reason_stability_erosion=0.45,
            ew_reason_coherence=0.43,
            ew_reason_pressure_persist=0.5,
            ew_reason_recovery_fall=0.47,
            ew_reason_directional_fall=0.47,
            ew_reason_branch_pressure=0.43,
            ew_alert_pre=0.82,
            ew_alert_commit=0.64,
            ew_emerging_pre=0.6,
            ew_emerging_mid=0.44,
            ew_emerging_short=0.56,
            ew_pre_pre=0.42,
            ew_pre_erosion=0.36,
            onset_hist_quantile_onset=0.74,
            onset_hist_add_onset=0.03,
            onset_hist_quantile_watch=0.66,
            onset_hist_add_watch=0.025,
            onset_cold_onset=0.46,
            onset_cold_watch=0.4,
            onset_pressure_tail_gate=0.45,
            onset_alert_drift_dev=0.34,
            onset_alert_drift_persist=0.42,
            onset_alert_pressure_persist=0.12,
            onset_alert_low_freq=0.34,
            onset_alert_directional=0.54,
            onset_alert_agreement=0.44,
            onset_alert_smoothness=0.24,
            onset_watch_evidence_drift_persist=0.34,
            onset_watch_evidence_directional=0.46,
            state_watch_drift_persist=0.36,
            state_watch_directional=0.5,
            sustained_watch_min_frames=3,
            energy_directional_min=0.45,
            sensor_shift_drift_persist=0.26,
            sensor_shift_directional=0.4,
            alert_candidate_streak_min=2,
            alert_recent_candidates_min=2,
            sensor_shift_streak_min=2,
            pressure_boost_from_erosion_scale=0.12,
            ew_emerging_persist_min=3,
            ew_pre_instability_count_required=3,
        )

    return CalibrationProfile(name="balanced")


def resolve_calibration(
    *,
    calibration_profile: str = "balanced",
    calibration: CalibrationProfile | dict[str, Any] | None = None,
) -> CalibrationProfile:
    """
    Resolve final profile: named base plus optional overrides.

    ``calibration`` overrides must use only valid ``CalibrationProfile`` field names.
    """
    base = get_calibration_profile(calibration_profile)
    if calibration is None:
        return base
    if isinstance(calibration, CalibrationProfile):
        return calibration
    if isinstance(calibration, dict):
        names = _field_names()
        filtered = {k: v for k, v in calibration.items() if k in names and k != "name"}
        return replace(base, **filtered)
    raise TypeError("calibration must be CalibrationProfile, dict, or None")
