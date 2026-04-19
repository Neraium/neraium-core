from apps.api.routers.demo import _canonical_demo_stage


def test_canonical_demo_stage_preserves_high_risk_alert_after_120_frames() -> None:
    stage = _canonical_demo_stage(frames_processed=120, risk_level="HIGH", run_status="running")
    assert stage == "alert"


def test_canonical_demo_stage_preserves_medium_risk_drift_after_120_frames() -> None:
    stage = _canonical_demo_stage(frames_processed=180, risk_level="MEDIUM", run_status="running")
    assert stage == "early_structural_drift"


def test_canonical_demo_stage_uses_instability_gate_for_low_risk_late_run() -> None:
    stage = _canonical_demo_stage(frames_processed=130, risk_level="LOW", run_status="running")
    assert stage == "instability"
