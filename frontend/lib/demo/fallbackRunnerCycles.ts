import type { RunnerCycleRecord } from '@/lib/runnerOutputLoader'

// Small deterministic fallback dataset with a believable grow escalation arc.
// Uses the same contract as runner-derived cycles.
export const FALLBACK_RUNNER_CYCLES: RunnerCycleRecord[] = [
  { unit_id: 'unit_demo', cycle: 1, structural_drift_score: 0.05, composite_instability: 0.08, risk_level: 'LOW', phase: 'stable', trend: 'stable', dominant_driver: 'climate', operator_message: 'Stable conditions' },
  { unit_id: 'unit_demo', cycle: 2, structural_drift_score: 0.09, composite_instability: 0.10, risk_level: 'LOW', phase: 'stable', trend: 'stable', dominant_driver: 'climate', operator_message: 'Stable conditions' },
  { unit_id: 'unit_demo', cycle: 3, structural_drift_score: 0.16, composite_instability: 0.14, risk_level: 'LOW', phase: 'stable', trend: 'stable', dominant_driver: 'airflow', operator_message: 'Minor airflow variance' },
  { unit_id: 'unit_demo', cycle: 4, structural_drift_score: 0.22, composite_instability: 0.18, risk_level: 'LOW', phase: 'stable', trend: 'stable', dominant_driver: 'airflow', operator_message: 'Airflow variance' },
  { unit_id: 'unit_demo', cycle: 5, structural_drift_score: 0.31, composite_instability: 0.28, risk_level: 'MEDIUM', phase: 'watch', trend: 'rising', dominant_driver: 'airflow', operator_message: 'Airflow imbalance' },
  { unit_id: 'unit_demo', cycle: 6, structural_drift_score: 0.38, composite_instability: 0.34, risk_level: 'MEDIUM', phase: 'watch', trend: 'rising', dominant_driver: 'airflow', operator_message: 'Airflow imbalance' },
  { unit_id: 'unit_demo', cycle: 7, structural_drift_score: 0.46, composite_instability: 0.49, risk_level: 'MEDIUM', phase: 'watch', trend: 'rising', dominant_driver: 'airflow', operator_message: 'Airflow imbalance' },
  { unit_id: 'unit_demo', cycle: 8, structural_drift_score: 0.55, composite_instability: 0.62, risk_level: 'HIGH', phase: 'alert', trend: 'worsening', dominant_driver: 'airflow', operator_message: 'Airflow imbalance' },
  { unit_id: 'unit_demo', cycle: 9, structural_drift_score: 0.62, composite_instability: 0.71, risk_level: 'HIGH', phase: 'alert', trend: 'worsening', dominant_driver: 'airflow', operator_message: 'Airflow imbalance' },
  { unit_id: 'unit_demo', cycle: 10, structural_drift_score: 0.70, composite_instability: 0.78, risk_level: 'HIGH', phase: 'critical', trend: 'worsening', dominant_driver: 'airflow', operator_message: 'Flow instability', divergence_alert: true },
  { unit_id: 'unit_demo', cycle: 11, structural_drift_score: 0.66, composite_instability: 0.70, risk_level: 'HIGH', phase: 'alert', trend: 'stabilizing', dominant_driver: 'airflow', operator_message: 'Flow instability', divergence_alert: true },
  { unit_id: 'unit_demo', cycle: 12, structural_drift_score: 0.58, composite_instability: 0.58, risk_level: 'MEDIUM', phase: 'watch', trend: 'stabilizing', dominant_driver: 'climate', operator_message: 'Climate instability' },
  { unit_id: 'unit_demo', cycle: 13, structural_drift_score: 0.43, composite_instability: 0.42, risk_level: 'MEDIUM', phase: 'watch', trend: 'stable', dominant_driver: 'climate', operator_message: 'Climate instability' },
  { unit_id: 'unit_demo', cycle: 14, structural_drift_score: 0.28, composite_instability: 0.26, risk_level: 'LOW', phase: 'stable', trend: 'recovering', dominant_driver: 'climate', operator_message: 'Stable conditions' },
  { unit_id: 'unit_demo', cycle: 15, structural_drift_score: 0.12, composite_instability: 0.14, risk_level: 'LOW', phase: 'stable', trend: 'stable', dominant_driver: 'climate', operator_message: 'Stable conditions' },
]

