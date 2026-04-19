/**
 * Datacenter Cooling System Demo Scenarios
 * Shows realistic CRAC unit degradation through all stages
 */

import {
  DecisionUIState,
  Severity,
  Trajectory,
  DegradationStage,
  ActionHorizon,
} from '@/lib/decisionToUI'

const DEMO_BASE_TIME = new Date('2026-04-18T03:00:00.000Z').getTime()

function isoAtOffset(minutesOffset: number): string {
  return new Date(DEMO_BASE_TIME - minutesOffset * 60000).toISOString()
}

export const COOLING_BASELINE: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.BASELINE,
    severity: Severity.LOW,
    trajectory: Trajectory.STABLE,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.WATCHLIST,
    primaryAction: 'Continue routine monitoring',
    urgencyLevel: 1,
  },
  tetrahedron: {
    currentPosition: { x: 0.18, y: 0.88, z: 0.12 },
    severityScalar: 0.15,
    trailPoints: Array.from({ length: 16 }, (_, i) => ({
      position: {
        x: 0.16 + i * 0.006,
        y: 0.88 - i * 0.002,
        z: 0.12 + i * 0.001,
      },
      timestamp: isoAtOffset(15 - i),
      frameIdx: i,
    })),
    velocity: [0.004, 0.002, 0.001],
    nearestVertex: 'AUTHORITY',
  },
  decisionTrace: {
    primaryFactor: 'Cooling efficiency nominal',
    secondaryFactors: ['Fan speed optimal', 'Discharge temp stable at 18°C', 'No airflow anomalies'],
    confidenceRationale: 'Confidence high',
    patternInsight: {
      tier: 'moderate',
      outcomeType: 'nominal',
      influenceSummary: 'System operating within design parameters',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: true },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Drift', isCurrent: false },
      { stage: DegradationStage.EMERGING, label: 'Structural Shift', isCurrent: false },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Pre-instability', isCurrent: false },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure', isCurrent: false },
    ],
    currentIndex: 0,
  },
  driftChart: {
    dataPoints: Array.from({ length: 22 }, (_, i) => ({
      timestamp: isoAtOffset(21 - i),
      driftScore: 0.08 + Math.sin(i * 0.2) * 0.02,
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 21,
  },
  timestamp: new Date(DEMO_BASE_TIME).toISOString(),
}

export const COOLING_EARLY_DRIFT: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.EARLY_SHIFT,
    severity: Severity.MODERATE,
    trajectory: Trajectory.UNCERTAIN,
    confidence: 'MEDIUM',
  },
  actionPanel: {
    horizon: ActionHorizon.SOON,
    primaryAction: 'Inspect fan bearings and filter condition',
    urgencyLevel: 3,
  },
  tetrahedron: {
    currentPosition: { x: 0.42, y: 0.68, z: 0.31 },
    severityScalar: 0.45,
    trailPoints: Array.from({ length: 16 }, (_, i) => ({
      position: {
        x: 0.18 + i * 0.015,
        y: 0.87 - i * 0.011,
        z: 0.13 + i * 0.011,
      },
      timestamp: isoAtOffset(15 - i),
      frameIdx: i,
    })),
    velocity: [0.025, 0.022, 0.018],
    nearestVertex: 'TEMPORAL',
  },
  decisionTrace: {
    primaryFactor: 'Cooling efficiency degrading over last 48 hours',
    secondaryFactors: [
      'Fan operating hours increased 12%',
      'Discharge temperature rising: 18°C → 21°C',
      'Airflow-temperature correlation weakening',
    ],
    confidenceRationale: 'Confidence medium - pattern emerging but not fully established',
    patternInsight: {
      tier: 'moderate',
      outcomeType: 'efficiency_loss',
      influenceSummary: 'Likely causes: filter loading, bearing wear, or coolant flow restriction',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: false },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Drift', isCurrent: true },
      { stage: DegradationStage.EMERGING, label: 'Structural Shift', isCurrent: false },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Pre-instability', isCurrent: false },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure', isCurrent: false },
    ],
    currentIndex: 1,
  },
  driftChart: {
    dataPoints: Array.from({ length: 24 }, (_, i) => ({
      timestamp: isoAtOffset(23 - i),
      driftScore: 0.09 + i * 0.009,
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 23,
  },
  timestamp: new Date(DEMO_BASE_TIME).toISOString(),
}

export const COOLING_STRUCTURAL_SHIFT: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.EMERGING,
    severity: Severity.ELEVATED,
    trajectory: Trajectory.DEGRADING,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.NOW,
    primaryAction: 'Schedule immediate maintenance - check compressor and coolant loop',
    urgencyLevel: 4,
  },
  tetrahedron: {
    currentPosition: { x: 0.65, y: 0.42, z: 0.58 },
    severityScalar: 0.72,
    trailPoints: Array.from({ length: 18 }, (_, i) => ({
      position: {
        x: 0.22 + i * 0.022,
        y: 0.85 - i * 0.023,
        z: 0.15 + i * 0.023,
      },
      timestamp: isoAtOffset(17 - i),
      frameIdx: i,
    })),
    velocity: [0.072, 0.065, 0.058],
    nearestVertex: 'STRUCTURAL',
  },
  decisionTrace: {
    primaryFactor: 'Structural relationships weakening - correlation breakdown between airflow and temperature',
    secondaryFactors: [
      'Discharge temp now 26°C despite higher fan speed',
      'Coolant loop pressure fluctuating (variance 40%)',
      '3 corroborating signals detected simultaneously',
    ],
    confidenceRationale: 'Confidence high - strong pattern match indicates imminent failure mode',
    patternInsight: {
      tier: 'strong',
      outcomeType: 'compressor_degradation',
      influenceSummary: 'Historical data suggests 8-16 hour window before critical failure',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: false },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Drift', isCurrent: false },
      { stage: DegradationStage.EMERGING, label: 'Structural Shift', isCurrent: true },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Pre-instability', isCurrent: false },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure', isCurrent: false },
    ],
    currentIndex: 2,
  },
  driftChart: {
    dataPoints: Array.from({ length: 26 }, (_, i) => ({
      timestamp: isoAtOffset(25 - i),
      driftScore: Math.min(0.68, 0.12 + i * 0.022),
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 25,
  },
  timestamp: new Date(DEMO_BASE_TIME).toISOString(),
}

export const COOLING_PRE_INSTABILITY: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.ACCELERATED,
    severity: Severity.HIGH,
    trajectory: Trajectory.DEGRADING,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.NOW,
    primaryAction: 'CRITICAL: Activate redundant cooling immediately',
    urgencyLevel: 5,
  },
  tetrahedron: {
    currentPosition: { x: 0.82, y: 0.28, z: 0.78 },
    severityScalar: 0.92,
    trailPoints: Array.from({ length: 20 }, (_, i) => ({
      position: {
        x: 0.28 + i * 0.027,
        y: 0.83 - i * 0.028,
        z: 0.18 + i * 0.03,
      },
      timestamp: isoAtOffset(19 - i),
      frameIdx: i,
    })),
    velocity: [0.14, 0.13, 0.12],
    nearestVertex: 'STRUCTURAL',
  },
  decisionTrace: {
    primaryFactor: 'Trajectory indicates emerging risk - rapid decompensation detected',
    secondaryFactors: [
      'Discharge temperature reached critical: 32°C',
      'Coolant loop pressure at 85% of safe limit',
      'Vibration sensors flagging bearing distress',
      '6 simultaneous warning conditions',
    ],
    confidenceRationale: 'Confidence high - multiple independent systems confirming failure progression',
    patternInsight: {
      tier: 'strong',
      outcomeType: 'imminent_compressor_failure',
      influenceSummary: 'Failure window narrowing: 2-4 hours estimated before thermal runaway',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: false },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Drift', isCurrent: false },
      { stage: DegradationStage.EMERGING, label: 'Structural Shift', isCurrent: false },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Pre-instability', isCurrent: true },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure', isCurrent: false },
    ],
    currentIndex: 4,
  },
  driftChart: {
    dataPoints: Array.from({ length: 28 }, (_, i) => ({
      timestamp: isoAtOffset(27 - i),
      driftScore: Math.min(0.88, 0.15 + i * 0.026),
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 27,
  },
  timestamp: new Date(DEMO_BASE_TIME).toISOString(),
}

export const COOLING_DEMO_SCENARIOS = [
  {
    name: 'CRAC-Unit-North-01 (Baseline)',
    state: COOLING_BASELINE,
    durationMs: 14000,
    narrative: 'Datacenter cooling operating at design efficiency',
  },
  {
    name: 'CRAC-Unit-North-01 (Early Drift)',
    state: COOLING_EARLY_DRIFT,
    durationMs: 14000,
    narrative: 'Cooling efficiency degrading - early signs of mechanical wear',
  },
  {
    name: 'CRAC-Unit-North-01 (Structural Shift)',
    state: COOLING_STRUCTURAL_SHIFT,
    durationMs: 16000,
    narrative: 'Structural relationships weakening - imminent failure risk detected',
  },
  {
    name: 'CRAC-Unit-North-01 (Pre-instability)',
    state: COOLING_PRE_INSTABILITY,
    durationMs: 14000,
    narrative: 'System approaching failure conditions - immediate action required',
  },
]
