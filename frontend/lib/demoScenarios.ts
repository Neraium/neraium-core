/**
 * Demo scenarios for DecisionUI
 * Shows progression: Stable → Early Shift → Degradation → Escalation
 */

import {
  DecisionUIState,
  Severity,
  Trajectory,
  DegradationStage,
  ActionHorizon,
} from '@/lib/decisionToUI'

export const DEMO_SCENARIO_STABLE: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.BASELINE,
    severity: Severity.LOW,
    trajectory: Trajectory.STABLE,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.WATCHLIST,
    primaryAction: 'Monitor system metrics within normal parameters',
    urgencyLevel: 1,
  },
  tetrahedron: {
    currentPosition: { x: 0.2, y: 0.85, z: 0.15 },
    severityScalar: 0.2,
    trailPoints: Array.from({ length: 15 }, (_, i) => ({
      position: {
        x: 0.15 + i * 0.01,
        y: 0.85 - i * 0.005,
        z: 0.15 + i * 0.002,
      },
      timestamp: new Date(Date.now() - (14 - i) * 60000).toISOString(),
      frameIdx: i,
    })),
    velocity: [0.01, 0.01, 0.01],
    nearestVertex: 'AUTHORITY',
  },
  decisionTrace: {
    primaryFactor: 'All metrics within baseline parameters',
    secondaryFactors: ['Stable relational integrity', 'Consistent temporal patterns'],
    confidenceRationale: 'Confidence level: HIGH',
    patternInsight: undefined,
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: true },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Shift', isCurrent: false },
      { stage: DegradationStage.EMERGING, label: 'Emerging', isCurrent: false },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Accelerated', isCurrent: false },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure Approach', isCurrent: false },
    ],
    currentIndex: 0,
  },
  driftChart: {
    dataPoints: Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (19 - i) * 60000).toISOString(),
      driftScore: 0.15 + Math.sin(i * 0.2) * 0.05,
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 19,
  },
  timestamp: new Date().toISOString(),
}

export const DEMO_SCENARIO_EARLY_SHIFT: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.EARLY_SHIFT,
    severity: Severity.MODERATE,
    trajectory: Trajectory.UNCERTAIN,
    confidence: 'MEDIUM',
  },
  actionPanel: {
    horizon: ActionHorizon.SOON,
    primaryAction: 'Increase monitoring frequency; review system configuration changes',
    urgencyLevel: 3,
  },
  tetrahedron: {
    currentPosition: { x: 0.45, y: 0.55, z: 0.35 },
    severityScalar: 0.5,
    trailPoints: Array.from({ length: 15 }, (_, i) => ({
      position: {
        x: 0.2 + i * 0.02,
        y: 0.85 - i * 0.02,
        z: 0.15 + i * 0.012,
      },
      timestamp: new Date(Date.now() - (14 - i) * 60000).toISOString(),
      frameIdx: i,
    })),
    velocity: [0.05, 0.04, 0.03],
    nearestVertex: 'STRUCTURAL',
  },
  decisionTrace: {
    primaryFactor: 'Structural drift increased to 38%',
    secondaryFactors: [
      'Relational stability declining gradually',
      '2 corroborating signals detected',
    ],
    confidenceRationale: 'Confidence level: MEDIUM',
    patternInsight: {
      tier: 'moderate',
      outcomeType: 'gradual_drift',
      influenceSummary: 'Similar pattern observed in 3 historical events',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: false },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Shift', isCurrent: true },
      { stage: DegradationStage.EMERGING, label: 'Emerging', isCurrent: false },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Accelerated', isCurrent: false },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure Approach', isCurrent: false },
    ],
    currentIndex: 1,
  },
  driftChart: {
    dataPoints: Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (19 - i) * 60000).toISOString(),
      driftScore: 0.15 + (i * 0.012),
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 19,
  },
  timestamp: new Date().toISOString(),
}

export const DEMO_SCENARIO_EMERGING: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.EMERGING,
    severity: Severity.ELEVATED,
    trajectory: Trajectory.DEGRADING,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.NOW,
    primaryAction: 'Begin mitigation procedures; prepare contingency protocols',
    urgencyLevel: 5,
  },
  tetrahedron: {
    currentPosition: { x: 0.68, y: 0.35, z: 0.62 },
    severityScalar: 0.75,
    trailPoints: Array.from({ length: 15 }, (_, i) => ({
      position: {
        x: 0.2 + i * 0.035,
        y: 0.85 - i * 0.038,
        z: 0.15 + i * 0.035,
      },
      timestamp: new Date(Date.now() - (14 - i) * 60000).toISOString(),
      frameIdx: i,
    })),
    velocity: [0.12, 0.11, 0.08],
    nearestVertex: 'STRUCTURAL',
  },
  decisionTrace: {
    primaryFactor: 'Structural drift reached 54% with persistent degradation',
    secondaryFactors: [
      'Relational stability dropped to 42%',
      '5 corroborating signals across multiple dimensions',
    ],
    confidenceRationale: 'Confidence level: HIGH',
    patternInsight: {
      tier: 'strong',
      outcomeType: 'structural_cascade',
      influenceSummary: 'Matches failure pattern from Q2 2023 incident; escalation expected within 4-6 hours',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: false },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Shift', isCurrent: false },
      { stage: DegradationStage.EMERGING, label: 'Emerging', isCurrent: true },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Accelerated', isCurrent: false },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure Approach', isCurrent: false },
    ],
    currentIndex: 2,
  },
  driftChart: {
    dataPoints: Array.from({ length: 24 }, (_, i) => ({
      timestamp: new Date(Date.now() - (23 - i) * 60000).toISOString(),
      driftScore: Math.min(0.65, 0.15 + (i * 0.022)),
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 23,
  },
  timestamp: new Date().toISOString(),
}

export const DEMO_SCENARIO_CRITICAL: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.ACCELERATED,
    severity: Severity.HIGH,
    trajectory: Trajectory.DEGRADING,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.NOW,
    primaryAction: 'Execute emergency shutdown sequence; activate failover systems immediately',
    urgencyLevel: 5,
  },
  tetrahedron: {
    currentPosition: { x: 0.88, y: 0.15, z: 0.85 },
    severityScalar: 0.95,
    trailPoints: Array.from({ length: 20 }, (_, i) => ({
      position: {
        x: 0.2 + i * 0.035,
        y: 0.85 - i * 0.038,
        z: 0.15 + i * 0.035,
      },
      timestamp: new Date(Date.now() - (19 - i) * 60000).toISOString(),
      frameIdx: i,
    })),
    velocity: [0.24, 0.22, 0.18],
    nearestVertex: 'STRUCTURAL',
  },
  decisionTrace: {
    primaryFactor: 'Structural integrity critical at 82%; acceleration phase detected',
    secondaryFactors: [
      'Relational stability at 18% - system coherence failing',
      '8 signals confirming cascade failure mode',
    ],
    confidenceRationale: 'Confidence level: HIGH',
    patternInsight: {
      tier: 'strong',
      outcomeType: 'cascade_failure',
      influenceSummary: 'Critical threshold breach imminent; 15-30 minute window to mitigation',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: false },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Shift', isCurrent: false },
      { stage: DegradationStage.EMERGING, label: 'Emerging', isCurrent: false },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Accelerated', isCurrent: true },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure Approach', isCurrent: false },
    ],
    currentIndex: 4,
  },
  driftChart: {
    dataPoints: Array.from({ length: 24 }, (_, i) => ({
      timestamp: new Date(Date.now() - (23 - i) * 60000).toISOString(),
      driftScore: Math.min(0.85, 0.15 + (i * 0.03)),
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 23,
  },
  timestamp: new Date().toISOString(),
}

export const DEMO_SCENARIOS = [
  { name: 'Stable', state: DEMO_SCENARIO_STABLE },
  { name: 'Early Shift', state: DEMO_SCENARIO_EARLY_SHIFT },
  { name: 'Emerging', state: DEMO_SCENARIO_EMERGING },
  { name: 'Critical', state: DEMO_SCENARIO_CRITICAL },
]
