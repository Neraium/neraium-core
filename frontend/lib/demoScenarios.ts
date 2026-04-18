/**
 * Demo scenarios for DecisionUI
 * Optimized 30-second flow: Baseline → Degradation → Emergency
 *
 * Hero scenario: EMERGING - shows pattern detection and actionable guidance
 */

import {
  DecisionUIState,
  Severity,
  Trajectory,
  DegradationStage,
  ActionHorizon,
} from '@/lib/decisionToUI'

export const DEMO_SCENARIO_BASELINE: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.BASELINE,
    severity: Severity.LOW,
    trajectory: Trajectory.STABLE,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.WATCHLIST,
    primaryAction: 'All nominal. Continue standard monitoring.',
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
    primaryFactor: 'System stable within baseline',
    secondaryFactors: ['Coherence stable', 'No degradation detected'],
    confidenceRationale: 'Confidence: HIGH',
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

// HERO SCENARIO: Shows Neraium's core value - early detection with pattern matching
export const DEMO_SCENARIO_EMERGING: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.EMERGING,
    severity: Severity.ELEVATED,
    trajectory: Trajectory.DEGRADING,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.NOW,
    primaryAction: 'Begin mitigation. Patterns match Q2 2023 failure event.',
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
    primaryFactor: 'Drift at 54% with structural degradation',
    secondaryFactors: ['Stability at 42%', '5 corroborating signals'],
    confidenceRationale: 'Confidence: HIGH',
    patternInsight: {
      tier: 'strong',
      outcomeType: 'structural_cascade',
      influenceSummary: 'Strong match to Q2 2023 cascade. 4-6 hour window to critical.',
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
    primaryAction: 'Activate failover immediately. Threshold breach imminent.',
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
    primaryFactor: 'Integrity critical (82%), cascade acceleration',
    secondaryFactors: ['Stability 18%', '8 cascade signals confirmed'],
    confidenceRationale: 'Confidence: HIGH',
    patternInsight: {
      tier: 'strong',
      outcomeType: 'cascade_failure',
      influenceSummary: 'Cascade confirmed. Window: 15-30 minutes to failure.',
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

// Optimized order for demo: Baseline → Emerging (hero) → Critical (escalation)
export const DEMO_SCENARIOS = [
  { name: 'Baseline', state: DEMO_SCENARIO_BASELINE },
  { name: 'Emerging', state: DEMO_SCENARIO_EMERGING },
  { name: 'Critical', state: DEMO_SCENARIO_CRITICAL },
]

// For standalone hero demo
export const HERO_SCENARIO = DEMO_SCENARIO_EMERGING
