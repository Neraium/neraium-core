/**
 * Demo scenarios for DecisionUI
 * Narrative flow: Baseline → Early Shift → Emerging → Critical
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

export const DEMO_SCENARIO_BASELINE: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.BASELINE,
    severity: Severity.LOW,
    trajectory: Trajectory.STABLE,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.WATCHLIST,
    primaryAction: 'Continue monitoring',
    urgencyLevel: 1,
  },
  tetrahedron: {
    currentPosition: { x: 0.2, y: 0.85, z: 0.15 },
    severityScalar: 0.2,
    trailPoints: Array.from({ length: 16 }, (_, i) => ({
      position: {
        x: 0.15 + i * 0.008,
        y: 0.85 - i * 0.004,
        z: 0.15 + i * 0.002,
      },
      timestamp: isoAtOffset(15 - i),
      frameIdx: i,
    })),
    velocity: [0.008, 0.006, 0.004],
    nearestVertex: 'AUTHORITY',
  },
  decisionTrace: {
    primaryFactor: 'System stable',
    secondaryFactors: ['Coherence steady', 'No drift signal'],
    confidenceRationale: 'Confidence high',
    patternInsight: {
      tier: 'moderate',
      outcomeType: 'nominal',
      influenceSummary: 'Action window remains open',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: true },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Shift', isCurrent: false },
      { stage: DegradationStage.EMERGING, label: 'Emerging', isCurrent: false },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Accelerated', isCurrent: false },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure', isCurrent: false },
    ],
    currentIndex: 0,
  },
  driftChart: {
    dataPoints: Array.from({ length: 22 }, (_, i) => ({
      timestamp: isoAtOffset(21 - i),
      driftScore: 0.13 + Math.sin(i * 0.18) * 0.03,
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 21,
  },
  timestamp: new Date(DEMO_BASE_TIME).toISOString(),
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
    primaryAction: 'Schedule inspection',
    urgencyLevel: 3,
  },
  tetrahedron: {
    currentPosition: { x: 0.45, y: 0.65, z: 0.34 },
    severityScalar: 0.48,
    trailPoints: Array.from({ length: 16 }, (_, i) => ({
      position: {
        x: 0.2 + i * 0.017,
        y: 0.82 - i * 0.012,
        z: 0.16 + i * 0.011,
      },
      timestamp: isoAtOffset(15 - i),
      frameIdx: i,
    })),
    velocity: [0.03, 0.025, 0.02],
    nearestVertex: 'TEMPORAL',
  },
  decisionTrace: {
    primaryFactor: 'Early drift detected',
    secondaryFactors: ['Drift rising', 'Stability softening'],
    confidenceRationale: 'Confidence medium',
    patternInsight: {
      tier: 'moderate',
      outcomeType: 'emerging_shift',
      influenceSummary: 'Failure window: stable',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: false },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Shift', isCurrent: true },
      { stage: DegradationStage.EMERGING, label: 'Emerging', isCurrent: false },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Accelerated', isCurrent: false },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure', isCurrent: false },
    ],
    currentIndex: 1,
  },
  driftChart: {
    dataPoints: Array.from({ length: 24 }, (_, i) => ({
      timestamp: isoAtOffset(23 - i),
      driftScore: 0.14 + i * 0.01,
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 23,
  },
  timestamp: new Date(DEMO_BASE_TIME).toISOString(),
}

export const DEMO_SCENARIO_EMERGING: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.PERSISTENT,
    severity: Severity.ELEVATED,
    trajectory: Trajectory.DEGRADING,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.NOW,
    primaryAction: 'Escalate to operations',
    urgencyLevel: 4,
  },
  tetrahedron: {
    currentPosition: { x: 0.68, y: 0.38, z: 0.62 },
    severityScalar: 0.76,
    trailPoints: Array.from({ length: 18 }, (_, i) => ({
      position: {
        x: 0.25 + i * 0.024,
        y: 0.8 - i * 0.025,
        z: 0.2 + i * 0.024,
      },
      timestamp: isoAtOffset(17 - i),
      frameIdx: i,
    })),
    velocity: [0.08, 0.07, 0.06],
    nearestVertex: 'STRUCTURAL',
  },
  decisionTrace: {
    primaryFactor: 'Degradation persists',
    secondaryFactors: ['Pattern match strong', 'Corroborating signals rising'],
    confidenceRationale: 'Confidence high',
    patternInsight: {
      tier: 'strong',
      outcomeType: 'structural_cascade',
      influenceSummary: 'Estimated impact window: 4–6 hours',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: false },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Shift', isCurrent: false },
      { stage: DegradationStage.EMERGING, label: 'Emerging', isCurrent: false },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: true },
      { stage: DegradationStage.ACCELERATED, label: 'Accelerated', isCurrent: false },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure', isCurrent: false },
    ],
    currentIndex: 3,
  },
  driftChart: {
    dataPoints: Array.from({ length: 26 }, (_, i) => ({
      timestamp: isoAtOffset(25 - i),
      driftScore: Math.min(0.7, 0.18 + i * 0.02),
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 25,
  },
  timestamp: new Date(DEMO_BASE_TIME).toISOString(),
}

export const DEMO_SCENARIO_CRITICAL: DecisionUIState = {
  statusHeader: {
    degradationStage: DegradationStage.FAILURE_APPROACH,
    severity: Severity.HIGH,
    trajectory: Trajectory.DEGRADING,
    confidence: 'HIGH',
  },
  actionPanel: {
    horizon: ActionHorizon.NOW,
    primaryAction: 'Activate failover',
    urgencyLevel: 5,
  },
  tetrahedron: {
    currentPosition: { x: 0.9, y: 0.14, z: 0.88 },
    severityScalar: 0.96,
    trailPoints: Array.from({ length: 20 }, (_, i) => ({
      position: {
        x: 0.28 + i * 0.03,
        y: 0.72 - i * 0.03,
        z: 0.24 + i * 0.03,
      },
      timestamp: isoAtOffset(19 - i),
      frameIdx: i,
    })),
    velocity: [0.14, 0.12, 0.1],
    nearestVertex: 'STRUCTURAL',
  },
  decisionTrace: {
    primaryFactor: 'Failure risk accelerating',
    secondaryFactors: ['Cascade confirmed', 'Instability compounding'],
    confidenceRationale: 'Confidence high',
    patternInsight: {
      tier: 'strong',
      outcomeType: 'cascade_failure',
      influenceSummary: 'Failure window: narrowing',
    },
  },
  timeline: {
    stages: [
      { stage: DegradationStage.BASELINE, label: 'Baseline', isCurrent: false },
      { stage: DegradationStage.EARLY_SHIFT, label: 'Early Shift', isCurrent: false },
      { stage: DegradationStage.EMERGING, label: 'Emerging', isCurrent: false },
      { stage: DegradationStage.PERSISTENT, label: 'Persistent', isCurrent: false },
      { stage: DegradationStage.ACCELERATED, label: 'Accelerated', isCurrent: false },
      { stage: DegradationStage.FAILURE_APPROACH, label: 'Failure', isCurrent: true },
    ],
    currentIndex: 5,
  },
  driftChart: {
    dataPoints: Array.from({ length: 26 }, (_, i) => ({
      timestamp: isoAtOffset(25 - i),
      driftScore: Math.min(0.9, 0.22 + i * 0.028),
      frameIdx: i,
    })),
    detectionThreshold: 0.2,
    currentFrameIdx: 25,
  },
  timestamp: new Date(DEMO_BASE_TIME).toISOString(),
}

export const DEMO_SCENARIOS = [
  { name: 'Baseline', state: DEMO_SCENARIO_BASELINE, durationMs: 17000, narrative: 'System operating normally' },
  { name: 'Early Shift', state: DEMO_SCENARIO_EARLY_SHIFT, durationMs: 17000, narrative: 'Early structural change detected' },
  { name: 'Emerging', state: DEMO_SCENARIO_EMERGING, durationMs: 24000, narrative: 'Degradation detected before failure conditions' },
  { name: 'Critical', state: DEMO_SCENARIO_CRITICAL, durationMs: 20000, narrative: 'Failure imminent. Action required now' },
]

export const HERO_SCENARIO = DEMO_SCENARIO_EMERGING
