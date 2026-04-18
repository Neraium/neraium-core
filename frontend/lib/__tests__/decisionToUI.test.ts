/**
 * Test and demonstration of decision → UI state mapping.
 *
 * Shows how gate decision + records transform into complete UI state.
 */

import { transformDecisionToUIState, Severity, DegradationStage, ActionHorizon } from '../decisionToUI'

describe('Decision → UI State Transformation', () => {
  // Sample gate decision (from decision engine)
  const sampleGateDecision = {
    decision: 'ADMIT',
    confidence_label: 'HIGH',
    confidence_value: 0.87,
    explanation: 'Persistent degradation detected with stable trajectory',
    refusal_reason: null,
    doctrine_version: 'v1.2',
    transition: {
      type: 'REORGANIZATION',
      delta_drift: 0.15,
      delta_stability: -0.22,
      delta_coherence: -0.08,
    },
    pattern_match: {
      tier: 'moderate',
      outcome_type: 'failure-progressing',
      influence_summary: 'Similar to prior failure-progressing pattern',
    },
  }

  // Sample telemetry records (history with most recent last)
  const sampleRecords = [
    {
      timestamp: '2026-04-18T14:00:00Z',
      structural_drift_score: 0.35,
      relational_stability_score: 0.75,
      coherence_score: 0.82,
      persistence_minutes: 5,
      corroborating_signal_count: 0,
      delta_drift: 0.02,
      delta_stability: -0.01,
      event_admitted: false,
    },
    {
      timestamp: '2026-04-18T14:10:00Z',
      structural_drift_score: 0.42,
      relational_stability_score: 0.68,
      coherence_score: 0.79,
      persistence_minutes: 15,
      corroborating_signal_count: 1,
      delta_drift: 0.07,
      delta_stability: -0.07,
      event_admitted: false,
    },
    {
      timestamp: '2026-04-18T14:20:00Z',
      structural_drift_score: 0.53,
      relational_stability_score: 0.52,
      coherence_score: 0.74,
      persistence_minutes: 25,
      corroborating_signal_count: 2,
      delta_drift: 0.11,
      delta_stability: -0.16,
      event_admitted: false,
    },
    {
      timestamp: '2026-04-18T14:32:00Z',
      structural_drift_score: 0.68,
      relational_stability_score: 0.31,
      coherence_score: 0.72,
      persistence_minutes: 87,
      corroborating_signal_count: 4,
      delta_drift: 0.15,
      delta_stability: -0.21,
      event_admitted: true,
    },
  ]

  test('transforms high severity decision to correct UI state', () => {
    const uiState = transformDecisionToUIState(sampleGateDecision, sampleRecords)

    // Verify status header
    expect(uiState.statusHeader.severity).toBe(Severity.HIGH)
    expect(uiState.statusHeader.confidence).toBe('HIGH')
    expect(uiState.statusHeader.degradationStage).toBe(DegradationStage.PERSISTENT)

    // Verify action panel
    expect(uiState.actionPanel.horizon).toBe(ActionHorizon.NOW)
    expect(uiState.actionPanel.urgencyLevel).toBe(5)

    // Verify tetrahedron state
    expect(uiState.tetrahedron.severityScalar).toBe(0.95)
    expect(uiState.tetrahedron.currentPosition.x).toBeGreaterThan(0.5)
    expect(uiState.tetrahedron.trailPoints.length).toBeLessThanOrEqual(20)

    // Verify decision trace
    expect(uiState.decisionTrace.primaryFactor).toContain('87')
    expect(uiState.decisionTrace.secondaryFactors.length).toBeGreaterThan(0)

    // Verify pattern insight
    expect(uiState.decisionTrace.patternInsight).toBeDefined()
    expect(uiState.decisionTrace.patternInsight?.tier).toBe('moderate')
  })

  test('low severity decision maps to LOW severity and WATCHLIST horizon', () => {
    const lowSeverityDecision = {
      decision: 'SUPPRESS',
      confidence_label: 'LOW',
      explanation: 'System within normal parameters',
    }

    const lowSeverityRecords = [
      {
        timestamp: '2026-04-18T15:00:00Z',
        structural_drift_score: 0.12,
        relational_stability_score: 0.92,
        coherence_score: 0.88,
        persistence_minutes: 0,
        corroborating_signal_count: 0,
        delta_drift: 0.01,
        delta_stability: 0.02,
        event_admitted: false,
      },
    ]

    const uiState = transformDecisionToUIState(lowSeverityDecision, lowSeverityRecords)

    expect(uiState.statusHeader.severity).toBe(Severity.LOW)
    expect(uiState.actionPanel.horizon).toBe(ActionHorizon.WATCHLIST)
    expect(uiState.actionPanel.urgencyLevel).toBe(1)
    expect(uiState.statusHeader.degradationStage).toBe(DegradationStage.BASELINE)
  })

  test('trajectory detection works correctly', () => {
    const degradingRecords = [
      {
        timestamp: '2026-04-18T14:00:00Z',
        structural_drift_score: 0.35,
        relational_stability_score: 0.75,
        coherence_score: 0.82,
        persistence_minutes: 5,
        corroborating_signal_count: 0,
        delta_drift: 0.0,
        delta_stability: 0.0,
        event_admitted: false,
      },
      {
        timestamp: '2026-04-18T14:10:00Z',
        structural_drift_score: 0.55,
        relational_stability_score: 0.5,
        coherence_score: 0.79,
        persistence_minutes: 15,
        corroborating_signal_count: 1,
        delta_drift: 0.2,
        delta_stability: -0.25,
        event_admitted: false,
      },
    ]

    const uiState = transformDecisionToUIState(sampleGateDecision, degradingRecords)

    // Large delta_drift should be detected as DEGRADING
    expect(uiState.statusHeader.trajectory.toString()).toContain('Degrading')
  })

  test('trail generation includes all recent records', () => {
    const manyRecords = Array.from({ length: 30 }, (_, i) => ({
      timestamp: new Date(Date.now() - (30 - i) * 10000).toISOString(),
      structural_drift_score: 0.2 + (i * 0.02),
      relational_stability_score: 0.8 - (i * 0.01),
      coherence_score: 0.8,
      persistence_minutes: i,
      corroborating_signal_count: 0,
      delta_drift: 0.02,
      delta_stability: -0.01,
      event_admitted: false,
    }))

    const uiState = transformDecisionToUIState(sampleGateDecision, manyRecords)

    // Trail should be limited to 20 points
    expect(uiState.tetrahedron.trailPoints.length).toBeLessThanOrEqual(20)
    expect(uiState.tetrahedron.trailPoints.length).toBeGreaterThan(0)

    // Most recent point should be at the end
    expect(uiState.tetrahedron.trailPoints[uiState.tetrahedron.trailPoints.length - 1].position).toEqual(
      uiState.tetrahedron.currentPosition
    )
  })

  test('tetrahedron position stays within valid bounds', () => {
    const uiState = transformDecisionToUIState(sampleGateDecision, sampleRecords)

    const { x, y, z } = uiState.tetrahedron.currentPosition
    expect(x).toBeGreaterThanOrEqual(0)
    expect(x).toBeLessThanOrEqual(1)
    expect(y).toBeGreaterThanOrEqual(0)
    expect(y).toBeLessThanOrEqual(1)
    expect(z).toBeGreaterThanOrEqual(0)
    expect(z).toBeLessThanOrEqual(1)
  })

  test('severity scalar maps to correct color classification', () => {
    const uiState = transformDecisionToUIState(sampleGateDecision, sampleRecords)

    // HIGH severity should map to 0.95
    expect(uiState.tetrahedron.severityScalar).toBe(0.95)
  })

  test('nearest vertex identifies dominant structural dimension', () => {
    const uiState = transformDecisionToUIState(sampleGateDecision, sampleRecords)

    // With high severity position, should be near a vertex
    expect(['STRUCTURAL', 'RELATIONAL', 'TEMPORAL', 'AUTHORITY']).toContain(
      uiState.tetrahedron.nearestVertex
    )
  })

  test('drift chart includes last 24 records', () => {
    const manyRecords = Array.from({ length: 50 }, (_, i) => ({
      timestamp: new Date(Date.now() - (50 - i) * 10000).toISOString(),
      structural_drift_score: 0.2 + (i * 0.01),
      relational_stability_score: 0.8,
      coherence_score: 0.8,
      persistence_minutes: i,
      corroborating_signal_count: 0,
      delta_drift: 0.01,
      delta_stability: 0.0,
      event_admitted: false,
    }))

    const uiState = transformDecisionToUIState(sampleGateDecision, manyRecords)

    expect(uiState.driftChart.dataPoints.length).toBeLessThanOrEqual(24)
    expect(uiState.driftChart.dataPoints.length).toBeGreaterThan(0)
  })

  test('timeline includes all degradation stages with current marked', () => {
    const uiState = transformDecisionToUIState(sampleGateDecision, sampleRecords)

    expect(uiState.timeline.stages.length).toBe(6)

    // Current stage should be marked
    const currentStage = uiState.timeline.stages[uiState.timeline.currentIndex]
    expect(currentStage.isCurrent).toBe(true)

    // Other stages should not be current
    uiState.timeline.stages.forEach((stage, idx) => {
      if (idx !== uiState.timeline.currentIndex) {
        expect(stage.isCurrent).toBe(false)
      }
    })
  })
})

/**
 * Example usage and data flow
 */
export function exampleUsage() {
  // 1. Receive gate decision + records from backend
  const gateDecision = {
    decision: 'ADMIT',
    confidence_label: 'HIGH',
    explanation: 'Persistent degradation with stable trajectory',
    pattern_match: {
      tier: 'moderate',
      outcome_type: 'failure-progressing',
      influence_summary: 'Similar to prior failure-progressing pattern',
    },
  }

  const records = [
    // ... telemetry records
  ]

  // 2. Transform to UI state
  const uiState = transformDecisionToUIState(gateDecision, records)

  // 3. Render components using uiState
  // <StatusHeader {...uiState.statusHeader} />
  // <ActionPanel {...uiState.actionPanel} />
  // <EnhancedTetrahedronViz tetrahedronState={uiState.tetrahedron} />
  // <DecisionTrace {...uiState.decisionTrace} />
  // <SystemTimeline {...uiState.timeline} />
  // <DriftChart {...uiState.driftChart} />

  return uiState
}
