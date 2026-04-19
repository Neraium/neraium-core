/**
 * Demo component showing complete decision system data flow.
 *
 * This is a reference implementation showing:
 * 1. Sample decision data
 * 2. Integration layer transformation (decision → UI state)
 * 3. Visualization using mapped state
 */

'use client'

import { useMemo } from 'react'
import { transformDecisionToUIState } from '@/lib/decisionToUI'
import EnhancedTetrahedronViz from './EnhancedTetrahedronViz'

interface DecisionSystemDemoProps {
  gateDecision?: Record<string, unknown>
  records?: Record<string, unknown>[]
}

/**
 * Sample gate decision (as returned by decision engine)
 */
const SAMPLE_GATE_DECISION = {
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

/**
 * Sample telemetry records (history with most recent last)
 */
const SAMPLE_RECORDS = [
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
    timestamp: '2026-04-18T14:05:00Z',
    structural_drift_score: 0.38,
    relational_stability_score: 0.71,
    coherence_score: 0.81,
    persistence_minutes: 10,
    corroborating_signal_count: 0,
    delta_drift: 0.03,
    delta_stability: -0.04,
    event_admitted: false,
  },
  {
    timestamp: '2026-04-18T14:10:00Z',
    structural_drift_score: 0.42,
    relational_stability_score: 0.68,
    coherence_score: 0.79,
    persistence_minutes: 15,
    corroborating_signal_count: 1,
    delta_drift: 0.04,
    delta_stability: -0.03,
    event_admitted: false,
  },
  {
    timestamp: '2026-04-18T14:15:00Z',
    structural_drift_score: 0.47,
    relational_stability_score: 0.64,
    coherence_score: 0.77,
    persistence_minutes: 20,
    corroborating_signal_count: 1,
    delta_drift: 0.05,
    delta_stability: -0.04,
    event_admitted: false,
  },
  {
    timestamp: '2026-04-18T14:20:00Z',
    structural_drift_score: 0.53,
    relational_stability_score: 0.52,
    coherence_score: 0.74,
    persistence_minutes: 25,
    corroborating_signal_count: 2,
    delta_drift: 0.06,
    delta_stability: -0.12,
    event_admitted: false,
  },
  {
    timestamp: '2026-04-18T14:25:00Z',
    structural_drift_score: 0.58,
    relational_stability_score: 0.42,
    coherence_score: 0.72,
    persistence_minutes: 50,
    corroborating_signal_count: 3,
    delta_drift: 0.05,
    delta_stability: -0.1,
    event_admitted: false,
  },
  {
    timestamp: '2026-04-18T14:30:00Z',
    structural_drift_score: 0.63,
    relational_stability_score: 0.36,
    coherence_score: 0.73,
    persistence_minutes: 70,
    corroborating_signal_count: 4,
    delta_drift: 0.05,
    delta_stability: -0.06,
    event_admitted: true,
  },
  {
    timestamp: '2026-04-18T14:32:00Z',
    structural_drift_score: 0.68,
    relational_stability_score: 0.31,
    coherence_score: 0.72,
    persistence_minutes: 87,
    corroborating_signal_count: 4,
    delta_drift: 0.05,
    delta_stability: -0.05,
    event_admitted: true,
  },
]

export default function DecisionSystemDemo({ gateDecision, records }: DecisionSystemDemoProps) {
  // Use provided data or sample data
  const decision = gateDecision ?? SAMPLE_GATE_DECISION
  const telemetryRecords = records ?? SAMPLE_RECORDS

  // Transform decision + records → UI state
  const uiState = useMemo(() => {
    try {
      return transformDecisionToUIState(decision, telemetryRecords)
    } catch (error) {
      console.error('Failed to transform decision to UI state:', error)
      return null
    }
  }, [decision, telemetryRecords])

  if (!uiState) {
    return <div className="text-red-500 p-4">Failed to transform decision state</div>
  }

  // Display the tetrahedron visualization
  return (
    <div className="space-y-6 p-6">
      {/* DEBUG: Show state transformation */}
      <div className="space-y-4">
        <h2 className="text-xl font-bold text-slate-200">Decision System Visualization</h2>

        {/* Status Summary */}
        <div className="bg-slate-800 rounded p-4 space-y-2">
          <h3 className="font-semibold text-slate-300">Status</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-slate-500">Degradation Stage:</span>
              <span className="ml-2 text-slate-200">{uiState.statusHeader.degradationStage}</span>
            </div>
            <div>
              <span className="text-slate-500">Severity:</span>
              <span className="ml-2 text-slate-200">{uiState.statusHeader.severity}</span>
            </div>
            <div>
              <span className="text-slate-500">Trajectory:</span>
              <span className="ml-2 text-slate-200">{uiState.statusHeader.trajectory}</span>
            </div>
            <div>
              <span className="text-slate-500">Confidence:</span>
              <span className="ml-2 text-slate-200">{uiState.statusHeader.confidence}</span>
            </div>
          </div>
        </div>

        {/* Action Panel */}
        <div className="bg-slate-800 rounded p-4 space-y-2">
          <h3 className="font-semibold text-slate-300">Action</h3>
          <div>
            <span className="text-slate-500">Horizon:</span>
            <span className="ml-2 text-slate-200 uppercase">{uiState.actionPanel.horizon}</span>
            <span className="ml-2 text-slate-500">
              (Urgency: {uiState.actionPanel.urgencyLevel}/5)
            </span>
          </div>
          <div className="text-slate-200">{uiState.actionPanel.primaryAction}</div>
        </div>

        {/* Tetrahedron State */}
        <div className="bg-slate-800 rounded p-4 space-y-2">
          <h3 className="font-semibold text-slate-300">Tetrahedron State</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-slate-500">Current Position:</span>
              <div className="text-slate-200 text-xs mt-1">
                x={uiState.tetrahedron.currentPosition.x.toFixed(3)}, y=
                {uiState.tetrahedron.currentPosition.y.toFixed(3)}, z=
                {uiState.tetrahedron.currentPosition.z.toFixed(3)}
              </div>
            </div>
            <div>
              <span className="text-slate-500">Severity Scalar:</span>
              <span className="ml-2 text-slate-200">{uiState.tetrahedron.severityScalar.toFixed(2)}</span>
            </div>
            <div>
              <span className="text-slate-500">Trail Points:</span>
              <span className="ml-2 text-slate-200">{uiState.tetrahedron.trailPoints.length}</span>
            </div>
            <div>
              <span className="text-slate-500">Nearest Vertex:</span>
              <span className="ml-2 text-slate-200">{uiState.tetrahedron.nearestVertex}</span>
            </div>
          </div>
        </div>

        {/* Decision Trace */}
        <div className="bg-slate-800 rounded p-4 space-y-2">
          <h3 className="font-semibold text-slate-300">Decision Trace</h3>
          <div>
            <span className="text-slate-500 text-sm">Primary Factor:</span>
            <div className="text-slate-200">{uiState.decisionTrace.primaryFactor}</div>
          </div>
          {uiState.decisionTrace.secondaryFactors.length > 0 && (
            <div>
              <span className="text-slate-500 text-sm">Secondary Factors:</span>
              <ul className="text-slate-200 text-sm ml-2">
                {uiState.decisionTrace.secondaryFactors.map((factor, idx) => (
                  <li key={idx}>• {factor}</li>
                ))}
              </ul>
            </div>
          )}
          {uiState.decisionTrace.patternInsight && (
            <div className="text-blue-400 text-sm">
              📊 Pattern: {uiState.decisionTrace.patternInsight.outcomeType}
            </div>
          )}
        </div>
      </div>

      {/* Tetrahedron Visualization */}
      <div className="space-y-2">
        <h3 className="font-semibold text-slate-300">3D System State</h3>
        <EnhancedTetrahedronViz tetrahedronState={uiState.tetrahedron} isInteractive={true} />
      </div>

      {/* Timeline */}
      <div className="bg-slate-800 rounded p-4 space-y-2">
        <h3 className="font-semibold text-slate-300">System Timeline</h3>
        <div className="flex gap-2 flex-wrap">
          {uiState.timeline.stages.map((stage) => (
            <div
              key={stage.stage}
              className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
                stage.isCurrent
                  ? 'bg-orange-500 text-white scale-110'
                  : 'bg-slate-700 text-slate-300'
              }`}
            >
              {stage.label}
            </div>
          ))}
        </div>
      </div>

      {/* Drift Chart Data */}
      <div className="bg-slate-800 rounded p-4 space-y-2">
        <h3 className="font-semibold text-slate-300">Drift Progression (last 24 cycles)</h3>
        <div className="text-xs text-slate-400">
          {uiState.driftChart.dataPoints.length} data points available
        </div>
        <div className="grid grid-cols-4 gap-2">
          {uiState.driftChart.dataPoints.map((point, idx) => (
            <div key={idx} className="text-xs">
              <div className="text-slate-500">Frame {point.frameIdx}</div>
              <div className="text-slate-200">{(point.driftScore * 100).toFixed(0)}%</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
