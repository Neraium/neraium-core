'use client'

import { DecisionUIState } from '@/lib/decisionToUI'
import SystemTimeline from '@/components/SystemTimeline'
import DriftChart from '@/components/DriftChart'
import EnhancedTetrahedronViz from '@/components/EnhancedTetrahedronViz'

interface SystemStateTrajectoryProps {
  state: DecisionUIState
  systemName?: string
}

export default function SystemStateTrajectory({ state, systemName }: SystemStateTrajectoryProps) {
  const stateLabels: Record<string, string> = {
    baseline: 'Stable',
    early_shift: 'Early Drift',
    emerging: 'Structural Shift',
    persistent: 'Structural Shift',
    accelerated: 'Pre-instability',
    failure_approach: 'Pre-instability',
  }

  const currentStateLabel = stateLabels[state.statusHeader.degradationStage] || 'Unknown'

  const stateDescriptions: Record<string, string> = {
    baseline: 'System operating within normal parameters',
    early_shift: 'System behavior deviating from baseline',
    emerging: 'Structural relationships weakening',
    persistent: 'Structural relationships weakening',
    accelerated: 'Trajectory indicates emerging risk',
    failure_approach: 'Trajectory indicates emerging risk',
  }

  const currentDescription = stateDescriptions[state.statusHeader.degradationStage] || 'System state change detected'

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div>
          <div style={styles.title}>System State & Trajectory</div>
          {systemName && <div style={styles.systemName}>{systemName}</div>}
        </div>
        <div style={styles.currentState}>
          <div style={styles.stateLabel}>{currentStateLabel}</div>
          <div style={styles.stateDescription}>{currentDescription}</div>
        </div>
      </div>

      <div style={styles.content}>
        {/* Timeline showing state progression */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>State Progression</div>
          <SystemTimeline timeline={state.timeline} />
        </div>

        {/* Drift trajectory chart */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Drift Curve</div>
          {state.driftChart.dataPoints.length > 0 ? (
            <DriftChart chart={state.driftChart} minimal={false} severity={state.statusHeader.severity} />
          ) : (
            <div style={styles.placeholder}>No drift data available</div>
          )}
        </div>

        {/* System state tetrahedron visualization */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>System State Space</div>
          {state.tetrahedron ? (
            <div style={styles.tetrahedronContainer}>
              <EnhancedTetrahedronViz tetrahedronState={state.tetrahedron} isInteractive={false} />
            </div>
          ) : (
            <div style={styles.placeholder}>No state space data available</div>
          )}
        </div>
      </div>

      {/* Key insight panel */}
      <div style={styles.insightPanel}>
        <div style={styles.insightTitle}>System Evolution</div>
        <div style={styles.insightContent}>
          <div>
            <div style={styles.insightLabel}>Current trajectory:</div>
            <div style={styles.insightValue}>{state.statusHeader.trajectory}</div>
          </div>
          <div>
            <div style={styles.insightLabel}>Confidence:</div>
            <div style={styles.insightValue}>{state.statusHeader.confidence}</div>
          </div>
          {state.decisionTrace.secondaryFactors.length > 0 && (
            <div>
              <div style={styles.insightLabel}>Contributing factors:</div>
              {state.decisionTrace.secondaryFactors.map((factor, idx) => (
                <div key={idx} style={styles.factorItem}>• {factor}</div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    padding: '20px',
    backgroundColor: 'rgba(15, 23, 42, 0.3)',
    borderRadius: '12px',
    border: '1px solid rgba(148, 163, 184, 0.08)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingBottom: '12px',
    borderBottom: '1px solid rgba(148, 163, 184, 0.08)',
  },
  title: {
    fontSize: '12px',
    fontWeight: '600',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: 'rgba(148, 163, 184, 0.72)',
  },
  systemName: {
    fontSize: '13px',
    color: '#e2e8f0',
    marginTop: '4px',
    fontWeight: '500',
  },
  currentState: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    textAlign: 'right',
  },
  stateLabel: {
    fontSize: '14px',
    fontWeight: '600',
    color: '#60a5fa',
    letterSpacing: '0.05em',
  },
  stateDescription: {
    fontSize: '11px',
    color: 'rgba(148, 163, 184, 0.6)',
    fontWeight: '400',
  },
  content: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr 1fr',
    gap: '16px',
    minHeight: '300px',
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  sectionTitle: {
    fontSize: '11px',
    fontWeight: '500',
    textTransform: 'uppercase',
    color: 'rgba(148, 163, 184, 0.55)',
    letterSpacing: '0.05em',
  },
  placeholder: {
    padding: '20px',
    textAlign: 'center',
    color: 'rgba(148, 163, 184, 0.5)',
    fontSize: '13px',
  },
  insightPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    padding: '12px 14px',
    backgroundColor: 'rgba(30, 41, 59, 0.5)',
    borderRadius: '8px',
    borderLeft: '3px solid rgba(96, 165, 250, 0.5)',
  },
  insightTitle: {
    fontSize: '10px',
    fontWeight: '600',
    textTransform: 'uppercase',
    color: 'rgba(148, 163, 184, 0.6)',
    letterSpacing: '0.05em',
  },
  insightContent: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
    gap: '10px',
    fontSize: '12px',
  },
  insightLabel: {
    fontSize: '10px',
    color: 'rgba(148, 163, 184, 0.5)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: '2px',
  },
  insightValue: {
    fontSize: '12px',
    color: '#cbd5e1',
    fontWeight: '500',
  },
  factorItem: {
    fontSize: '11px',
    color: '#cbd5e1',
    marginTop: '2px',
  },
  tetrahedronContainer: {
    width: '100%',
    height: '300px',
    borderRadius: '8px',
    overflow: 'hidden',
  },
}
