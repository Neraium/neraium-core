'use client'

import { useEffect, useState } from 'react'
import { DecisionUIState } from '@/lib/decisionToUI'
import SystemTimeline from '@/components/SystemTimeline'
import DriftChart from '@/components/DriftChart'
import EnhancedTetrahedronViz from '@/components/EnhancedTetrahedronViz'

interface SystemStateTrajectoryProps {
  state: DecisionUIState
  systemName?: string
}

export default function SystemStateTrajectory({ state, systemName }: SystemStateTrajectoryProps) {
  const [showTransitionGlow, setShowTransitionGlow] = useState(false)
  const [prevStage, setPrevStage] = useState(state.statusHeader.degradationStage)

  useEffect(() => {
    if (state.statusHeader.degradationStage !== prevStage) {
      setShowTransitionGlow(true)
      setPrevStage(state.statusHeader.degradationStage)
      const timer = setTimeout(() => setShowTransitionGlow(false), 1500)
      return () => clearTimeout(timer)
    }
  }, [state.statusHeader.degradationStage, prevStage])

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
    <div style={{
      ...styles.container,
      ...(showTransitionGlow ? styles.containerGlow : {}),
    }}>
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
    backgroundColor: 'rgba(0, 20, 40, 0.4)',
    borderRadius: '0',
    border: '2px solid rgba(0, 255, 255, 0.2)',
    backdropFilter: 'blur(10px)',
    boxShadow: '0 0 30px rgba(0, 255, 255, 0.1), inset 0 0 20px rgba(0, 255, 255, 0.05)',
    position: 'relative',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingBottom: '12px',
    borderBottom: '1px solid rgba(0, 255, 255, 0.15)',
  },
  title: {
    fontSize: '11px',
    fontWeight: '700',
    letterSpacing: '0.15em',
    textTransform: 'uppercase',
    color: '#00ffff',
    textShadow: '0 0 8px rgba(0, 255, 255, 0.6)',
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
    fontWeight: '700',
    color: '#ff00ff',
    letterSpacing: '0.08em',
    textShadow: '0 0 10px rgba(255, 0, 255, 0.8), 0 0 20px rgba(255, 0, 255, 0.4)',
  },
  stateDescription: {
    fontSize: '11px',
    color: 'rgba(148, 163, 184, 0.6)',
    fontWeight: '400',
  },
  content: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '16px',
    minHeight: '300px',
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  sectionTitle: {
    fontSize: '10px',
    fontWeight: '700',
    textTransform: 'uppercase',
    color: '#00d4ff',
    letterSpacing: '0.12em',
    textShadow: '0 0 6px rgba(0, 212, 255, 0.5)',
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
    padding: '14px 16px',
    backgroundColor: 'rgba(0, 30, 50, 0.4)',
    borderRadius: '0',
    borderLeft: '3px solid #ff00ff',
    backdropFilter: 'blur(8px)',
    boxShadow: '0 0 20px rgba(255, 0, 255, 0.1), inset 0 0 15px rgba(255, 0, 255, 0.05)',
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
    color: '#00ffff',
    fontWeight: '700',
    fontFamily: "'Space Mono', monospace",
    textShadow: '0 0 8px rgba(0, 255, 255, 0.6)',
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
  containerGlow: {
    animation: 'transitionGlow 1.5s ease-out',
    boxShadow: '0 0 20px rgba(96, 165, 250, 0.4), 0 0 40px rgba(96, 165, 250, 0.2)',
  },
}

// Add animation styles
if (typeof document !== 'undefined') {
  const style = document.createElement('style')
  style.textContent = `
    @keyframes transitionGlow {
      0% {
        box-shadow: 0 0 30px rgba(96, 165, 250, 0.8), 0 0 60px rgba(96, 165, 250, 0.4);
      }
      100% {
        box-shadow: 0 0 10px rgba(96, 165, 250, 0.1), 0 0 20px rgba(96, 165, 250, 0.05);
      }
    }
  `
  document.head.appendChild(style)
}
