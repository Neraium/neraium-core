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
        {/* System state tetrahedron visualization - HERO */}
        <div style={styles.tetrahedronHero}>
          <div style={styles.sectionTitle}>System State Space</div>
          {state.tetrahedron ? (
            <div style={styles.tetrahedronContainerLarge}>
              <EnhancedTetrahedronViz tetrahedronState={state.tetrahedron} isInteractive={true} />
            </div>
          ) : (
            <div style={styles.placeholder}>No state space data available</div>
          )}
        </div>

        {/* Supporting metrics - smaller */}
        <div style={styles.supportingMetrics}>
          <div style={styles.metricBox}>
            <div style={styles.sectionTitle}>State Progression</div>
            <SystemTimeline timeline={state.timeline} />
          </div>

          <div style={styles.metricBox}>
            <div style={styles.sectionTitle}>Drift Curve</div>
            {state.driftChart.dataPoints.length > 0 ? (
              <DriftChart chart={state.driftChart} minimal={true} severity={state.statusHeader.severity} />
            ) : (
              <div style={styles.placeholder}>No drift data available</div>
            )}
          </div>
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
    border: '2px solid rgba(0, 102, 255,0.2)',
    backdropFilter: 'blur(10px)',
    boxShadow: '0 0 30px rgba(0, 102, 255,0.1), inset 0 0 20px rgba(0, 102, 255,0.05)',
    position: 'relative',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingBottom: '12px',
    borderBottom: '1px solid rgba(0, 102, 255,0.15)',
  },
  title: {
    fontSize: '11px',
    fontWeight: '700',
    letterSpacing: '0.15em',
    textTransform: 'uppercase',
    color: '#0099ff',
    textShadow: '0 0 8px rgba(0, 102, 255,0.6)',
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
    color: '#0077dd',
    letterSpacing: '0.08em',
    textShadow: '0 0 10px rgba(255, 0, 255, 0.8), 0 0 20px rgba(255, 0, 255, 0.4)',
  },
  stateDescription: {
    fontSize: '11px',
    color: 'rgba(148, 163, 184, 0.6)',
    fontWeight: '400',
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
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
    color: '#0088ff',
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
    borderLeft: '3px solid #0077dd',
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
    color: '#0099ff',
    fontWeight: '700',
    fontFamily: "'Space Mono', monospace",
    textShadow: '0 0 8px rgba(0, 102, 255,0.6)',
  },
  factorItem: {
    fontSize: '11px',
    color: '#cbd5e1',
    marginTop: '2px',
  },
  tetrahedronHero: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '24px',
    backgroundColor: 'rgba(0, 30, 60, 0.6)',
    borderRadius: '0',
    border: '2px solid rgba(0, 102, 255, 0.3)',
    backdropFilter: 'blur(12px)',
    boxShadow: '0 0 50px rgba(0, 102, 255, 0.15), inset 0 0 30px rgba(0, 102, 255, 0.08)',
  },
  tetrahedronContainerLarge: {
    width: '100%',
    height: '500px',
    borderRadius: '0',
    overflow: 'hidden',
    boxShadow: '0 0 60px rgba(0, 153, 255, 0.3), inset 0 0 40px rgba(0, 102, 255, 0.1)',
  },
  tetrahedronContainer: {
    width: '100%',
    height: '300px',
    borderRadius: '8px',
    overflow: 'hidden',
  },
  supportingMetrics: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '16px',
  },
  metricBox: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  containerGlow: {
    animation: 'transitionGlow 1.5s ease-out',
    boxShadow: '0 0 20px rgba(96, 165, 250, 0.4), 0 0 40px rgba(96, 165, 250, 0.2)',
  },
}

// Add animation styles
if (typeof document !== 'undefined' && !document.getElementById('system-state-trajectory-styles')) {
  const style = document.createElement('style')
  style.id = 'system-state-trajectory-styles'
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
