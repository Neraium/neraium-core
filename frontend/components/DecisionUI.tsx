'use client'

import { DecisionUIState, ActionHorizon, Severity, DegradationStage } from '@/lib/decisionToUI'
import StatusHeader from '@/components/StatusHeader'
import ActionPanel from '@/components/ActionPanel'
import DecisionTrace from '@/components/DecisionTrace'
import SystemTimeline from '@/components/SystemTimeline'
import DriftChart from '@/components/DriftChart'
import EnhancedTetrahedronViz from '@/components/EnhancedTetrahedronViz'

interface DecisionUIProps {
  state?: DecisionUIState
  isLoading?: boolean
  error?: string | null
  narrative?: string
  meaningLine?: string | null
}

function inferImpactWindow(state: DecisionUIState): string {
  const summary = state.decisionTrace.patternInsight?.influenceSummary
  if (summary && /window/i.test(summary)) return summary

  if (state.actionPanel.horizon === ActionHorizon.NOW && state.statusHeader.severity === Severity.HIGH) {
    return 'Failure window: narrowing'
  }
  if (state.actionPanel.horizon === ActionHorizon.NOW) return 'Impact window: active'
  if (state.actionPanel.horizon === ActionHorizon.SOON) return 'Action window: open'
  return 'Action window: open'
}

export default function DecisionUI({ state, isLoading = false, error = null, narrative, meaningLine }: DecisionUIProps) {
  if (error) {
    return <div style={styles.root}><div style={styles.simpleText}>Unable to load: {error}</div></div>
  }

  if (isLoading || !state) {
    return <div style={styles.root}><div style={styles.simpleText}>Loading…</div></div>
  }

  const impactWindow = inferImpactWindow(state)
  const isEmergingMoment = Boolean(meaningLine) && state.statusHeader.degradationStage !== DegradationStage.FAILURE_APPROACH
  const severityTone = state.tetrahedron.severityScalar
  const ambientDim = 1 - severityTone * 0.13

  return (
    <div style={styles.root}>
      <div
        style={{
          ...styles.surface,
          background: `linear-gradient(180deg, rgba(15,23,42,${0.26 + severityTone * 0.12}) 0%, rgba(2,6,23,${0.13 + severityTone * 0.16}) 100%)`,
        }}
      >
        {narrative && <div style={{ ...styles.narrative, opacity: ambientDim }}>{narrative}</div>}

        <div style={{ opacity: ambientDim, transition: 'opacity 0.62s ease' }}>
          <StatusHeader status={state.statusHeader} />
        </div>

        <div style={{ opacity: ambientDim, transition: 'opacity 0.62s ease', marginLeft: '1px' }}>
          <ActionPanel action={state.actionPanel} impactWindow={impactWindow} />
        </div>

        {isEmergingMoment && <div style={{ ...styles.meaningLine, opacity: 0.9 * ambientDim }}>{meaningLine}</div>}

        <div style={styles.heroSection}>
          <EnhancedTetrahedronViz tetrahedronState={state.tetrahedron} isInteractive={true} />
        </div>

        <div style={{ ...styles.contextGrid, opacity: ambientDim, transition: 'opacity 0.62s ease' }}>
          <SystemTimeline timeline={state.timeline} />
          {state.driftChart.dataPoints.length > 0 ? <DriftChart chart={state.driftChart} /> : <div style={styles.simpleText}>No drift data</div>}
        </div>

        <div style={{ opacity: ambientDim, transition: 'opacity 0.62s ease', marginLeft: '2px' }}>
          <DecisionTrace trace={state.decisionTrace} />
        </div>

        <div style={{ ...styles.footer, opacity: 0.75 * ambientDim }}>
          {new Date(state.timestamp).toLocaleString(undefined, {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
          })}
        </div>
      </div>
    </div>
  )
}

const styles = {
  root: {
    minHeight: '100vh',
    padding: '22px 28px 80px',
    background: 'radial-gradient(circle at 50% -10%, #0b1222 0%, #020617 58%, #000 100%)',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
  },
  surface: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '25px',
    maxWidth: '1480px',
    margin: '0 auto',
    padding: '21px 25px 0 23px',
    borderRadius: '24px',
    transition: 'background 0.62s ease',
  },
  narrative: {
    fontSize: '12px',
    letterSpacing: '0.08em',
    color: 'rgba(203, 213, 225, 0.65)',
    textTransform: 'uppercase' as const,
    transition: 'opacity 0.62s ease',
  },
  meaningLine: {
    fontSize: '13px',
    color: 'rgba(147, 197, 253, 0.78)',
    letterSpacing: '0.04em',
    marginTop: '-7px',
    transition: 'opacity 0.62s ease',
  },
  heroSection: {
    minHeight: '740px',
    transition: 'all 0.8s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  contextGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '21px',
  },
  footer: {
    fontSize: '11px',
    color: 'rgba(203,213,225,0.34)',
    letterSpacing: '0.08em',
    textAlign: 'right' as const,
    fontVariantNumeric: 'tabular-nums',
    paddingBottom: '8px',
    transition: 'opacity 0.62s ease',
  },
  simpleText: {
    color: 'rgba(226, 232, 240, 0.62)',
    fontSize: '13px',
  },
}
