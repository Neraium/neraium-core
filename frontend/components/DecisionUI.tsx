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

  return (
    <div style={styles.root}>
      <div style={styles.surface}>
        {narrative && <div style={styles.narrative}>{narrative}</div>}

        <StatusHeader status={state.statusHeader} />
        <ActionPanel action={state.actionPanel} impactWindow={impactWindow} />
        {isEmergingMoment && <div style={styles.meaningLine}>{meaningLine}</div>}

        <div style={styles.heroSection}>
          <EnhancedTetrahedronViz tetrahedronState={state.tetrahedron} isInteractive={true} />
        </div>

        <div style={styles.contextGrid}>
          <SystemTimeline timeline={state.timeline} />
          {state.driftChart.dataPoints.length > 0 ? <DriftChart chart={state.driftChart} /> : <div style={styles.simpleText}>No drift data</div>}
        </div>

        <DecisionTrace trace={state.decisionTrace} />

        <div style={styles.footer}>
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
    gap: '26px',
    maxWidth: '1480px',
    margin: '0 auto',
    padding: '20px 24px 0',
    background: 'linear-gradient(180deg, rgba(15,23,42,0.28) 0%, rgba(2,6,23,0.14) 100%)',
    borderRadius: '24px',
  },
  narrative: {
    fontSize: '12px',
    letterSpacing: '0.08em',
    color: 'rgba(203, 213, 225, 0.65)',
    textTransform: 'uppercase' as const,
    transition: 'opacity 0.45s ease',
  },
  meaningLine: {
    fontSize: '13px',
    color: 'rgba(147, 197, 253, 0.78)',
    letterSpacing: '0.04em',
    marginTop: '-8px',
    transition: 'opacity 0.45s ease',
  },
  heroSection: {
    minHeight: '740px',
    transition: 'all 0.7s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  contextGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '20px',
  },
  footer: {
    fontSize: '11px',
    color: 'rgba(203,213,225,0.34)',
    letterSpacing: '0.08em',
    textAlign: 'right' as const,
    fontVariantNumeric: 'tabular-nums',
    paddingBottom: '8px',
  },
  simpleText: {
    color: 'rgba(226, 232, 240, 0.62)',
    fontSize: '13px',
  },
}
