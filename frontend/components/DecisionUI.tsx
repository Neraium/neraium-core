'use client'

import { DecisionUIState, ActionHorizon, Severity } from '@/lib/decisionToUI'
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
}

function inferImpactWindow(state: DecisionUIState): string | null {
  const summary = state.decisionTrace.patternInsight?.influenceSummary
  if (summary && /window/i.test(summary)) return summary

  if (state.actionPanel.horizon === ActionHorizon.NOW && state.statusHeader.severity === Severity.HIGH) {
    return 'Failure window: narrowing'
  }
  if (state.actionPanel.horizon === ActionHorizon.NOW) return 'Estimated impact window: active'
  if (state.actionPanel.horizon === ActionHorizon.SOON) return 'Action window open'
  return 'Action window remains open'
}

export default function DecisionUI({ state, isLoading = false, error = null, narrative }: DecisionUIProps) {
  if (error) {
    return <div style={styles.root}><div style={styles.simpleText}>Unable to load decision state: {error}</div></div>
  }

  if (isLoading || !state) {
    return <div style={styles.root}><div style={styles.simpleText}>Loading decision state…</div></div>
  }

  const impactWindow = inferImpactWindow(state)

  return (
    <div style={styles.root}>
      {narrative && <div style={styles.narrative}>{narrative}</div>}

      <StatusHeader status={state.statusHeader} />
      <ActionPanel action={state.actionPanel} impactWindow={impactWindow} />

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
  )
}

const styles = {
  root: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '28px',
    padding: '36px 42px 84px',
    background: 'radial-gradient(circle at 50% -10%, #111827 0%, #030712 50%, #000 100%)',
    color: '#fff',
    minHeight: '100vh',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
  },
  narrative: {
    fontSize: '13px',
    letterSpacing: '0.06em',
    color: 'rgba(226, 232, 240, 0.68)',
    textTransform: 'uppercase' as const,
    transition: 'opacity 0.45s ease',
  },
  heroSection: {
    minHeight: '700px',
    backgroundColor: 'rgba(7, 12, 24, 0.45)',
    borderRadius: '18px',
    padding: '22px',
    transition: 'all 0.7s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  contextGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '20px',
  },
  footer: {
    fontSize: '11px',
    color: 'rgba(255,255,255,0.34)',
    letterSpacing: '0.06em',
    textAlign: 'right' as const,
    fontVariantNumeric: 'tabular-nums',
  },
  simpleText: {
    color: 'rgba(226, 232, 240, 0.62)',
    fontSize: '13px',
  },
}
