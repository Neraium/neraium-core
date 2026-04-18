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
  isAutoPlay?: boolean
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

export default function DecisionUI({ state, isLoading = false, error = null, narrative, meaningLine, isAutoPlay }: DecisionUIProps) {
  if (error) {
    return <div style={styles.root}><div style={styles.simpleText}>Unable to load: {error}</div></div>
  }

  if (isLoading || !state) {
    return <div style={styles.root}><div style={styles.simpleText}>Loading…</div></div>
  }

  const impactWindow = inferImpactWindow(state)
  const severityTone = state.tetrahedron.severityScalar
  const ambientDim = 1 - severityTone * 0.13
  const isChartMinimal = isAutoPlay && (
    state.statusHeader.degradationStage === DegradationStage.BASELINE ||
    state.statusHeader.degradationStage === DegradationStage.EARLY_SHIFT
  )

  return (
    <div style={styles.root}>
      <div
        style={{
          ...styles.surface,
          background: `linear-gradient(180deg, rgba(15,23,42,${0.26 + severityTone * 0.12}) 0%, rgba(2,6,23,${0.13 + severityTone * 0.16}) 100%)`,
        }}
      >
        {/* Top narrative */}
        {narrative && <div style={styles.narrative}>{narrative}</div>}

        {/* Hero split */}
        <div style={styles.heroSplit}>
          {/* Left: status + action + meaning */}
          <div style={styles.leftColumn}>
            <StatusHeader status={state.statusHeader} />
            <div style={{ marginTop: '8px' }}>
              <ActionPanel action={state.actionPanel} impactWindow={impactWindow} />
            </div>
            {meaningLine && <div style={styles.meaningLine}>{meaningLine}</div>}
          </div>

          {/* Right: tetrahedron large and vertically centered */}
          <div style={styles.rightColumn}>
            <EnhancedTetrahedronViz tetrahedronState={state.tetrahedron} isInteractive={true} />
          </div>
        </div>

        {/* Bottom support band: timeline + drift */}
        <div style={{ ...styles.supportBand, opacity: ambientDim, transition: 'opacity 0.62s ease' }}>
          <div style={styles.supportItem}>
            <SystemTimeline timeline={state.timeline} />
          </div>
          <div style={styles.supportItem}>
            {state.driftChart.dataPoints.length > 0 ? (
              <DriftChart chart={state.driftChart} minimal={isChartMinimal} />
            ) : (
              <div style={styles.simpleText}>No drift data</div>
            )}
          </div>
        </div>

        {/* Lower-left trace panel */}
        <div style={{ ...styles.traceRow, opacity: ambientDim, transition: 'opacity 0.62s ease' }}>
          <div style={styles.tracePanel}>
            <DecisionTrace trace={state.decisionTrace} />
          </div>
          <div style={styles.footer}>
            {new Date(state.timestamp).toLocaleString(undefined, {
              month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    minHeight: '100vh',
    padding: '22px 28px 28px',
    background: 'radial-gradient(circle at 50% -10%, #0b1222 0%, #020617 58%, #000 100%)',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
  },
  surface: {
    display: 'flex',
    flexDirection: 'column',
    gap: '18px',
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
    textTransform: 'uppercase',
    transition: 'opacity 0.62s ease',
  },
  heroSplit: {
    display: 'grid',
    gridTemplateColumns: 'minmax(320px, 1fr) 2fr',
    gap: '28px',
    alignItems: 'center',
    minHeight: '0',
  },
  leftColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: '18px',
    paddingTop: '8px',
  },
  rightColumn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '0',
  },
  meaningLine: {
    fontSize: '13px',
    color: 'rgba(147, 197, 253, 0.78)',
    letterSpacing: '0.04em',
    marginTop: '4px',
    transition: 'opacity 0.62s ease',
  },
  supportBand: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '21px',
    paddingTop: '4px',
  },
  supportItem: {
    minWidth: 0,
  },
  traceRow: {
    display: 'grid',
    gridTemplateColumns: 'minmax(320px, 1fr) 2fr',
    gap: '28px',
    alignItems: 'end',
    paddingBottom: '12px',
  },
  tracePanel: {
    paddingTop: '4px',
  },
  footer: {
    fontSize: '11px',
    color: 'rgba(203,213,225,0.34)',
    letterSpacing: '0.08em',
    textAlign: 'right',
    fontVariantNumeric: 'tabular-nums',
    paddingBottom: '8px',
    transition: 'opacity 0.62s ease',
    alignSelf: 'end',
  },
  simpleText: {
    color: 'rgba(226, 232, 240, 0.62)',
    fontSize: '13px',
  },
}
