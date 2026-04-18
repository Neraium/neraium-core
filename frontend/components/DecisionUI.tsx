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
  const ambientDim = 1 - severityTone * 0.10
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
        {/* Narrative — subdued, top anchor */}
        {narrative && (
          <div style={styles.narrative}>{narrative}</div>
        )}

        {/* Hero split: left decision stack + right tetrahedron */}
        <div style={styles.heroSplit}>
          <div style={styles.leftColumn}>
            <StatusHeader status={state.statusHeader} />
            <div style={styles.actionWrap}>
              <ActionPanel action={state.actionPanel} impactWindow={impactWindow} />
            </div>
            {meaningLine && <div style={styles.meaningLine}>{meaningLine}</div>}
            <div style={styles.reasoningWrap}>
              <DecisionTrace trace={state.decisionTrace} severity={state.statusHeader.severity} />
            </div>
          </div>

          <div style={styles.rightColumn}>
            <EnhancedTetrahedronViz tetrahedronState={state.tetrahedron} isInteractive={true} />
          </div>
        </div>

        {/* Support band: timeline + drift — visually connected */}
        <div style={{ ...styles.supportBand, opacity: ambientDim }}>
          <div style={styles.supportItem}>
            <SystemTimeline timeline={state.timeline} />
          </div>
          <div style={styles.supportDivider} />
          <div style={styles.supportItem}>
            {state.driftChart.dataPoints.length > 0 ? (
              <DriftChart chart={state.driftChart} minimal={isChartMinimal} severity={state.statusHeader.severity} />
            ) : (
              <div style={styles.simpleText}>No drift data</div>
            )}
          </div>
        </div>

        {/* Footer row — timestamp only, aligned */}
        <div style={{ ...styles.footerRow, opacity: 0.55 * ambientDim }}>
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
    padding: '28px 32px 32px',
    background: 'radial-gradient(circle at 50% -10%, #0b1222 0%, #020617 58%, #000 100%)',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
  },
  surface: {
    display: 'flex',
    flexDirection: 'column',
    gap: '22px',
    maxWidth: '1480px',
    margin: '0 auto',
    padding: '28px 32px 18px',
    borderRadius: '24px',
    transition: 'background 1.1s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  narrative: {
    fontSize: '11px',
    letterSpacing: '0.1em',
    color: 'rgba(148, 163, 184, 0.55)',
    textTransform: 'uppercase',
    transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  heroSplit: {
    display: 'flex',
    flexDirection: 'row',
    gap: '36px',
    alignItems: 'stretch',
    minHeight: '640px',
    paddingTop: '4px',
  },
  leftColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: '18px',
    paddingTop: '8px',
    paddingLeft: '4px',
    minWidth: '480px',
    maxWidth: '540px',
    flex: '0 0 auto',
  },
  actionWrap: {
    marginTop: '2px',
  },
  meaningLine: {
    fontSize: '13px',
    color: 'rgba(147, 197, 253, 0.72)',
    letterSpacing: '0.04em',
    marginTop: '-4px',
    transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  reasoningWrap: {
    marginTop: '8px',
    paddingTop: '14px',
    borderTop: '1px solid rgba(148, 163, 184, 0.12)',
  },
  rightColumn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: '480px',
    minHeight: '640px',
    flex: '1 1 auto',
    overflow: 'visible',
  },
  supportBand: {
    display: 'grid',
    gridTemplateColumns: '1fr auto 1fr',
    gap: '0',
    alignItems: 'start',
    paddingTop: '10px',
    borderTop: '1px solid rgba(148, 163, 184, 0.10)',
    transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  supportItem: {
    minWidth: 0,
    padding: '0 8px',
  },
  supportDivider: {
    width: '1px',
    backgroundColor: 'rgba(148, 163, 184, 0.10)',
    minHeight: '100%',
    alignSelf: 'stretch',
  },
  footerRow: {
    display: 'flex',
    justifyContent: 'flex-end',
    paddingTop: '4px',
    transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  footer: {
    fontSize: '10px',
    color: 'rgba(148, 163, 184, 0.38)',
    letterSpacing: '0.08em',
    textAlign: 'right',
    fontVariantNumeric: 'tabular-nums',
  },
  simpleText: {
    color: 'rgba(226, 232, 240, 0.62)',
    fontSize: '13px',
  },
}
