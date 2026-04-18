'use client'

import { DecisionTrace as DecisionTraceType, Severity } from '@/lib/decisionToUI'

interface DecisionTraceProps {
  trace: DecisionTraceType
  severity?: Severity
}

const compress = (text: string) =>
  text
    .replace('Structural degradation is persistent', 'Degradation persists')
    .replace('Degradation is accelerating', 'Accelerating')
    .replace('Pattern weak; using live evidence', 'Pattern weak')
    .replace('Confidence level: ', '')

const confidenceBadge = (rationale: string, severity?: Severity): { label: string; tone: string } => {
  const raw = compress(rationale).toUpperCase()
  if (raw.includes('HIGH')) return { label: 'HIGH', tone: severity === Severity.HIGH || severity === Severity.ELEVATED ? '#ef4444' : '#22c55e' }
  if (raw.includes('MEDIUM')) return { label: 'MEDIUM', tone: '#eab308' }
  return { label: 'LOW', tone: '#94a3b8' }
}

export default function DecisionTrace({ trace, severity }: DecisionTraceProps) {
  const factors = trace.secondaryFactors.slice(0, 2).map((item) => compress(item))
  const badge = confidenceBadge(trace.confidenceRationale, severity)

  return (
    <div style={styles.container}>
      <div style={styles.section}>
        <div style={styles.label}>Driver</div>
        <div style={styles.primary}>{compress(trace.primaryFactor)}</div>
      </div>

      {factors.length > 0 && (
        <div style={styles.section}>
          <div style={styles.label}>Signals</div>
          <div style={styles.factorsList}>
            {factors.map((factor, idx) => (
              <div key={idx} style={styles.factorItem}>{factor}</div>
            ))}
          </div>
        </div>
      )}

      {trace.patternInsight && (
        <div style={styles.section}>
          <div style={styles.label}>Pattern</div>
          <div style={styles.patternLine}>{compress(trace.patternInsight.influenceSummary)}</div>
        </div>
      )}

      <div style={styles.badgeRow}>
        <div style={styles.label}>Confidence</div>
        <div style={{ ...styles.badge, color: badge.tone, borderColor: badge.tone }}>{badge.label}</div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
    padding: '2px 6px 4px',
    maxWidth: '420px',
    transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
  },
  label: {
    fontSize: '10px',
    letterSpacing: '0.14em',
    color: 'rgba(148, 163, 184, 0.62)',
    textTransform: 'uppercase',
    fontWeight: '700',
  },
  primary: {
    fontSize: '16px',
    lineHeight: 1.35,
    color: '#f8fafc',
    fontWeight: '500',
  },
  factorsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  factorItem: {
    color: 'rgba(203, 213, 225, 0.78)',
    fontSize: '14px',
    paddingLeft: '12px',
    position: 'relative',
  },
  patternLine: {
    color: 'rgba(253, 186, 116, 0.88)',
    fontSize: '14px',
    fontWeight: '500',
  },
  badgeRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginTop: '2px',
  },
  badge: {
    fontSize: '10px',
    fontWeight: '780',
    letterSpacing: '0.1em',
    padding: '3px 10px',
    borderRadius: '4px',
    border: '1px solid',
    backgroundColor: 'rgba(2, 6, 23, 0.35)',
    textTransform: 'uppercase',
  },
}
