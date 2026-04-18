'use client'

import { DecisionTrace as DecisionTraceType } from '@/lib/decisionToUI'

interface DecisionTraceProps {
  trace: DecisionTraceType
}

const compress = (text: string) =>
  text
    .replace('Structural degradation is persistent', 'Degradation persists')
    .replace('Degradation is accelerating', 'Accelerating')
    .replace('Pattern weak; using live evidence', 'Pattern weak')
    .replace('Confidence level: ', 'Confidence ')

export default function DecisionTrace({ trace }: DecisionTraceProps) {
  const factors = trace.secondaryFactors.slice(0, 2).map((item) => compress(item))

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
              <div key={idx} style={styles.factorItem}>• {factor}</div>
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

      <div style={styles.rationale}>{compress(trace.confidenceRationale)}</div>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '15px',
    padding: '18px 0 5px',
    transition: 'opacity 0.62s ease',
  },
  section: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '6px',
  },
  label: {
    fontSize: '11px',
    letterSpacing: '0.12em',
    color: 'rgba(203, 213, 225, 0.54)',
    textTransform: 'uppercase' as const,
    fontWeight: '600',
  },
  primary: {
    fontSize: '18px',
    lineHeight: 1.35,
    color: '#f8fafc',
  },
  factorsList: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '5px',
  },
  factorItem: {
    color: 'rgba(226, 232, 240, 0.72)',
    fontSize: '15px',
  },
  patternLine: {
    color: 'rgba(253, 186, 116, 0.92)',
    fontSize: '15px',
  },
  rationale: {
    marginTop: '2px',
    fontSize: '12px',
    color: 'rgba(148, 163, 184, 0.82)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
  },
}
