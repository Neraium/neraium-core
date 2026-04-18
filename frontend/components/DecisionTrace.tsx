'use client'

import { DecisionTrace as DecisionTraceType } from '@/lib/decisionToUI'

interface DecisionTraceProps {
  trace: DecisionTraceType
}

export default function DecisionTrace({ trace }: DecisionTraceProps) {
  const getPatternTierColor = (tier: string): string => {
    switch (tier) {
      case 'moderate':
        return '#f59e0b'
      case 'strong':
        return '#ef4444'
      default:
        return '#6b7280'
    }
  }

  const getPatternTierLabel = (tier: string): string => {
    switch (tier) {
      case 'moderate':
        return 'MODERATE MATCH'
      case 'strong':
        return 'STRONG MATCH'
      default:
        return 'WEAK MATCH'
    }
  }

  return (
    <div style={styles.container}>
      {/* Primary factor - most important */}
      <div style={styles.section}>
        <div style={styles.sectionLabel}>PRIMARY FACTOR</div>
        <div style={styles.primaryFactor}>{trace.primaryFactor}</div>
      </div>

      {/* Secondary factors */}
      {trace.secondaryFactors.length > 0 && (
        <div style={styles.section}>
          <div style={styles.sectionLabel}>SUPPORTING EVIDENCE</div>
          <div style={styles.factorsList}>
            {trace.secondaryFactors.map((factor, idx) => (
              <div key={idx} style={styles.factorItem}>
                <span style={styles.factorBullet}>•</span>
                <span>{factor}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pattern insight if available */}
      {trace.patternInsight && (
        <div style={styles.section}>
          <div style={styles.sectionLabel}>PATTERN RECOGNITION</div>
          <div
            style={{
              ...styles.patternCard,
              borderLeftColor: getPatternTierColor(trace.patternInsight.tier),
            }}
          >
            <div style={styles.patternTierBadge}>
              <span style={{ color: getPatternTierColor(trace.patternInsight.tier) }}>
                {getPatternTierLabel(trace.patternInsight.tier)}
              </span>
            </div>
            <div style={styles.patternType}>
              <span style={styles.patternTypeLabel}>Outcome:</span>
              <span>{trace.patternInsight.outcomeType}</span>
            </div>
            {trace.patternInsight.influenceSummary && (
              <div style={styles.patternSummary}>{trace.patternInsight.influenceSummary}</div>
            )}
          </div>
        </div>
      )}

      {/* Confidence rationale */}
      <div style={styles.section}>
        <div style={styles.sectionLabel}>CONFIDENCE BASIS</div>
        <div style={styles.rationale}>{trace.confidenceRationale}</div>
      </div>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '16px',
    padding: '20px',
    backgroundColor: 'rgba(30, 30, 30, 0.8)',
    borderRadius: '8px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  section: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '8px',
  },
  sectionLabel: {
    fontSize: '11px',
    fontWeight: '600',
    letterSpacing: '0.08em',
    color: 'rgba(255, 255, 255, 0.5)',
    textTransform: 'uppercase' as const,
  },
  primaryFactor: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#ffffff',
    lineHeight: '1.6',
    padding: '8px 0',
  },
  factorsList: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '6px',
  },
  factorItem: {
    display: 'flex',
    gap: '8px',
    fontSize: '13px',
    color: 'rgba(255, 255, 255, 0.8)',
    alignItems: 'flex-start',
  },
  factorBullet: {
    color: 'rgba(255, 255, 255, 0.3)',
    minWidth: '16px',
  },
  patternCard: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '10px',
    padding: '12px',
    borderLeft: '3px solid',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: '4px',
  },
  patternTierBadge: {
    fontSize: '11px',
    fontWeight: '700',
    letterSpacing: '0.05em',
    textTransform: 'uppercase' as const,
  },
  patternType: {
    display: 'flex',
    gap: '8px',
    fontSize: '13px',
    color: 'rgba(255, 255, 255, 0.8)',
  },
  patternTypeLabel: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontWeight: '600',
  },
  patternSummary: {
    fontSize: '12px',
    color: 'rgba(255, 255, 255, 0.6)',
    lineHeight: '1.5',
  },
  rationale: {
    fontSize: '13px',
    color: 'rgba(255, 255, 255, 0.7)',
    lineHeight: '1.6',
    padding: '8px 0',
  },
}
