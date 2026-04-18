'use client'

import { StatusHeader as StatusHeaderType, Severity, DegradationStage } from '@/lib/decisionToUI'

interface StatusHeaderProps {
  status: StatusHeaderType
}

export default function StatusHeader({ status }: StatusHeaderProps) {
  const severityColor = status.severity === Severity.HIGH
    ? '#ef4444'
    : status.severity === Severity.ELEVATED
      ? '#f97316'
      : status.severity === Severity.MODERATE
        ? '#eab308'
        : '#22c55e'

  const stageLabel: Record<DegradationStage, string> = {
    [DegradationStage.BASELINE]: 'BASELINE STABILITY',
    [DegradationStage.EARLY_SHIFT]: 'EARLY SHIFT',
    [DegradationStage.EMERGING]: 'EMERGING DEGRADATION',
    [DegradationStage.PERSISTENT]: 'PERSISTENT DEGRADATION',
    [DegradationStage.ACCELERATED]: 'ACCELERATED DEGRADATION',
    [DegradationStage.FAILURE_APPROACH]: 'FAILURE APPROACH',
  }

  return (
    <div style={styles.container}>
      <div style={styles.line1}>{stageLabel[status.degradationStage]}</div>
      <div style={{ ...styles.line2, color: severityColor }}>{status.severity} RISK</div>
      <div style={styles.line3}>{status.trajectory.toUpperCase()}</div>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '15px',
    padding: '6px 0 2px',
    transition: 'all 0.62s ease',
  },
  line1: {
    fontSize: '44px',
    fontWeight: '710',
    letterSpacing: '0.02em',
    lineHeight: 1.01,
    color: 'rgba(248, 250, 252, 0.94)',
    textTransform: 'uppercase' as const,
  },
  line2: {
    fontSize: '66px',
    fontWeight: '840',
    lineHeight: 0.92,
    letterSpacing: '0.01em',
    textTransform: 'uppercase' as const,
    transition: 'color 0.62s ease',
  },
  line3: {
    fontSize: '22px',
    fontWeight: '580',
    letterSpacing: '0.14em',
    color: 'rgba(203, 213, 225, 0.62)',
    textTransform: 'uppercase' as const,
  },
}
