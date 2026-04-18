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
    [DegradationStage.BASELINE]: 'Baseline stability',
    [DegradationStage.EARLY_SHIFT]: 'Early shift',
    [DegradationStage.EMERGING]: 'Emerging degradation',
    [DegradationStage.PERSISTENT]: 'Persistent degradation',
    [DegradationStage.ACCELERATED]: 'Accelerated degradation',
    [DegradationStage.FAILURE_APPROACH]: 'Failure approach',
  }

  return (
    <div style={styles.container}>
      <div style={styles.stageLabel}>{stageLabel[status.degradationStage]}</div>
      <div style={{ ...styles.riskLine, color: severityColor }}>{status.severity} RISK</div>
      <div style={styles.trajectoryLine}>{status.trajectory.toUpperCase()}</div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    padding: '6px 4px 2px',
    transition: 'all 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  stageLabel: {
    fontSize: '13px',
    fontWeight: '600',
    letterSpacing: '0.1em',
    lineHeight: 1.2,
    color: 'rgba(148, 163, 184, 0.72)',
    textTransform: 'uppercase',
  },
  riskLine: {
    fontSize: '72px',
    fontWeight: '860',
    lineHeight: 0.9,
    letterSpacing: '-0.01em',
    textTransform: 'uppercase',
    transition: 'color 1.1s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  trajectoryLine: {
    fontSize: '15px',
    fontWeight: '560',
    letterSpacing: '0.18em',
    color: 'rgba(148, 163, 184, 0.55)',
    textTransform: 'uppercase',
    marginTop: '2px',
  },
}
