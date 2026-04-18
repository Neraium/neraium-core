'use client'

import { StatusHeader as StatusHeaderType, Severity, Trajectory, DegradationStage } from '@/lib/decisionToUI'

interface StatusHeaderProps {
  status: StatusHeaderType
}

export default function StatusHeader({ status }: StatusHeaderProps) {
  const getSeverityColor = (severity: Severity): string => {
    switch (severity) {
      case Severity.HIGH:
        return '#dc2626'
      case Severity.ELEVATED:
        return '#ea580c'
      case Severity.MODERATE:
        return '#ca8a04'
      case Severity.LOW:
        return '#16a34a'
    }
  }

  const getSeverityLabel = (severity: Severity): string => {
    switch (severity) {
      case Severity.HIGH:
        return 'CRITICAL'
      case Severity.ELEVATED:
        return 'HIGH'
      case Severity.MODERATE:
        return 'MODERATE'
      case Severity.LOW:
        return 'LOW'
    }
  }

  const getStageLabel = (stage: DegradationStage): string => {
    const labels: Record<DegradationStage, string> = {
      [DegradationStage.BASELINE]: 'Baseline',
      [DegradationStage.EARLY_SHIFT]: 'Early Shift',
      [DegradationStage.EMERGING]: 'Emerging',
      [DegradationStage.PERSISTENT]: 'Persistent',
      [DegradationStage.ACCELERATED]: 'Accelerated',
      [DegradationStage.FAILURE_APPROACH]: 'Failure Approach',
    }
    return labels[stage]
  }

  const getTrajectoryIcon = (trajectory: Trajectory): string => {
    switch (trajectory) {
      case Trajectory.DEGRADING:
        return '↓'
      case Trajectory.UNCERTAIN:
        return '≈'
      case Trajectory.STABLE:
        return '→'
    }
  }

  const getConfidenceColor = (confidence: string): string => {
    switch (confidence) {
      case 'HIGH':
        return '#0284c7'
      case 'MEDIUM':
        return '#7c3aed'
      case 'LOW':
        return '#dc2626'
      default:
        return '#6b7280'
    }
  }

  const severityColor = getSeverityColor(status.severity)
  const severityLabel = getSeverityLabel(status.severity)
  const stageLabel = getStageLabel(status.degradationStage)
  const trajectoryIcon = getTrajectoryIcon(status.trajectory)
  const confidenceColor = getConfidenceColor(status.confidence)
  const isPulsing = status.severity === Severity.HIGH || status.severity === Severity.ELEVATED

  return (
    <div style={styles.container}>
      <div style={styles.severitySection}>
        <div
          style={{
            ...styles.severityBadge,
            borderColor: severityColor,
            backgroundColor: `${severityColor}12`,
            boxShadow: isPulsing ? `0 0 24px ${severityColor}40` : 'none',
            transition: 'all 0.3s ease',
          }}
        >
          <div style={{ ...styles.severityDot, backgroundColor: severityColor, animation: isPulsing ? 'pulse-sm 2s ease-in-out infinite' : 'none' }} />
          <span style={{ ...styles.severityText, color: severityColor }}>{severityLabel}</span>
        </div>
      </div>

      <div style={styles.infoRow}>
        <div style={styles.infoItem}>
          <div style={styles.label}>STAGE</div>
          <div style={styles.value}>{stageLabel}</div>
        </div>

        <div style={styles.infoItem}>
          <div style={styles.label}>TRAJECTORY</div>
          <div style={styles.value}>
            <span style={styles.trajectoryIcon}>{trajectoryIcon}</span>
            <span>{status.trajectory}</span>
          </div>
        </div>

        <div style={styles.infoItem}>
          <div style={styles.label}>CONFIDENCE</div>
          <div style={{ ...styles.value, color: confidenceColor }}>{status.confidence}</div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '24px',
    padding: '24px',
    backgroundColor: 'rgba(15, 15, 15, 0.6)',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    backdropFilter: 'blur(4px)',
    transition: 'all 0.3s ease',
  },
  severitySection: {
    display: 'flex',
    justifyContent: 'center',
  },
  severityBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '14px 24px',
    border: '2px solid',
    borderRadius: '8px',
    minWidth: '200px',
    justifyContent: 'center',
    transition: 'all 0.3s ease',
  },
  severityDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    flexShrink: 0,
  },
  severityText: {
    fontSize: '15px',
    fontWeight: '700',
    letterSpacing: '0.06em',
  },
  infoRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '20px',
  },
  infoItem: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '8px',
  },
  label: {
    fontSize: '10px',
    fontWeight: '700',
    letterSpacing: '0.1em',
    color: 'rgba(255, 255, 255, 0.4)',
    textTransform: 'uppercase' as const,
  },
  value: {
    fontSize: '13px',
    fontWeight: '500',
    color: '#ffffff',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  trajectoryIcon: {
    fontSize: '16px',
    fontWeight: '600',
  },
}

if (typeof document !== 'undefined') {
  const style = document.createElement('style')
  style.textContent = `
    @keyframes pulse-sm {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }
  `
  document.head.appendChild(style)
}
