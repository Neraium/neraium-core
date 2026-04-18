'use client'

import { StatusHeader as StatusHeaderType, Severity, Trajectory, DegradationStage } from '@/lib/decisionToUI'

interface StatusHeaderProps {
  status: StatusHeaderType
}

export default function StatusHeader({ status }: StatusHeaderProps) {
  const getSeverityColor = (severity: Severity): string => {
    switch (severity) {
      case Severity.HIGH:
        return '#ef4444'
      case Severity.ELEVATED:
        return '#f97316'
      case Severity.MODERATE:
        return '#eab308'
      case Severity.LOW:
        return '#22c55e'
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
    return stage.replace(/_/g, ' ').toUpperCase()
  }

  const getTrajectoryIcon = (trajectory: Trajectory): string => {
    switch (trajectory) {
      case Trajectory.DEGRADING:
        return '↓'
      case Trajectory.UNCERTAIN:
        return '~'
      case Trajectory.STABLE:
        return '→'
    }
  }

  const getConfidenceColor = (confidence: string): string => {
    switch (confidence) {
      case 'HIGH':
        return '#3b82f6'
      case 'MEDIUM':
        return '#8b5cf6'
      case 'LOW':
        return '#ef4444'
      default:
        return '#6b7280'
    }
  }

  const severityColor = getSeverityColor(status.severity)
  const severityLabel = getSeverityLabel(status.severity)
  const stageLabel = getStageLabel(status.degradationStage)
  const trajectoryIcon = getTrajectoryIcon(status.trajectory)
  const confidenceColor = getConfidenceColor(status.confidence)

  return (
    <div style={styles.container}>
      {/* Severity indicator - most prominent */}
      <div style={styles.severitySection}>
        <div style={{ ...styles.severityBadge, borderColor: severityColor, backgroundColor: `${severityColor}15` }}>
          <div style={{ ...styles.severityDot, backgroundColor: severityColor }} />
          <span style={{ ...styles.severityText, color: severityColor }}>{severityLabel}</span>
        </div>
      </div>

      {/* Stage and trajectory info */}
      <div style={styles.infoRow}>
        <div style={styles.infoItem}>
          <div style={styles.label}>STAGE</div>
          <div style={styles.value}>{stageLabel}</div>
        </div>

        <div style={styles.infoItem}>
          <div style={styles.label}>TRAJECTORY</div>
          <div style={styles.value}>
            <span style={{ fontSize: '20px', marginRight: '4px' }}>{trajectoryIcon}</span>
            {status.trajectory}
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
    gap: '16px',
    padding: '20px',
    backgroundColor: 'rgba(30, 30, 30, 0.8)',
    borderRadius: '8px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  severitySection: {
    display: 'flex',
    justifyContent: 'center',
  },
  severityBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '12px 20px',
    border: '2px solid',
    borderRadius: '6px',
    minWidth: '180px',
    justifyContent: 'center',
  },
  severityDot: {
    width: '12px',
    height: '12px',
    borderRadius: '50%',
    animation: 'pulse 2s ease-in-out infinite',
  },
  severityText: {
    fontSize: '16px',
    fontWeight: '700',
    letterSpacing: '0.05em',
  },
  infoRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '16px',
  },
  infoItem: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '6px',
  },
  label: {
    fontSize: '11px',
    fontWeight: '600',
    letterSpacing: '0.08em',
    color: 'rgba(255, 255, 255, 0.5)',
    textTransform: 'uppercase' as const,
  },
  value: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#ffffff',
  },
}

// Add keyframe animation via style tag
const styleSheet = document.createElement('style')
styleSheet.textContent = `
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }
`
if (typeof document !== 'undefined') {
  document.head.appendChild(styleSheet)
}
