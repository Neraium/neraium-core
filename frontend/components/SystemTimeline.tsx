'use client'

import { SystemTimeline as SystemTimelineType, DegradationStage } from '@/lib/decisionToUI'

interface SystemTimelineProps {
  timeline: SystemTimelineType
}

export default function SystemTimeline({ timeline }: SystemTimelineProps) {
  const getStageColor = (stage: DegradationStage, isCurrent: boolean): string => {
    const colorMap: Record<DegradationStage, string> = {
      [DegradationStage.BASELINE]: '#22c55e',
      [DegradationStage.EARLY_SHIFT]: '#eab308',
      [DegradationStage.EMERGING]: '#f59e0b',
      [DegradationStage.PERSISTENT]: '#f97316',
      [DegradationStage.ACCELERATED]: '#ef4444',
      [DegradationStage.FAILURE_APPROACH]: '#dc2626',
    }
    return colorMap[stage] || '#6b7280'
  }

  return (
    <div style={styles.container}>
      <div style={styles.label}>DEGRADATION TIMELINE</div>

      <div style={styles.timelineTrack}>
        {timeline.stages.map((stage, idx) => {
          const color = getStageColor(stage.stage, stage.isCurrent)
          const isCompleted = idx < timeline.currentIndex
          const isCurrent = stage.isCurrent

          return (
            <div key={idx} style={styles.timelineItem}>
              {/* Connector line to next */}
              {idx < timeline.stages.length - 1 && (
                <div
                  style={{
                    ...styles.connector,
                    backgroundColor: isCompleted || isCurrent ? color : 'rgba(255, 255, 255, 0.1)',
                  }}
                />
              )}

              {/* Stage dot */}
              <div
                style={{
                  ...styles.stageDot,
                  backgroundColor: isCurrent ? color : isCompleted ? color : 'rgba(255, 255, 255, 0.1)',
                  borderColor: isCurrent ? color : 'rgba(255, 255, 255, 0.2)',
                  boxShadow: isCurrent ? `0 0 12px ${color}80` : 'none',
                }}
              />

              {/* Stage label */}
              <div
                style={{
                  ...styles.stageLabel,
                  color: isCurrent ? color : isCompleted ? 'rgba(255, 255, 255, 0.6)' : 'rgba(255, 255, 255, 0.3)',
                  fontWeight: isCurrent ? '600' : '500',
                }}
              >
                {stage.label}
              </div>
            </div>
          )
        })}
      </div>

      {/* Current stage indicator */}
      <div style={styles.currentStageInfo}>
        <span style={styles.currentLabel}>Current:</span>
        <span style={styles.currentStageName}>{timeline.stages[timeline.currentIndex].label}</span>
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
  label: {
    fontSize: '11px',
    fontWeight: '600',
    letterSpacing: '0.08em',
    color: 'rgba(255, 255, 255, 0.5)',
    textTransform: 'uppercase' as const,
  },
  timelineTrack: {
    display: 'grid',
    gridTemplateColumns: 'repeat(6, 1fr)',
    gap: '2px',
    alignItems: 'center',
  },
  timelineItem: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    gap: '8px',
    position: 'relative',
    flex: 1,
  },
  connector: {
    position: 'absolute',
    top: '-12px',
    width: '100%',
    height: '2px',
    zIndex: 0,
  },
  stageDot: {
    width: '14px',
    height: '14px',
    borderRadius: '50%',
    border: '2px solid',
    zIndex: 1,
    transition: 'all 0.3s ease',
  },
  stageLabel: {
    fontSize: '10px',
    fontWeight: '500',
    letterSpacing: '0.05em',
    textAlign: 'center' as const,
    lineHeight: '1.3',
    maxWidth: '70px',
    textTransform: 'uppercase' as const,
  },
  currentStageInfo: {
    display: 'flex',
    gap: '8px',
    padding: '10px 0',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
    fontSize: '12px',
  },
  currentLabel: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontWeight: '600',
  },
  currentStageName: {
    color: '#ffffff',
    fontWeight: '600',
  },
}
