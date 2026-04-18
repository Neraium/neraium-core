'use client'

import { SystemTimeline as SystemTimelineType, DegradationStage } from '@/lib/decisionToUI'

interface SystemTimelineProps {
  timeline: SystemTimelineType
}

export default function SystemTimeline({ timeline }: SystemTimelineProps) {
  const getStageColor = (stage: DegradationStage, isNow: boolean): string => {
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
      <div style={styles.label}>Timeline</div>

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
        <span style={styles.currentLabel}>Now:</span>
        <span style={styles.currentStageName}>{timeline.stages[timeline.currentIndex].label}</span>
      </div>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '20px',
    padding: '18px 0',
    transition: 'all 0.45s ease',
  },
  label: {
    fontSize: '11px',
    fontWeight: '700',
    letterSpacing: '0.1em',
    color: 'rgba(255, 255, 255, 0.4)',
    textTransform: 'uppercase' as const,
  },
  timelineTrack: {
    display: 'grid',
    gridTemplateColumns: 'repeat(6, minmax(56px, 1fr))',
    gap: '0px',
    alignItems: 'center',
  },
  timelineItem: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    gap: '10px',
    position: 'relative',
    flex: 1,
  },
  connector: {
    position: 'absolute',
    top: '-14px',
    width: '100%',
    height: '2px',
    zIndex: 0,
    transition: 'background-color 0.3s ease',
  },
  stageDot: {
    width: '12px',
    height: '12px',
    borderRadius: '50%',
    border: '2px solid',
    zIndex: 1,
    transition: 'all 0.3s ease',
  },
  stageLabel: {
    fontSize: '9px',
    fontWeight: '500',
    letterSpacing: '0.06em',
    textAlign: 'center' as const,
    lineHeight: '1.4',
    maxWidth: '65px',
    textTransform: 'capitalize' as const,
  },
  currentStageInfo: {
    display: 'flex',
    gap: '10px',
    padding: '14px 0',
    borderTop: '1px solid rgba(148, 163, 184, 0.2)',
    fontSize: '11px',
    alignItems: 'center',
  },
  currentLabel: {
    color: 'rgba(255, 255, 255, 0.4)',
    fontWeight: '700',
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
  },
  currentStageName: {
    color: '#f5f5f5',
    fontWeight: '500',
  },
}
