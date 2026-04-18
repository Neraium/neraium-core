'use client'

import { SystemTimeline as SystemTimelineType, DegradationStage } from '@/lib/decisionToUI'

interface SystemTimelineProps {
  timeline: SystemTimelineType
}

export default function SystemTimeline({ timeline }: SystemTimelineProps) {
  const getStageColor = (stage: DegradationStage): string => {
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
      <div style={styles.label}>Progression</div>

      <div style={styles.timelineTrack}>
        {timeline.stages.map((stage, idx) => {
          const color = getStageColor(stage.stage)
          const isCompleted = idx < timeline.currentIndex
          const isCurrent = stage.isCurrent
          const isFuture = idx > timeline.currentIndex

          return (
            <div key={idx} style={styles.timelineItem}>
              {idx < timeline.stages.length - 1 && (
                <div
                  style={{
                    ...styles.connector,
                    backgroundColor: isCompleted ? color : 'rgba(148, 163, 184, 0.12)',
                    opacity: isCurrent ? 1 : isCompleted ? 0.7 : 0.35,
                  }}
                />
              )}

              <div
                style={{
                  ...styles.stageDot,
                  backgroundColor: isCurrent ? color : isCompleted ? color : 'rgba(148, 163, 184, 0.1)',
                  borderColor: isCurrent ? color : isCompleted ? `${color}66` : 'rgba(148, 163, 184, 0.2)',
                  boxShadow: isCurrent ? `0 0 14px ${color}88, 0 0 4px ${color}44` : 'none',
                  transform: isCurrent ? 'scale(1.25)' : 'scale(1)',
                }}
              />

              <div
                style={{
                  ...styles.stageLabel,
                  color: isCurrent ? color : isCompleted ? 'rgba(226, 232, 240, 0.55)' : 'rgba(148, 163, 184, 0.38)',
                  fontWeight: isCurrent ? '700' : '500',
                }}
              >
                {stage.label}
              </div>
            </div>
          )
        })}
      </div>

      <div style={styles.nowAnchor}>
        <span style={styles.nowDot} />
        <span style={styles.nowLabel}>Now</span>
        <span style={styles.nowStage}>{timeline.stages[timeline.currentIndex].label}</span>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    padding: '10px 0',
    transition: 'all 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  label: {
    fontSize: '10px',
    fontWeight: '700',
    letterSpacing: '0.14em',
    color: 'rgba(148, 163, 184, 0.55)',
    textTransform: 'uppercase',
  },
  timelineTrack: {
    display: 'grid',
    gridTemplateColumns: 'repeat(6, minmax(52px, 1fr))',
    alignItems: 'center',
    gap: '2px',
  },
  timelineItem: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '8px',
    position: 'relative',
  },
  connector: {
    position: 'absolute',
    top: '6px',
    left: '50%',
    width: '100%',
    height: '2px',
    zIndex: 0,
    transition: 'all 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  stageDot: {
    width: '12px',
    height: '12px',
    borderRadius: '50%',
    border: '2px solid',
    zIndex: 1,
    transition: 'all 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  stageLabel: {
    fontSize: '9px',
    letterSpacing: '0.06em',
    textAlign: 'center',
    lineHeight: '1.35',
    maxWidth: '64px',
    textTransform: 'capitalize',
    transition: 'all 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  nowAnchor: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    paddingTop: '6px',
    fontSize: '11px',
  },
  nowDot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    backgroundColor: '#f8fafc',
    opacity: 0.7,
  },
  nowLabel: {
    color: 'rgba(148, 163, 184, 0.65)',
    fontWeight: '700',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
  },
  nowStage: {
    color: '#f8fafc',
    fontWeight: '600',
    letterSpacing: '0.02em',
  },
}
