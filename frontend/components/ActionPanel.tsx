'use client'

import { ActionPanel as ActionPanelType, ActionHorizon } from '@/lib/decisionToUI'

interface ActionPanelProps {
  action: ActionPanelType
}

export default function ActionPanel({ action }: ActionPanelProps) {
  const getHorizonColor = (horizon: ActionHorizon): string => {
    switch (horizon) {
      case ActionHorizon.NOW:
        return '#ef4444'
      case ActionHorizon.SOON:
        return '#f97316'
      case ActionHorizon.WATCHLIST:
        return '#8b5cf6'
    }
  }

  const getHorizonLabel = (horizon: ActionHorizon): string => {
    switch (horizon) {
      case ActionHorizon.NOW:
        return 'IMMEDIATE'
      case ActionHorizon.SOON:
        return 'NEAR TERM'
      case ActionHorizon.WATCHLIST:
        return 'MONITOR'
    }
  }

  const horizonColor = getHorizonColor(action.horizon)
  const horizonLabel = getHorizonLabel(action.horizon)

  // Build urgency visual (5-level bar)
  const urgencyBars = Array.from({ length: 5 }, (_, i) => i < action.urgencyLevel)

  return (
    <div style={styles.container}>
      {/* Horizon badge */}
      <div style={{ ...styles.horizonBadge, backgroundColor: `${horizonColor}20`, borderColor: horizonColor }}>
        <span style={{ ...styles.horizonLabel, color: horizonColor }}>{horizonLabel}</span>
      </div>

      {/* Primary action text */}
      <div style={styles.actionSection}>
        <div style={styles.actionLabel}>RECOMMENDED ACTION</div>
        <div style={styles.actionText}>{action.primaryAction}</div>
      </div>

      {/* Urgency indicator */}
      <div style={styles.urgencySection}>
        <div style={styles.urgencyLabel}>URGENCY</div>
        <div style={styles.urgencyBars}>
          {urgencyBars.map((isActive, idx) => (
            <div
              key={idx}
              style={{
                ...styles.urgencyBar,
                backgroundColor: isActive ? horizonColor : 'rgba(255, 255, 255, 0.1)',
                opacity: isActive ? 1 : 0.3,
              }}
            />
          ))}
        </div>
        <div style={styles.urgencyValue}>{action.urgencyLevel}/5</div>
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
  horizonBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '8px 16px',
    borderRadius: '4px',
    border: '1px solid',
    width: 'fit-content',
  },
  horizonLabel: {
    fontSize: '12px',
    fontWeight: '700',
    letterSpacing: '0.05em',
    textTransform: 'uppercase' as const,
  },
  actionSection: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '8px',
  },
  actionLabel: {
    fontSize: '11px',
    fontWeight: '600',
    letterSpacing: '0.08em',
    color: 'rgba(255, 255, 255, 0.5)',
    textTransform: 'uppercase' as const,
  },
  actionText: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#ffffff',
    lineHeight: '1.5',
  },
  urgencySection: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  urgencyLabel: {
    fontSize: '11px',
    fontWeight: '600',
    letterSpacing: '0.08em',
    color: 'rgba(255, 255, 255, 0.5)',
    textTransform: 'uppercase' as const,
    minWidth: '70px',
  },
  urgencyBars: {
    display: 'flex',
    gap: '4px',
    flex: 1,
  },
  urgencyBar: {
    flex: 1,
    height: '6px',
    borderRadius: '2px',
    transition: 'all 0.3s ease',
  },
  urgencyValue: {
    fontSize: '12px',
    fontWeight: '600',
    color: 'rgba(255, 255, 255, 0.6)',
    minWidth: '40px',
    textAlign: 'right' as const,
  },
}
