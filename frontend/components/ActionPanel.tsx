'use client'

import { ActionPanel as ActionPanelType, ActionHorizon } from '@/lib/decisionToUI'

interface ActionPanelProps {
  action: ActionPanelType
}

export default function ActionPanel({ action }: ActionPanelProps) {
  const getHorizonColor = (horizon: ActionHorizon): string => {
    switch (horizon) {
      case ActionHorizon.NOW:
        return '#dc2626'
      case ActionHorizon.SOON:
        return '#ea580c'
      case ActionHorizon.WATCHLIST:
        return '#7c3aed'
    }
  }

  const getHorizonLabel = (horizon: ActionHorizon): string => {
    switch (horizon) {
      case ActionHorizon.NOW:
        return 'Immediate'
      case ActionHorizon.SOON:
        return 'Near term'
      case ActionHorizon.WATCHLIST:
        return 'Monitor'
    }
  }

  const horizonColor = getHorizonColor(action.horizon)
  const horizonLabel = getHorizonLabel(action.horizon)
  const urgencyBars = Array.from({ length: 5 }, (_, i) => i < action.urgencyLevel)

  return (
    <div style={styles.container}>
      <div
        style={{
          ...styles.horizonBadge,
          backgroundColor: `${horizonColor}15`,
          borderColor: horizonColor,
        }}
      >
        <span style={{ ...styles.horizonLabel, color: horizonColor }}>{horizonLabel}</span>
      </div>

      <div style={styles.actionSection}>
        <div style={styles.actionLabel}>ACTION</div>
        <div style={styles.actionText}>{action.primaryAction}</div>
      </div>

      <div style={styles.urgencySection}>
        <div style={styles.urgencyLabel}>URGENCY</div>
        <div style={styles.urgencyBars}>
          {urgencyBars.map((isActive, idx) => (
            <div
              key={idx}
              style={{
                ...styles.urgencyBar,
                backgroundColor: isActive ? horizonColor : 'rgba(255, 255, 255, 0.06)',
                transition: 'all 0.3s ease',
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
    gap: '20px',
    padding: '24px',
    backgroundColor: 'rgba(15, 15, 15, 0.6)',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    backdropFilter: 'blur(4px)',
    transition: 'all 0.3s ease',
  },
  horizonBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '10px 16px',
    borderRadius: '8px',
    border: '1px solid',
    width: 'fit-content',
    transition: 'all 0.3s ease',
  },
  horizonLabel: {
    fontSize: '11px',
    fontWeight: '700',
    letterSpacing: '0.06em',
    textTransform: 'capitalize' as const,
  },
  actionSection: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '10px',
  },
  actionLabel: {
    fontSize: '10px',
    fontWeight: '700',
    letterSpacing: '0.1em',
    color: 'rgba(255, 255, 255, 0.4)',
    textTransform: 'uppercase' as const,
  },
  actionText: {
    fontSize: '14px',
    fontWeight: '400',
    color: '#f5f5f5',
    lineHeight: '1.6',
  },
  urgencySection: {
    display: 'flex',
    alignItems: 'center',
    gap: '14px',
    paddingTop: '8px',
    borderTop: '1px solid rgba(255, 255, 255, 0.08)',
  },
  urgencyLabel: {
    fontSize: '10px',
    fontWeight: '700',
    letterSpacing: '0.1em',
    color: 'rgba(255, 255, 255, 0.4)',
    textTransform: 'uppercase' as const,
    minWidth: '60px',
  },
  urgencyBars: {
    display: 'flex',
    gap: '5px',
    flex: 1,
  },
  urgencyBar: {
    flex: 1,
    height: '5px',
    borderRadius: '1px',
  },
  urgencyValue: {
    fontSize: '11px',
    fontWeight: '600',
    color: 'rgba(255, 255, 255, 0.5)',
    minWidth: '38px',
    textAlign: 'right' as const,
    fontVariantNumeric: 'tabular-nums',
  },
}
