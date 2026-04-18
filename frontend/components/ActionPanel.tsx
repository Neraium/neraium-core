'use client'

import { ActionPanel as ActionPanelType, ActionHorizon } from '@/lib/decisionToUI'

interface ActionPanelProps {
  action: ActionPanelType
  impactWindow?: string | null
}

export default function ActionPanel({ action, impactWindow }: ActionPanelProps) {
  const isCritical = action.horizon === ActionHorizon.NOW && action.urgencyLevel >= 5

  const getHorizonColor = (horizon: ActionHorizon): string => {
    switch (horizon) {
      case ActionHorizon.NOW:
        return '#ef4444'
      case ActionHorizon.SOON:
        return '#f97316'
      case ActionHorizon.WATCHLIST:
        return '#22c55e'
    }
  }

  const getHorizonLabel = (horizon: ActionHorizon): string => {
    switch (horizon) {
      case ActionHorizon.NOW:
        return 'ACT NOW'
      case ActionHorizon.SOON:
        return 'PLAN ACTION'
      case ActionHorizon.WATCHLIST:
        return 'MONITOR'
    }
  }

  const horizonColor = getHorizonColor(action.horizon)

  return (
    <div style={styles.container}>
      <div
        style={{
          ...styles.horizonLabel,
          color: horizonColor,
          animation: isCritical ? 'commandPulse 2.8s ease-in-out infinite' : 'none',
        }}
      >
        {getHorizonLabel(action.horizon)}
      </div>
      <div style={styles.actionText}>{action.primaryAction}</div>
      {impactWindow && <div style={styles.windowText}>{impactWindow}</div>}
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '12px',
    padding: '18px 0 16px',
    transition: 'all 0.5s ease',
  },
  horizonLabel: {
    fontSize: '30px',
    fontWeight: '780',
    letterSpacing: '0.1em',
    textTransform: 'uppercase' as const,
    lineHeight: 1,
    transition: 'color 0.5s ease',
  },
  actionText: {
    marginTop: '8px',
    fontSize: '38px',
    fontWeight: '520',
    color: '#f8fafc',
    lineHeight: 1.05,
    maxWidth: '18ch',
    transition: 'opacity 0.45s ease',
  },
  windowText: {
    marginTop: '8px',
    fontSize: '13px',
    letterSpacing: '0.08em',
    color: 'rgba(203, 213, 225, 0.74)',
    textTransform: 'uppercase' as const,
  },
}

if (typeof document !== 'undefined' && !document.getElementById('action-panel-motion')) {
  const style = document.createElement('style')
  style.id = 'action-panel-motion'
  style.textContent = `
    @keyframes commandPulse {
      0%, 100% { opacity: 1; transform: translateY(0px); }
      50% { opacity: 0.92; transform: translateY(-1px); }
    }
  `
  document.head.appendChild(style)
}
