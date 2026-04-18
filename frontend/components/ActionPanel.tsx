'use client'

import { ActionPanel as ActionPanelType, ActionHorizon } from '@/lib/decisionToUI'

interface ActionPanelProps {
  action: ActionPanelType
  impactWindow?: string | null
}

export default function ActionPanel({ action, impactWindow }: ActionPanelProps) {
  const getHorizonColor = (horizon: ActionHorizon): string => {
    switch (horizon) {
      case ActionHorizon.NOW:
        return '#dc2626'
      case ActionHorizon.SOON:
        return '#ea580c'
      case ActionHorizon.WATCHLIST:
        return '#16a34a'
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
      <div style={{ ...styles.horizonLabel, color: horizonColor }}>{getHorizonLabel(action.horizon)}</div>
      <div style={styles.actionText}>{action.primaryAction}</div>
      {impactWindow && <div style={styles.windowText}>{impactWindow}</div>}
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '10px',
    padding: '24px 0 16px',
    borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
    transition: 'all 0.5s ease',
  },
  horizonLabel: {
    fontSize: '28px',
    fontWeight: '750',
    letterSpacing: '0.08em',
    textTransform: 'uppercase' as const,
    lineHeight: 1,
    transition: 'color 0.5s ease',
  },
  actionText: {
    fontSize: '34px',
    fontWeight: '500',
    color: '#f8fafc',
    lineHeight: 1.1,
    maxWidth: '18ch',
    transition: 'opacity 0.45s ease',
  },
  windowText: {
    marginTop: '6px',
    fontSize: '14px',
    letterSpacing: '0.04em',
    color: 'rgba(226, 232, 240, 0.74)',
    textTransform: 'uppercase' as const,
  },
}
