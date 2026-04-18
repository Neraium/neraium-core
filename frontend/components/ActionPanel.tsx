'use client'

import { useEffect, useState } from 'react'
import { ActionPanel as ActionPanelType, ActionHorizon } from '@/lib/decisionToUI'

interface ActionPanelProps {
  action: ActionPanelType
  impactWindow?: string | null
}

export default function ActionPanel({ action, impactWindow }: ActionPanelProps) {
  const isCritical = action.horizon === ActionHorizon.NOW && action.urgencyLevel >= 5
  const [showCommand, setShowCommand] = useState(true)

  useEffect(() => {
    if (!isCritical) {
      setShowCommand(true)
      return
    }

    setShowCommand(false)
    const timer = setTimeout(() => setShowCommand(true), 240)
    return () => clearTimeout(timer)
  }, [isCritical, action.primaryAction])

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
          animation: isCritical ? 'commandPulse 3.4s ease-in-out infinite' : 'none',
          opacity: showCommand ? 1 : 0.05,
        }}
      >
        {getHorizonLabel(action.horizon)}
      </div>
      <div
        style={{
          ...styles.actionText,
          opacity: showCommand ? 1 : 0.02,
          transform: showCommand ? 'translateY(0px)' : 'translateY(4px)',
        }}
      >
        {action.primaryAction}
      </div>
      {impactWindow && <div style={{ ...styles.windowText, opacity: showCommand ? 0.9 : 0 }}>{impactWindow}</div>}
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '12px',
    padding: '18px 0 16px',
    transition: 'all 0.62s ease',
  },
  horizonLabel: {
    fontSize: '30px',
    fontWeight: '780',
    letterSpacing: '0.1em',
    textTransform: 'uppercase' as const,
    lineHeight: 1,
    transition: 'color 0.62s ease, opacity 0.62s ease',
  },
  actionText: {
    marginTop: '10px',
    fontSize: '40px',
    fontWeight: '540',
    color: '#f8fafc',
    lineHeight: 1.1,
    maxWidth: '18ch',
    transition: 'opacity 0.62s ease, transform 0.62s ease',
    letterSpacing: '0.01em',
  },
  windowText: {
    marginTop: '10px',
    fontSize: '13px',
    letterSpacing: '0.08em',
    color: 'rgba(203, 213, 225, 0.80)',
    textTransform: 'uppercase' as const,
    transition: 'opacity 0.62s ease',
    fontWeight: '500',
  },
}

if (typeof document !== 'undefined' && !document.getElementById('action-panel-motion')) {
  const style = document.createElement('style')
  style.id = 'action-panel-motion'
  style.textContent = `
    @keyframes commandPulse {
      0%, 100% { opacity: 1; transform: translateY(0px) scale(1); }
      50% { opacity: 0.96; transform: translateY(-2px) scale(1.01); }
    }
  `
  document.head.appendChild(style)
}
