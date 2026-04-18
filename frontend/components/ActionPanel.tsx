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
    const timer = setTimeout(() => setShowCommand(true), 340)
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
          opacity: showCommand ? 1 : 0.05,
          transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      >
        {getHorizonLabel(action.horizon)}
      </div>
      <div
        style={{
          ...styles.actionText,
          opacity: showCommand ? 1 : 0.02,
          transform: showCommand ? 'translateY(0px)' : 'translateY(5px)',
          transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1), transform 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      >
        {action.primaryAction}
      </div>
      {impactWindow && (
        <div
          style={{
            ...styles.windowText,
            opacity: showCommand ? 0.78 : 0,
            transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
          }}
        >
          {impactWindow}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    padding: '10px 4px 8px',
    transition: 'all 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  horizonLabel: {
    fontSize: '22px',
    fontWeight: '780',
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    lineHeight: 1,
  },
  actionText: {
    marginTop: '4px',
    fontSize: '32px',
    fontWeight: '540',
    color: '#f8fafc',
    lineHeight: 1.1,
    maxWidth: '18ch',
  },
  windowText: {
    marginTop: '6px',
    fontSize: '12px',
    letterSpacing: '0.08em',
    color: 'rgba(148, 163, 184, 0.68)',
    textTransform: 'uppercase',
  },
}
