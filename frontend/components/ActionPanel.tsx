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

  // Map decision momentum to visual strength
  const getMomentumOpacity = (momentum?: string): number => {
    switch (momentum) {
      case 'strengthening':
        return 1.0
      case 'establishing':
        return 0.87
      case 'emerging':
      default:
        return 0.68
    }
  }

  // Map commitment score to glow intensity [0, 1]
  const getGlowIntensity = (score?: number): number => {
    return Math.max(0, Math.min(1, score ?? 0.3))
  }

  // Create subtle glow effect based on commitment
  const getGlowBlur = (intensity: number): string => {
    return `0 0 ${Math.round(4 + intensity * 8)}px rgba(248, 250, 252, ${intensity * 0.12})`
  }

  // Subtle letter-spacing tightening as commitment strengthens
  const getLetterSpacing = (momentum?: string): string => {
    switch (momentum) {
      case 'strengthening':
        return '0.04em'
      case 'establishing':
        return '0.06em'
      case 'emerging':
      default:
        return '0.08em'
    }
  }

  // Subtle vertical stability: emerging has slight bounce, strengthening is locked
  const getVerticalStability = (momentum?: string): string => {
    switch (momentum) {
      case 'strengthening':
        return '0px'
      case 'establishing':
        return '0.5px'
      case 'emerging':
      default:
        return '1px'
    }
  }

  const horizonColor = getHorizonColor(action.horizon)
  const momentumOpacity = getMomentumOpacity(action.decisionMomentum)
  const glowIntensity = getGlowIntensity(action.commitmentScore)
  const glowShadow = getGlowBlur(glowIntensity)
  const letterSpacing = getLetterSpacing(action.decisionMomentum)
  const verticalStability = getVerticalStability(action.decisionMomentum)

  return (
    <div style={styles.container}>
      <div
        style={{
          ...styles.horizonLabel,
          color: horizonColor,
          opacity: showCommand ? momentumOpacity : 0.05,
          letterSpacing: letterSpacing,
          transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1), letter-spacing 1.2s cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      >
        {getHorizonLabel(action.horizon)}
      </div>
      <div
        style={{
          ...styles.actionText,
          opacity: showCommand ? momentumOpacity : 0.02,
          transform: showCommand ? `translateY(${verticalStability})` : 'translateY(5px)',
          transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1), transform 1.1s cubic-bezier(0.22, 1, 0.36, 1), text-shadow 1.1s cubic-bezier(0.22, 1, 0.36, 1)',
          textShadow: showCommand ? glowShadow : 'none',
        }}
      >
        {action.primaryAction}
      </div>
      {impactWindow && (
        <div
          style={{
            ...styles.windowText,
            opacity: showCommand ? 0.78 * momentumOpacity : 0,
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
    padding: '10px 6px 8px',
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
