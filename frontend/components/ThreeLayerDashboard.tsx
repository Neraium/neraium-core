'use client'

import { useState, useEffect, useMemo } from 'react'
import SystemOverview from '@/components/SystemOverview'
import SystemStateTrajectory from '@/components/SystemStateTrajectory'
import ActionRecommendations from '@/components/ActionRecommendations'
import { DEMO_SCENARIOS } from '@/lib/demoScenarios'
import { DecisionUIState } from '@/lib/decisionToUI'

export default function ThreeLayerDashboard() {
  const [scenarioIndex, setScenarioIndex] = useState(0)
  const [isAutoPlay, setIsAutoPlay] = useState(true)
  const [isMounted, setIsMounted] = useState(false)

  const scenario = DEMO_SCENARIOS[scenarioIndex]
  const state = scenario.state

  useEffect(() => {
    setIsMounted(true)
  }, [])

  useEffect(() => {
    if (!isAutoPlay) return
    if (scenarioIndex >= DEMO_SCENARIOS.length - 1) {
      setIsAutoPlay(false)
      return
    }
    const timer = setTimeout(() => {
      setScenarioIndex(i => i + 1)
    }, 8000)
    return () => clearTimeout(timer)
  }, [scenarioIndex, isAutoPlay])

  const systemHealthMetrics = useMemo(() => ({
    healthy: Math.max(1, 15 - scenarioIndex * 2),
    drifting: Math.min(5, scenarioIndex + 1),
    unstable: Math.max(0, Math.floor(scenarioIndex / 2)),
    earlyWarnings: Math.max(3, 8 - scenarioIndex),
    enteringInstability: Math.max(1, 4 - Math.floor(scenarioIndex / 2)),
  }), [scenarioIndex])

  return (
    <div style={styles.root}>
      <div style={styles.surface}>
        {/* Layer 1: System Overview */}
        <div style={styles.layer}>
          <SystemOverview metrics={systemHealthMetrics} />
        </div>

        {/* Layer 2: System State & Trajectory */}
        <div style={styles.layer}>
          <SystemStateTrajectory
            state={state}
            systemName={scenario.name}
          />
        </div>

        {/* Layer 3: Action Recommendations */}
        <div style={styles.layer}>
          <ActionRecommendations state={state} />
        </div>

        {/* Footer with navigation */}
        <div style={styles.footer}>
          <div style={styles.footerContent}>
            <div style={styles.timestamp}>
              {isMounted ? new Date().toLocaleString(undefined, {
                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
              }) : '\u00A0'}
            </div>
            <div style={styles.scenarioInfo}>
              Scenario {scenarioIndex + 1} of {DEMO_SCENARIOS.length}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    minHeight: '100vh',
    padding: '28px 32px 32px',
    background: 'radial-gradient(circle at 50% -10%, #0b1222 0%, #020617 58%, #000 100%)',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
  },
  surface: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
    maxWidth: '1400px',
    margin: '0 auto',
    transition: 'all 0.3s ease',
  },
  layer: {
    animation: 'fadeIn 0.5s ease',
  },
  footer: {
    display: 'flex',
    justifyContent: 'center',
    paddingTop: '20px',
    marginTop: '10px',
    borderTop: '1px solid rgba(148, 163, 184, 0.08)',
  },
  footerContent: {
    display: 'flex',
    gap: '20px',
    alignItems: 'center',
    color: 'rgba(148, 163, 184, 0.5)',
    fontSize: '11px',
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
  },
  timestamp: {
    fontVariantNumeric: 'tabular-nums',
  },
  scenarioInfo: {},
}

// Add animation keyframes via a style tag
if (typeof document !== 'undefined' && !document.getElementById('three-layer-dashboard-styles')) {
  const style = document.createElement('style')
  style.id = 'three-layer-dashboard-styles'
  style.textContent = `
    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
  `
  document.head.appendChild(style)
}
