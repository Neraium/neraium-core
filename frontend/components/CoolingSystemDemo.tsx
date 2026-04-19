'use client'

import { useState, useEffect, useMemo } from 'react'
import SystemOverview from '@/components/SystemOverview'
import SystemStateTrajectory from '@/components/SystemStateTrajectory'
import ActionRecommendations from '@/components/ActionRecommendations'
import EnhancedTetrahedronViz from '@/components/EnhancedTetrahedronViz'
import { COOLING_DEMO_SCENARIOS } from '@/lib/coolingSystemDemo'
import { DecisionUIState } from '@/lib/decisionToUI'

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

function interpolateState(from: DecisionUIState, to: DecisionUIState, progress: number): DecisionUIState {
  const t = easeInOutCubic(Math.min(1, progress))

  return {
    ...to,
    statusHeader: {
      ...to.statusHeader,
    },
    actionPanel: {
      ...to.actionPanel,
      urgencyLevel: Math.round(lerp(from.actionPanel.urgencyLevel, to.actionPanel.urgencyLevel, t)),
    },
    tetrahedron: {
      ...to.tetrahedron,
      severityScalar: lerp(from.tetrahedron.severityScalar, to.tetrahedron.severityScalar, t),
      currentPosition: {
        x: lerp(from.tetrahedron.currentPosition.x, to.tetrahedron.currentPosition.x, t),
        y: lerp(from.tetrahedron.currentPosition.y, to.tetrahedron.currentPosition.y, t),
        z: lerp(from.tetrahedron.currentPosition.z, to.tetrahedron.currentPosition.z, t),
      },
    },
    driftChart: {
      ...to.driftChart,
    },
  }
}

export default function CoolingSystemDemo() {
  const [scenarioIndex, setScenarioIndex] = useState(0)
  const [isAutoPlay, setIsAutoPlay] = useState(true)
  const [startTime, setStartTime] = useState<number | null>(null)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [isHydrated, setIsHydrated] = useState(false)

  const scenario = COOLING_DEMO_SCENARIOS[scenarioIndex]
  const previousScenario = scenarioIndex > 0 ? COOLING_DEMO_SCENARIOS[scenarioIndex - 1] : scenario
  const state = scenarioIndex > 0 && elapsedMs < scenario.durationMs
    ? interpolateState(previousScenario.state, scenario.state, elapsedMs / scenario.durationMs)
    : scenario.state

  // Initialize after hydration
  useEffect(() => {
    setIsHydrated(true)
    if (startTime === null) {
      setStartTime(Date.now())
    }
  }, [])

  // Update elapsed time smoothly
  useEffect(() => {
    if (!startTime) return
    const animationFrame = setInterval(() => {
      setElapsedMs(Date.now() - startTime)
    }, 30)
    return () => clearInterval(animationFrame)
  }, [startTime])

  // Auto-advance through scenarios
  useEffect(() => {
    if (!isAutoPlay) return
    if (scenarioIndex >= COOLING_DEMO_SCENARIOS.length - 1 && elapsedMs >= scenario.durationMs) {
      setIsAutoPlay(false)
      return
    }

    if (elapsedMs >= scenario.durationMs) {
      setScenarioIndex(i => Math.min(i + 1, COOLING_DEMO_SCENARIOS.length - 1))
      setStartTime(Date.now())
    }
  }, [elapsedMs, scenarioIndex, isAutoPlay, scenario.durationMs])

  // Fleet metrics degrading smoothly
  const systemHealthMetrics = useMemo(() => {
    const baseProgress = (scenarioIndex + Math.min(1, elapsedMs / scenario.durationMs)) / COOLING_DEMO_SCENARIOS.length
    return {
      healthy: Math.round(14 - baseProgress * 10),
      drifting: Math.round(1 + baseProgress * 4),
      unstable: Math.round(baseProgress * 3),
      earlyWarnings: Math.round(8 - baseProgress * 6),
      enteringInstability: Math.round(baseProgress * 3),
    }
  }, [scenarioIndex, elapsedMs, scenario.durationMs])

  const totalDuration = COOLING_DEMO_SCENARIOS.reduce((sum, s) => sum + s.durationMs, 0)
  const totalElapsed = startTime ? (scenarioIndex * 30000 + elapsedMs) : 0
  const progressPercent = (totalElapsed / totalDuration) * 100
  const secondsRemaining = Math.ceil((totalDuration - totalElapsed) / 1000)

  if (!isHydrated || !startTime) {
    return (
      <div style={styles.root}>
        <div style={styles.surface}>
          <div style={styles.headerSection}>
            <div>
              <h1 style={styles.title}>Datacenter CRAC Unit Degradation</h1>
              <p style={styles.subtitle}>Loading...</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={styles.root}>
      <div style={styles.surface}>
        {/* Header */}
        <div style={styles.headerSection}>
          <div>
            <h1 style={styles.title}>Datacenter CRAC Unit Degradation</h1>
            <p style={styles.subtitle}>Real-time system intelligence with predictive trajectory analysis</p>
          </div>
          <div style={styles.demoStatus}>
            <div style={styles.statusLabel}>Stage {scenarioIndex + 1} of {COOLING_DEMO_SCENARIOS.length}</div>
            <div style={styles.scenarioName}>{scenario.name}</div>
            <div style={styles.timerText}>{secondsRemaining}s remaining</div>
          </div>
        </div>

        {/* Layer 1: System Overview */}
        <div style={styles.layerWrapper}>
          <SystemOverview metrics={systemHealthMetrics} />
        </div>

        {/* Layer 2 + 3: Split view with tetrahedron */}
        <div style={styles.mainContent}>
          {/* Left: State & Trajectory + Actions */}
          <div style={styles.leftPanel}>
            <div style={styles.layerWrapper}>
              <SystemStateTrajectory state={state} systemName={scenario.name} />
            </div>
            <div style={styles.layerWrapper}>
              <ActionRecommendations state={state} />
            </div>
          </div>

          {/* Right: Tetrahedron Visualization */}
          <div style={styles.rightPanel}>
            <div style={styles.tetrahedronContainer}>
              <div style={styles.tetrahedronTitle}>System State Space</div>
              <EnhancedTetrahedronViz tetrahedronState={state.tetrahedron} isInteractive={false} />
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div style={styles.progressSection}>
          <div style={styles.progressBarContainer}>
            <div
              style={{
                ...styles.progressBar,
                width: `${Math.min(100, progressPercent)}%`,
              }}
            />
          </div>

          {/* Footer */}
          <div style={styles.footer}>
            <div style={styles.narrativeText}>{scenario.narrative}</div>
            <div style={styles.footerRight}>
              <span style={styles.timestamp}>
                {Math.floor(totalElapsed / 1000)}s / {Math.floor(totalDuration / 1000)}s
              </span>
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
    padding: '24px 32px 32px',
    background: 'radial-gradient(circle at 50% -10%, #0b1222 0%, #020617 58%, #000 100%)',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
  },
  surface: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    maxWidth: '1600px',
    margin: '0 auto',
  },
  headerSection: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingBottom: '16px',
    borderBottom: '1px solid rgba(148, 163, 184, 0.1)',
  },
  title: {
    fontSize: '22px',
    fontWeight: '700',
    color: '#f1f5f9',
    margin: '0 0 6px 0',
    letterSpacing: '-0.01em',
  },
  subtitle: {
    fontSize: '12px',
    color: 'rgba(148, 163, 184, 0.6)',
    margin: 0,
    fontWeight: '400',
  },
  demoStatus: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    textAlign: 'right',
  },
  statusLabel: {
    fontSize: '10px',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    color: 'rgba(148, 163, 184, 0.5)',
    fontWeight: '600',
  },
  scenarioName: {
    fontSize: '13px',
    color: '#60a5fa',
    fontWeight: '600',
  },
  timerText: {
    fontSize: '11px',
    color: 'rgba(148, 163, 184, 0.5)',
    fontWeight: '500',
  },
  layerWrapper: {
    animation: 'fadeIn 0.6s ease',
  },
  mainContent: {
    display: 'grid',
    gridTemplateColumns: '1fr 420px',
    gap: '16px',
    alignItems: 'start',
  },
  leftPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  rightPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  tetrahedronContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '16px',
    backgroundColor: 'rgba(15, 23, 42, 0.3)',
    borderRadius: '12px',
    border: '1px solid rgba(148, 163, 184, 0.08)',
    minHeight: '520px',
  },
  tetrahedronTitle: {
    fontSize: '11px',
    fontWeight: '600',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: 'rgba(148, 163, 184, 0.6)',
  },
  progressSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    paddingTop: '8px',
    marginTop: '8px',
    borderTop: '1px solid rgba(148, 163, 184, 0.08)',
  },
  progressBarContainer: {
    width: '100%',
    height: '3px',
    backgroundColor: 'rgba(148, 163, 184, 0.1)',
    borderRadius: '2px',
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    background: 'linear-gradient(90deg, #3b82f6 0%, #60a5fa 50%, #06b6d4 100%)',
    transition: 'width 0.03s linear',
  },
  footer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '16px',
  },
  narrativeText: {
    fontSize: '12px',
    color: '#cbd5e1',
    fontStyle: 'italic',
    lineHeight: 1.4,
    flex: 1,
  },
  footerRight: {
    display: 'flex',
    gap: '16px',
    alignItems: 'center',
    whiteSpace: 'nowrap',
  },
  timestamp: {
    fontSize: '11px',
    color: 'rgba(148, 163, 184, 0.5)',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: '500',
  },
}

// Add animation keyframes
if (typeof document !== 'undefined') {
  const style = document.createElement('style')
  style.textContent = `
    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(8px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
  `
  document.head.appendChild(style)
}
