'use client'

import { useState, useEffect, useMemo } from 'react'
import SystemOverview from '@/components/SystemOverview'
import SystemStateTrajectory from '@/components/SystemStateTrajectory'
import ActionRecommendations from '@/components/ActionRecommendations'
import { COOLING_DEMO_SCENARIOS } from '@/lib/coolingSystemDemo'
import { DecisionUIState } from '@/lib/decisionToUI'

export default function CoolingSystemDemo() {
  const [scenarioIndex, setScenarioIndex] = useState(0)
  const [isAutoPlay, setIsAutoPlay] = useState(true)
  const [isPaused, setIsPaused] = useState(false)
  const [displayedScenarioIndex, setDisplayedScenarioIndex] = useState(0)
  const [transitionStart, setTransitionStart] = useState<number | null>(null)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [speed, setSpeed] = useState(1)

  const scenario = COOLING_DEMO_SCENARIOS[scenarioIndex]
  const displayedScenario = COOLING_DEMO_SCENARIOS[displayedScenarioIndex]
  const state = displayedScenario.state

  // Track elapsed time for progress indication
  useEffect(() => {
    const interval = setInterval(() => {
      if (!isPaused) {
        setElapsedMs(prev => prev + (100 * speed))
      }
    }, 100)
    return () => clearInterval(interval)
  }, [isPaused, speed])

  // Auto-advance through scenarios
  useEffect(() => {
    if (!isAutoPlay || isPaused) return
    if (scenarioIndex >= COOLING_DEMO_SCENARIOS.length - 1) {
      setIsAutoPlay(false)
      return
    }
    const adjustedDuration = scenario.durationMs / speed
    const timer = setTimeout(() => {
      setDisplayedScenarioIndex(scenarioIndex)
      setTransitionStart(Date.now())
      setScenarioIndex(i => i + 1)
      setElapsedMs(0)
    }, adjustedDuration)
    return () => clearTimeout(timer)
  }, [scenarioIndex, isAutoPlay, isPaused, scenario.durationMs, speed])

  // Datacenter metrics showing cooling system health across fleet
  const systemHealthMetrics = useMemo(() => ({
    healthy: Math.max(1, 14 - scenarioIndex * 3),
    drifting: Math.min(5, scenarioIndex * 1.2),
    unstable: Math.max(0, Math.floor(scenarioIndex * 0.7)),
    earlyWarnings: Math.max(2, 8 - scenarioIndex * 1.5),
    enteringInstability: Math.max(0, scenarioIndex - 0.5),
  }), [scenarioIndex])

  // Calculate progress through full demo
  const totalDuration = COOLING_DEMO_SCENARIOS.reduce((sum, s) => sum + s.durationMs / speed, 0)
  const progressPercent = ((scenarioIndex * (scenario.durationMs / speed) + elapsedMs) / totalDuration) * 100

  return (
    <div style={styles.root}>
      <div style={styles.surface}>
        {/* Header */}
        <div style={styles.headerSection}>
          <div>
            <h1 style={styles.title}>Datacenter Cooling System Intelligence</h1>
            <p style={styles.subtitle}>Real-time CRAC unit degradation tracking and predictive maintenance</p>
          </div>
          <div style={styles.controls}>
            <div style={styles.controlGroup}>
              <button
                style={{...styles.button, ...(!isPaused ? styles.buttonActive : {})}}
                onClick={() => { setIsPaused(false); setIsAutoPlay(true) }}
              >
                ▶ Play
              </button>
              <button
                style={{...styles.button, ...(isPaused ? styles.buttonActive : {})}}
                onClick={() => setIsPaused(!isPaused)}
              >
                ⏸ Pause
              </button>
              <button
                style={styles.button}
                onClick={() => {
                  setScenarioIndex(Math.max(0, scenarioIndex - 1))
                  setElapsedMs(0)
                  setIsAutoPlay(false)
                }}
                disabled={scenarioIndex === 0}
              >
                ← Prev
              </button>
              <button
                style={styles.button}
                onClick={() => {
                  setScenarioIndex(Math.min(COOLING_DEMO_SCENARIOS.length - 1, scenarioIndex + 1))
                  setElapsedMs(0)
                  setIsAutoPlay(false)
                }}
                disabled={scenarioIndex === COOLING_DEMO_SCENARIOS.length - 1}
              >
                Next →
              </button>
            </div>
            <div style={styles.speedControl}>
              <label style={styles.speedLabel}>Speed:</label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.25"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                style={styles.speedSlider}
              />
              <span style={styles.speedValue}>{speed.toFixed(2)}x</span>
            </div>
          </div>
          <div style={styles.demoStatus}>
            <div style={styles.statusLabel}>Demo: {scenarioIndex + 1} of {COOLING_DEMO_SCENARIOS.length}</div>
            <div style={styles.scenarioName}>{scenario.name}</div>
          </div>
        </div>

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

        {/* Footer with scenario info */}
        <div style={styles.footer}>
          <div style={styles.footerContent}>
            <div style={styles.narrativeBox}>
              <div style={styles.narrativeLabel}>Scenario Narrative</div>
              <div style={styles.narrativeText}>{scenario.narrative}</div>
            </div>
            <div style={styles.timestamp}>
              {new Date().toLocaleString(undefined, {
                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
              })}
            </div>
          </div>
        </div>

        {/* Progress bar and demo timer */}
        <div style={styles.progressSection}>
          <div style={styles.progressBarContainer}>
            <div
              style={{
                ...styles.progressBar,
                width: `${Math.min(100, progressPercent)}%`,
              }}
            />
          </div>
          {(isAutoPlay || isPaused) && (
            <div style={styles.autoplayIndicator}>
              <div style={{...styles.autoplayDot, ...(isPaused ? {animation: 'none', opacity: 0.5} : {})}} />
              <span>Stage {scenarioIndex + 1} of {COOLING_DEMO_SCENARIOS.length}</span>
              <span style={styles.separator}>•</span>
              {isPaused ? (
                <span>Paused</span>
              ) : (
                <span>Next in {Math.ceil(((scenario.durationMs / speed) - elapsedMs % (scenario.durationMs / speed)) / 1000)}s</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    minHeight: '100vh',
    padding: 'clamp(16px, 5vw, 32px)',
    background: 'radial-gradient(circle at 50% -10%, #0b1222 0%, #020617 58%, #000 100%)',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
  },
  surface: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
    maxWidth: '1400px',
    margin: '0 auto',
  },
  headerSection: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingBottom: '20px',
    borderBottom: '1px solid rgba(148, 163, 184, 0.12)',
    gap: '20px',
    flexWrap: 'wrap',
  },
  title: {
    fontSize: 'clamp(20px, 5vw, 24px)',
    fontWeight: '700',
    color: '#f1f5f9',
    margin: '0 0 8px 0',
    letterSpacing: '-0.01em',
  },
  subtitle: {
    fontSize: '13px',
    color: 'rgba(148, 163, 184, 0.65)',
    margin: 0,
    fontWeight: '400',
  },
  demoStatus: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    textAlign: 'right',
  },
  statusLabel: {
    fontSize: '11px',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    color: 'rgba(148, 163, 184, 0.5)',
    fontWeight: '600',
  },
  scenarioName: {
    fontSize: '14px',
    color: '#60a5fa',
    fontWeight: '600',
  },
  layer: {
    animation: 'fadeIn 0.5s ease',
  },
  footer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: '20px',
    marginTop: '10px',
    borderTop: '1px solid rgba(148, 163, 184, 0.08)',
  },
  footerContent: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
    gap: '20px',
  },
  narrativeBox: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    flex: 1,
  },
  narrativeLabel: {
    fontSize: '10px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'rgba(148, 163, 184, 0.5)',
    fontWeight: '600',
  },
  narrativeText: {
    fontSize: '12px',
    color: '#cbd5e1',
    fontStyle: 'italic',
    lineHeight: 1.4,
  },
  timestamp: {
    fontSize: '11px',
    color: 'rgba(148, 163, 184, 0.5)',
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap',
  },
  progressSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    paddingTop: '12px',
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
    background: 'linear-gradient(90deg, #60a5fa, #3b82f6)',
    transition: 'width 0.1s linear',
  },
  autoplayIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 12px',
    backgroundColor: 'rgba(15, 23, 42, 0.6)',
    borderRadius: '6px',
    border: '1px solid rgba(148, 163, 184, 0.12)',
    fontSize: '11px',
    color: 'rgba(148, 163, 184, 0.7)',
  },
  autoplayDot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    backgroundColor: '#60a5fa',
    animation: 'pulse 1.5s ease-in-out infinite',
    flexShrink: 0,
  },
  separator: {
    color: 'rgba(148, 163, 184, 0.3)',
  },
  controls: {
    display: 'flex',
    alignItems: 'center',
    gap: '20px',
    flexWrap: 'wrap',
    flex: 1,
  },
  controlGroup: {
    display: 'flex',
    gap: '8px',
  },
  button: {
    padding: '6px 12px',
    fontSize: '12px',
    fontWeight: '500',
    border: '1px solid rgba(96, 165, 250, 0.3)',
    borderRadius: '6px',
    backgroundColor: 'rgba(15, 23, 42, 0.4)',
    color: '#cbd5e1',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    whiteSpace: 'nowrap',
  } as React.CSSProperties,
  buttonActive: {
    backgroundColor: 'rgba(96, 165, 250, 0.2)',
    borderColor: '#60a5fa',
    color: '#60a5fa',
  },
  speedControl: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  speedLabel: {
    fontSize: '12px',
    color: 'rgba(148, 163, 184, 0.7)',
    fontWeight: '500',
  },
  speedSlider: {
    width: '80px',
    height: '4px',
    cursor: 'pointer',
  } as React.CSSProperties,
  speedValue: {
    fontSize: '12px',
    color: '#60a5fa',
    fontWeight: '600',
    minWidth: '40px',
  },
}

// Add animation keyframes and responsive styles
if (typeof document !== 'undefined') {
  const style = document.createElement('style')
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
    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.4;
      }
    }
    @media (max-width: 768px) {
      button {
        padding: 4px 8px !important;
        font-size: 11px !important;
      }
    }
    @media (max-width: 640px) {
      [style*="gridTemplateColumns"] {
        grid-template-columns: 1fr !important;
      }
    }
  `
  document.head.appendChild(style)
}
