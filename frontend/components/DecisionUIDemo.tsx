'use client'

import { useState, useEffect } from 'react'
import DecisionUI from '@/components/DecisionUI'
import { DEMO_SCENARIOS } from '@/lib/demoScenarios'

export default function DecisionUIDemo() {
  const [scenarioIndex, setScenarioIndex] = useState(0)
  const [isAutoPlay, setIsAutoPlay] = useState(false)

  const scenario = DEMO_SCENARIOS[scenarioIndex]

  useEffect(() => {
    if (!isAutoPlay) return

    const timer = setTimeout(() => {
      setScenarioIndex((prev) => (prev + 1) % DEMO_SCENARIOS.length)
    }, 4000)

    return () => clearTimeout(timer)
  }, [isAutoPlay, scenarioIndex])

  return (
    <div style={styles.container}>
      <DecisionUI state={scenario.state} />

      {/* Demo controls */}
      <div style={styles.controls}>
        <div style={styles.controlsContent}>
          <div style={styles.scenarioIndicator}>
            <span style={styles.label}>Scenario:</span>
            <span style={styles.scenarioName}>{scenario.name}</span>
            <span style={styles.progress}>
              {scenarioIndex + 1} / {DEMO_SCENARIOS.length}
            </span>
          </div>

          <div style={styles.buttonGroup}>
            {scenarioIndex > 0 && (
              <button
                onClick={() => setScenarioIndex(scenarioIndex - 1)}
                style={{ ...styles.button, ...styles.buttonSecondary }}
              >
                ← Previous
              </button>
            )}

            {isAutoPlay ? (
              <button
                onClick={() => setIsAutoPlay(false)}
                style={{ ...styles.button, ...styles.buttonPrimary }}
              >
                ⏸ Pause
              </button>
            ) : (
              <button
                onClick={() => setIsAutoPlay(true)}
                style={{ ...styles.button, ...styles.buttonPrimary }}
              >
                ▶ Play
              </button>
            )}

            {scenarioIndex < DEMO_SCENARIOS.length - 1 && (
              <button
                onClick={() => setScenarioIndex(scenarioIndex + 1)}
                style={{ ...styles.button, ...styles.buttonSecondary }}
              >
                Next →
              </button>
            )}
          </div>

          <div style={styles.hint}>Click Play to auto-advance through scenarios</div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  container: {
    position: 'relative' as const,
  },

  controls: {
    position: 'fixed' as const,
    bottom: '0',
    left: '0',
    right: '0',
    padding: '16px',
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(8px)',
    zIndex: 100,
  },

  controlsContent: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '20px',
    maxWidth: '1400px',
    margin: '0 auto',
  },

  scenarioIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    fontSize: '12px',
  },

  label: {
    color: 'rgba(255, 255, 255, 0.4)',
    fontWeight: '600',
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
  },

  scenarioName: {
    color: '#0284c7',
    fontWeight: '600',
  },

  progress: {
    color: 'rgba(255, 255, 255, 0.3)',
    fontSize: '11px',
  },

  buttonGroup: {
    display: 'flex',
    gap: '8px',
  },

  button: {
    padding: '8px 16px',
    borderRadius: '6px',
    border: 'none',
    fontSize: '12px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },

  buttonPrimary: {
    backgroundColor: '#0284c7',
    color: '#ffffff',
    boxShadow: '0 0 12px rgba(2, 132, 199, 0.3)',
  },

  buttonPrimaryHover: {
    backgroundColor: '#0369a1',
    boxShadow: '0 0 16px rgba(2, 132, 199, 0.4)',
  },

  buttonSecondary: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    color: 'rgba(255, 255, 255, 0.8)',
  },

  buttonSecondaryHover: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
  },

  hint: {
    fontSize: '11px',
    color: 'rgba(255, 255, 255, 0.3)',
    fontStyle: 'italic',
    whiteSpace: 'nowrap',
  },
}
