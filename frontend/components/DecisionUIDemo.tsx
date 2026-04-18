'use client'

import { useState, useEffect } from 'react'
import DecisionUI from '@/components/DecisionUI'
import { DEMO_SCENARIOS } from '@/lib/demoScenarios'

export default function DecisionUIDemo() {
  const [scenarioIndex, setScenarioIndex] = useState(0)
  const [isAutoPlay, setIsAutoPlay] = useState(true)

  const scenario = DEMO_SCENARIOS[scenarioIndex]

  useEffect(() => {
    if (!isAutoPlay) return

    const timer = setTimeout(() => {
      setScenarioIndex((prev) => (prev + 1) % DEMO_SCENARIOS.length)
    }, 5000)

    return () => clearTimeout(timer)
  }, [isAutoPlay, scenarioIndex])

  return (
    <div style={styles.container}>
      <DecisionUI state={scenario.state} />

      {/* Minimal demo controls */}
      <div style={styles.controls}>
        <div style={styles.controlsContent}>
          <div style={styles.status}>
            <span style={styles.label}>{scenario.name}</span>
            <span style={styles.progress}>{scenarioIndex + 1}/{DEMO_SCENARIOS.length}</span>
          </div>

          <div style={styles.controls2}>
            {scenarioIndex > 0 && (
              <button
                onClick={() => setScenarioIndex(scenarioIndex - 1)}
                style={styles.button}
                title="Previous scenario"
              >
                ‹
              </button>
            )}

            <button
              onClick={() => setIsAutoPlay(!isAutoPlay)}
              style={styles.button}
              title={isAutoPlay ? 'Pause' : 'Play'}
            >
              {isAutoPlay ? '❚❚' : '▶'}
            </button>

            {scenarioIndex < DEMO_SCENARIOS.length - 1 && (
              <button
                onClick={() => setScenarioIndex(scenarioIndex + 1)}
                style={styles.button}
                title="Next scenario"
              >
                ›
              </button>
            )}
          </div>
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
    bottom: '32px',
    left: '32px',
    padding: '0',
    backgroundColor: 'transparent',
    borderTop: 'none',
    backdropFilter: 'none',
    zIndex: 100,
  },

  controlsContent: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-start',
    gap: '16px',
  },

  status: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '12px',
  },

  label: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontWeight: '600',
    letterSpacing: '0.05em',
  },

  progress: {
    color: 'rgba(255, 255, 255, 0.3)',
    fontSize: '11px',
    fontVariantNumeric: 'tabular-nums',
  },

  controls2: {
    display: 'flex',
    gap: '4px',
    padding: '8px 12px',
    borderRadius: '6px',
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    backdropFilter: 'blur(4px)',
  },

  button: {
    width: '28px',
    height: '28px',
    padding: '0',
    borderRadius: '4px',
    border: 'none',
    backgroundColor: 'transparent',
    color: 'rgba(255, 255, 255, 0.6)',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
}
