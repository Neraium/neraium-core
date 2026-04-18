'use client'

import { useState, useEffect, useMemo } from 'react'
import DecisionUI from '@/components/DecisionUI'
import { DEMO_SCENARIOS } from '@/lib/demoScenarios'
import { DecisionUIState } from '@/lib/decisionToUI'

const INTERPOLATION_MS = 2500
const TEXT_LAG_MS = 260

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
}

function interpolateStates(from: DecisionUIState, to: DecisionUIState, t: number): DecisionUIState {
  const eased = easeInOut(Math.max(0, Math.min(1, t)))
  const trailLength = Math.min(from.tetrahedron.trailPoints.length, to.tetrahedron.trailPoints.length)

  const trailPoints = Array.from({ length: trailLength }, (_, i) => {
    const fromPoint = from.tetrahedron.trailPoints[i]
    const toPoint = to.tetrahedron.trailPoints[i]
    return {
      ...toPoint,
      position: {
        x: lerp(fromPoint.position.x, toPoint.position.x, eased),
        y: lerp(fromPoint.position.y, toPoint.position.y, eased),
        z: lerp(fromPoint.position.z, toPoint.position.z, eased),
      },
    }
  })

  const currentFrameIdx = Math.round(lerp(from.driftChart.currentFrameIdx, to.driftChart.currentFrameIdx, eased))

  return {
    ...to,
    actionPanel: {
      ...to.actionPanel,
      urgencyLevel: Math.round(lerp(from.actionPanel.urgencyLevel, to.actionPanel.urgencyLevel, eased)),
    },
    tetrahedron: {
      ...to.tetrahedron,
      severityScalar: lerp(from.tetrahedron.severityScalar, to.tetrahedron.severityScalar, eased),
      currentPosition: {
        x: lerp(from.tetrahedron.currentPosition.x, to.tetrahedron.currentPosition.x, eased),
        y: lerp(from.tetrahedron.currentPosition.y, to.tetrahedron.currentPosition.y, eased),
        z: lerp(from.tetrahedron.currentPosition.z, to.tetrahedron.currentPosition.z, eased),
      },
      velocity: [
        lerp(from.tetrahedron.velocity[0], to.tetrahedron.velocity[0], eased),
        lerp(from.tetrahedron.velocity[1], to.tetrahedron.velocity[1], eased),
        lerp(from.tetrahedron.velocity[2], to.tetrahedron.velocity[2], eased),
      ],
      trailPoints,
    },
    driftChart: {
      ...to.driftChart,
      currentFrameIdx,
    },
  }
}

export default function DecisionUIDemo() {
  const [scenarioIndex, setScenarioIndex] = useState(0)
  const [displayedScenarioIndex, setDisplayedScenarioIndex] = useState(0)
  const [isAutoPlay, setIsAutoPlay] = useState(true)
  const [fromScenarioIndex, setFromScenarioIndex] = useState(0)
  const [transitionStart, setTransitionStart] = useState<number | null>(null)
  const [progress, setProgress] = useState(1)

  const scenario = DEMO_SCENARIOS[scenarioIndex]
  const displayedScenario = DEMO_SCENARIOS[displayedScenarioIndex]

  useEffect(() => {
    if (!isAutoPlay) return

    const timer = setTimeout(() => {
      setFromScenarioIndex(scenarioIndex)
      setScenarioIndex((prev) => (prev + 1) % DEMO_SCENARIOS.length)
      setTransitionStart(performance.now())
      setProgress(0)
    }, scenario.durationMs)

    return () => clearTimeout(timer)
  }, [isAutoPlay, scenarioIndex, scenario.durationMs])

  useEffect(() => {
    const timer = setTimeout(() => setDisplayedScenarioIndex(scenarioIndex), TEXT_LAG_MS)
    return () => clearTimeout(timer)
  }, [scenarioIndex])

  useEffect(() => {
    if (transitionStart === null || progress >= 1) return

    let raf = 0
    const tick = () => {
      const elapsed = performance.now() - transitionStart
      const nextProgress = Math.min(1, elapsed / INTERPOLATION_MS)
      setProgress(nextProgress)
      if (nextProgress < 1) raf = requestAnimationFrame(tick)
    }

    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [transitionStart, progress])

  const uiState = useMemo(() => {
    if (progress >= 1 || fromScenarioIndex === scenarioIndex) return scenario.state
    return interpolateStates(DEMO_SCENARIOS[fromScenarioIndex].state, scenario.state, progress)
  }, [scenario, scenarioIndex, fromScenarioIndex, progress])

  const meaningLine = displayedScenario.name === 'Emerging' ? 'Detected before failure conditions' : null

  return (
    <div style={styles.container}>
      <DecisionUI state={uiState} narrative={displayedScenario.narrative} meaningLine={meaningLine} />

      <div style={styles.controls}>
        <div style={styles.controlsContent}>
          <div style={styles.status}>
            <span style={styles.label}>{displayedScenario.name}</span>
            <span style={styles.progress}>{displayedScenarioIndex + 1}/{DEMO_SCENARIOS.length}</span>
          </div>

          <div style={styles.controls2}>
            {scenarioIndex > 0 && (
              <button
                onClick={() => {
                  setFromScenarioIndex(scenarioIndex)
                  setScenarioIndex(scenarioIndex - 1)
                  setTransitionStart(performance.now())
                  setProgress(0)
                }}
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
                onClick={() => {
                  setFromScenarioIndex(scenarioIndex)
                  setScenarioIndex(scenarioIndex + 1)
                  setTransitionStart(performance.now())
                  setProgress(0)
                }}
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
  container: { position: 'relative' as const },
  controls: {
    position: 'fixed' as const,
    bottom: '30px',
    left: '32px',
    zIndex: 100,
  },
  controlsContent: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-start',
    gap: '14px',
  },
  status: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '12px',
  },
  label: {
    color: 'rgba(255, 255, 255, 0.56)',
    fontWeight: '600',
    letterSpacing: '0.07em',
  },
  progress: {
    color: 'rgba(255, 255, 255, 0.34)',
    fontSize: '11px',
    fontVariantNumeric: 'tabular-nums',
  },
  controls2: {
    display: 'flex',
    gap: '4px',
    padding: '8px 12px',
    borderRadius: '8px',
    backgroundColor: 'rgba(7, 15, 30, 0.56)',
    border: '1px solid rgba(148, 163, 184, 0.2)',
    backdropFilter: 'blur(8px)',
  },
  button: {
    width: '28px',
    height: '28px',
    padding: '0',
    borderRadius: '4px',
    border: 'none',
    backgroundColor: 'transparent',
    color: 'rgba(255, 255, 255, 0.68)',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.46s ease',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
}
