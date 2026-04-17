'use client'

import { useEffect, useState, useRef } from 'react'
import { DemoFrame } from '@/lib/types'
import { parseCSV, csvRowsToDemoFrames } from '@/lib/csvParser'
import HeaderBar from '@/components/HeaderBar'
import TetrahedronPanel from '@/components/TetrahedronPanel'
import InsightPanels from '@/components/InsightPanels'
import ReplayChart from '@/components/ReplayChart'
import DriftAlert from '@/components/DriftAlert'
import PlaybackControls from '@/components/PlaybackControls'

function getSystemHealth(structuralDrift: number, relationalStability: number): string {
  const drift = Math.max(0, Math.min(1, structuralDrift))
  const stability = Math.max(0, Math.min(1, relationalStability))
  const instability = 1 - stability

  if (drift >= 0.78 || instability >= 0.55) return 'critical'
  if (drift >= 0.55 || instability >= 0.35) return 'warning'
  return 'nominal'
}

// Transform DemoFrame to the format the components expect
function transformFrame(demoFrame: DemoFrame): Frame {
  const structuralDrift = demoFrame.metrics.structural_drift
  const relationalStability = demoFrame.metrics.relational_stability
  const derivedHealth = getSystemHealth(structuralDrift, relationalStability)

  return {
    index: demoFrame.frame_index,
    total: demoFrame.frame_count,
    timestamp: demoFrame.timestamp,
    phase: demoFrame.current_phase,
    system_health: derivedHealth,
    confidence: demoFrame.metrics.confidence,
    structural_drift_score: structuralDrift,
    relational_stability_score: relationalStability,
    coherence_score: demoFrame.metrics.coherence,
    event_admitted: demoFrame.verdict === 'ADMITTED' || demoFrame.verdict === 'admitted',
    transition_type: demoFrame.verdict,
    persistence_minutes: demoFrame.frame_index,
  }
}

interface Frame {
  index: number
  total: number
  timestamp: string
  phase: string
  system_health: string
  confidence: number
  structural_drift_score: number
  relational_stability_score: number
  coherence_score: number
  event_admitted: boolean
  transition_type: string
  persistence_minutes: number
  [key: string]: any
}

const DEFAULT_FRAME_INTERVAL = 500 // milliseconds
const TRAILING_WINDOW = 24

function formatPhase(phase: string): string {
  return phase.replace(/_/g, ' ').toUpperCase()
}

export default function Home() {
  const [frames, setFrames] = useState<Frame[]>([])
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(true)
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
  const [loading, setLoading] = useState(true)
  const [isConnected, setIsConnected] = useState(false)
  const animationFrameRef = useRef<number | null>(null)
  const lastFrameTimeRef = useRef<number>(0)

  // Load demo data from CSV file
  useEffect(() => {
    const loadDemoData = async () => {
      try {
        const response = await fetch('/fd004_ingest_ready_part_1.csv')
        if (!response.ok) throw new Error('Failed to load CSV')

        const csvContent = await response.text()
        const csvRows = parseCSV(csvContent)

        if (!csvRows || csvRows.length === 0) {
          throw new Error('No CSV rows parsed')
        }

        const demoFrames = csvRowsToDemoFrames(csvRows)

        if (!demoFrames || demoFrames.length === 0) {
          throw new Error('No demo frames generated from CSV rows')
        }

        const transformedFrames = demoFrames.map(transformFrame)
        setFrames(transformedFrames)
        setCurrentFrameIndex(0)
        setIsConnected(true)
        setLoading(false)
      } catch (error) {
        console.error('Error loading CSV data:', error)
        setIsConnected(false)
        setLoading(false)
      }
    }

    loadDemoData()
  }, [])

  // Animation loop for playback
  useEffect(() => {
    if (!isPlaying || frames.length === 0) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
      return
    }

    const animate = (currentTime: number) => {
      if (lastFrameTimeRef.current === 0) {
        lastFrameTimeRef.current = currentTime
      }

      const elapsed = currentTime - lastFrameTimeRef.current
      const adjustedInterval = DEFAULT_FRAME_INTERVAL / playbackSpeed

      if (elapsed >= adjustedInterval) {
        setCurrentFrameIndex((prevIndex) => {
          const nextIndex = prevIndex + 1
          if (nextIndex >= frames.length) {
            setIsPlaying(false)
            return prevIndex
          }
          return nextIndex
        })
        lastFrameTimeRef.current = currentTime
      }

      animationFrameRef.current = requestAnimationFrame(animate)
    }

    animationFrameRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
    }
  }, [isPlaying, frames.length, playbackSpeed])

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleIndexChange = (index: number) => {
    setCurrentFrameIndex(index)
    lastFrameTimeRef.current = 0
  }

  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed)
    lastFrameTimeRef.current = 0
  }

  const handleRestart = () => {
    setCurrentFrameIndex(0)
    lastFrameTimeRef.current = 0
    setIsPlaying(true)
  }

  const handleStep = (direction: 1 | -1) => {
    setCurrentFrameIndex((prevIndex) => {
      const nextIndex = Math.max(0, Math.min(frames.length - 1, prevIndex + direction))
      return nextIndex
    })
    lastFrameTimeRef.current = 0
  }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!frames.length) return
      if (event.key === ' ') {
        event.preventDefault()
        setIsPlaying((prev) => !prev)
      } else if (event.key === 'ArrowRight') {
        event.preventDefault()
        handleStep(1)
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        handleStep(-1)
      } else if (event.key.toLowerCase() === 'r') {
        event.preventDefault()
        handleRestart()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [frames.length])

  const currentFrame = frames[currentFrameIndex] || null
  const playbackProgress = frames.length ? ((currentFrameIndex + 1) / frames.length) * 100 : 0

  const currentDrift = currentFrame ? currentFrame.structural_drift_score * 100 : 0
  const currentStability = currentFrame ? currentFrame.relational_stability_score * 100 : 0
  const currentCoherence = currentFrame ? currentFrame.coherence_score * 100 : 0
  const currentConfidence = currentFrame ? currentFrame.confidence * 100 : 0

  const trailingStart = Math.max(0, currentFrameIndex - TRAILING_WINDOW + 1)
  const trailingFrames = frames.slice(trailingStart, currentFrameIndex + 1)
  const trailingDriftAvg = trailingFrames.length
    ? trailingFrames.reduce((acc, frame) => acc + frame.structural_drift_score, 0) / trailingFrames.length
    : 0

  const phaseBoundaries = frames.reduce<Array<{ label: string; index: number; key: string }>>((acc, frame, index) => {
    const key = frame.phase || 'unknown'
    const previous = acc[acc.length - 1]
    if (!previous || previous.key !== key) {
      acc.push({ key, label: formatPhase(key), index })
    }
    return acc
  }, [])

  const activePhase = currentFrame ? formatPhase(currentFrame.phase || 'unknown') : 'UNKNOWN'
  const admittedRate = frames.length
    ? (frames.slice(0, currentFrameIndex + 1).filter((frame) => frame.event_admitted).length / (currentFrameIndex + 1)) * 100
    : 0

  return (
    <div className="demo-app">
      {loading ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
          Loading CSV demo data...
        </div>
      ) : !isConnected ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
          Failed to load CSV data. Please refresh.
        </div>
      ) : frames.length === 0 ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
          No frames available.
        </div>
      ) : currentFrame ? (
        <>
          <HeaderBar frame={currentFrame} />
          <section className="session-strip">
            <div className="session-strip-head">
              <div>
                <p className="session-title">Session Overview</p>
                <p className="session-subtitle">
                  Phase <strong>{activePhase}</strong> • frame {currentFrameIndex + 1}/{frames.length}
                </p>
              </div>
              <div className="session-hotkeys">
                <span>Space: Play/Pause</span>
                <span>←/→: Step frame</span>
                <span>R: Restart</span>
              </div>
            </div>
            <div className="session-progress-track">
              <div className="session-progress-fill" style={{ width: `${playbackProgress.toFixed(2)}%` }} />
            </div>
            <div className="session-grid">
              <article className="session-card">
                <p>Structural drift</p>
                <strong>{currentDrift.toFixed(1)}%</strong>
                <span>Trailing avg {(trailingDriftAvg * 100).toFixed(1)}%</span>
              </article>
              <article className="session-card">
                <p>Relational stability</p>
                <strong>{currentStability.toFixed(1)}%</strong>
                <span>Instability {(100 - currentStability).toFixed(1)}%</span>
              </article>
              <article className="session-card">
                <p>Coherence / confidence</p>
                <strong>{currentCoherence.toFixed(1)}% / {currentConfidence.toFixed(1)}%</strong>
                <span>Signal admitted {admittedRate.toFixed(1)}% of replay</span>
              </article>
            </div>
          </section>
          <PlaybackControls
            currentIndex={currentFrameIndex}
            totalFrames={frames.length}
            isPlaying={isPlaying}
            playbackSpeed={playbackSpeed}
            phaseMarkers={phaseBoundaries}
            onIndexChange={handleIndexChange}
            onPlayPause={handlePlayPause}
            onSpeedChange={handleSpeedChange}
            onRestart={handleRestart}
            onStep={handleStep}
          />
          <div className="demo-main">
            <DriftAlert frames={frames} currentIndex={currentFrameIndex} />
            <TetrahedronPanel frame={currentFrame} />
            <ReplayChart frames={frames} currentIndex={currentFrameIndex} />
            <InsightPanels frame={currentFrame} />
          </div>
        </>
      ) : (
        <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
          Loading system data...
        </div>
      )}
    </div>
  )
}
