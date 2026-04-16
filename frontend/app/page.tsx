'use client'

import { useEffect, useState, useRef } from 'react'
import { fetchFD004DemoInit } from '@/lib/api'
import { DemoFrame } from '@/lib/types'
import HeaderBar from '@/components/HeaderBar'
import TetrahedronPanel from '@/components/TetrahedronPanel'
import InsightPanels from '@/components/InsightPanels'
import ReplayChart from '@/components/ReplayChart'
import PlaybackControls from '@/components/PlaybackControls'

// Transform DemoFrame to the format the components expect
function transformFrame(demoFrame: DemoFrame): any {
  return {
    index: demoFrame.frame_index,
    total: demoFrame.frame_count,
    timestamp: demoFrame.timestamp,
    phase: demoFrame.current_phase,
    system_health: demoFrame.status.toLowerCase(),
    confidence: demoFrame.metrics.confidence,
    structural_drift_score: demoFrame.metrics.structural_drift,
    relational_stability_score: demoFrame.metrics.relational_stability,
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

export default function Home() {
  const [frames, setFrames] = useState<Frame[]>([])
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(true)
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
  const [loading, setLoading] = useState(true)
  const [isConnected, setIsConnected] = useState(false)
  const animationFrameRef = useRef<number | null>(null)
  const lastFrameTimeRef = useRef<number>(0)

  // Load demo data from the correct backend endpoint
  useEffect(() => {
    const loadDemoData = async () => {
      try {
        const demoData = await fetchFD004DemoInit('unit_001')
        if (demoData.frames && demoData.frames.length > 0) {
          const transformedFrames = demoData.frames.map(transformFrame)
          setFrames(transformedFrames)
          setCurrentFrameIndex(0)
          setIsConnected(true)
        }
        setLoading(false)
      } catch (error) {
        console.error('Failed to fetch demo data:', error)
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

  const currentFrame = frames[currentFrameIndex] || null

  return (
    <div className="demo-app">
      {loading ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
          Loading FD004 engine data...
        </div>
      ) : !isConnected ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
          Failed to load demo data. Please refresh.
        </div>
      ) : frames.length === 0 ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
          No frames available.
        </div>
      ) : currentFrame ? (
        <>
          <HeaderBar frame={currentFrame} />
          <PlaybackControls
            currentIndex={currentFrameIndex}
            totalFrames={frames.length}
            isPlaying={isPlaying}
            playbackSpeed={playbackSpeed}
            onIndexChange={handleIndexChange}
            onPlayPause={handlePlayPause}
            onSpeedChange={handleSpeedChange}
            onRestart={handleRestart}
          />
          <div className="demo-main">
            <ReplayChart frames={frames} currentIndex={currentFrameIndex} />
            <TetrahedronPanel frame={currentFrame} />
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
