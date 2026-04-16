'use client'

import { useEffect, useState, useRef } from 'react'
import axios from 'axios'
import HeaderBar from '@/components/HeaderBar'
import PlaybackControls from '@/components/PlaybackControls'
import ReplayChart from '@/components/ReplayChart'
import TetrahedronPanel from '@/components/TetrahedronPanel'
import InsightPanels from '@/components/InsightPanels'

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

export default function Home() {
  const [frames, setFrames] = useState<Frame[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
  const [loading, setLoading] = useState(true)
  const playbackIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  // Fetch frames on mount
  useEffect(() => {
    const fetchFrames = async () => {
      try {
        const response = await axios.get(`${apiBase}/api/ui/frames`)
        setFrames(response.data.frames || [])
        setCurrentIndex(0)
      } catch (error) {
        console.error('Failed to fetch frames:', error)
        // Fallback: create synthetic frames
        const syntheticFrames = generateSyntheticFrames()
        setFrames(syntheticFrames)
      } finally {
        setLoading(false)
      }
    }
    fetchFrames()
  }, [apiBase])

  // Handle playback
  useEffect(() => {
    if (!isPlaying || frames.length === 0) {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current)
        playbackIntervalRef.current = null
      }
      return
    }

    const delay = Math.max(100, Math.floor(1000 / playbackSpeed))
    playbackIntervalRef.current = setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev >= frames.length - 1) {
          setIsPlaying(false)
          return 0
        }
        return prev + 1
      })
    }, delay)

    return () => {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current)
      }
    }
  }, [isPlaying, playbackSpeed, frames.length])

  const currentFrame = frames[currentIndex] || {}

  return (
    <div className="demo-app">
      <HeaderBar frame={currentFrame} />
      <PlaybackControls
        currentIndex={currentIndex}
        totalFrames={frames.length}
        isPlaying={isPlaying}
        playbackSpeed={playbackSpeed}
        onIndexChange={setCurrentIndex}
        onPlayPause={() => setIsPlaying(!isPlaying)}
        onSpeedChange={setPlaybackSpeed}
        onRestart={() => {
          setCurrentIndex(0)
          setIsPlaying(false)
        }}
      />
      <div className="demo-main">
        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
            Loading demo...
          </div>
        ) : frames.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
            No frames available
          </div>
        ) : (
          <>
            <ReplayChart frames={frames} currentIndex={currentIndex} />
            <TetrahedronPanel frame={currentFrame} />
            <InsightPanels frame={currentFrame} />
          </>
        )}
      </div>
    </div>
  )
}

function generateSyntheticFrames(): Frame[] {
  const frames: Frame[] = []
  const totalFrames = 250

  for (let i = 0; i < totalFrames; i++) {
    const progress = i / totalFrames
    const phases = ['baseline', 'transition', 'reorganization']
    let phase = phases[0]
    if (progress > 0.4) phase = phases[1]
    if (progress > 0.7) phase = phases[2]

    const drift = Math.sin(progress * Math.PI * 2) * 0.3 + 0.4 + progress * 0.2
    const stability = 0.8 - progress * 0.4 + Math.sin(progress * Math.PI * 1.5) * 0.1
    const confidence = 0.5 + progress * 0.4

    frames.push({
      index: i + 1,
      total: totalFrames,
      timestamp: new Date(Date.now() - (totalFrames - i) * 60000).toISOString(),
      phase,
      system_health: progress > 0.7 ? 'degraded' : progress > 0.4 ? 'watch' : 'nominal',
      confidence: Math.max(0, Math.min(1, confidence)),
      structural_drift_score: Math.max(0, Math.min(1, drift)),
      relational_stability_score: Math.max(0, Math.min(1, stability)),
      coherence_score: Math.max(0, Math.min(1, 0.7 + progress * 0.2)),
      event_admitted: progress > 0.6,
      transition_type: phase === 'baseline' ? 'STABLE' : phase === 'transition' ? 'TRANSITION' : 'REORGANIZATION',
      persistence_minutes: Math.floor(progress * 60),
    })
  }

  return frames
}
