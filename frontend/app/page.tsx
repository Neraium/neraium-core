'use client'

import { useEffect, useState, useRef } from 'react'
import axios from 'axios'
import HeaderBar from '@/components/HeaderBar'
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
  const [currentFrame, setCurrentFrame] = useState<Frame | null>(null)
  const [allFrames, setAllFrames] = useState<Frame[]>([])
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [isConnected, setIsConnected] = useState(false)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  // Load all frames once on mount
  useEffect(() => {
    const loadFrames = async () => {
      try {
        const response = await axios.get(`${apiBase}/api/ui/frames`)
        const frames = response.data.frames || []
        setAllFrames(frames)
        setIsConnected(true)
        setLoading(false)
      } catch (error) {
        console.error('Failed to fetch frames:', error)
        setIsConnected(false)
        setLoading(false)
      }
    }
    loadFrames()
  }, [apiBase])

  // Simulate live polling through the frame sequence
  useEffect(() => {
    if (allFrames.length === 0) return

    // Set initial frame
    setCurrentFrame(allFrames[0])
    setCurrentFrameIndex(0)

    // Poll for "new" data by advancing through frames
    // This simulates real-time updates at ~2 second intervals
    pollIntervalRef.current = setInterval(() => {
      setCurrentFrameIndex((prev) => {
        const nextIndex = (prev + 1) % allFrames.length
        setCurrentFrame(allFrames[nextIndex])
        return nextIndex
      })
    }, 2000)

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [allFrames])

  return (
    <div className="demo-app">
      {loading ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
          Connecting to live system monitoring...
        </div>
      ) : !isConnected ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
          Connection lost. Retrying...
        </div>
      ) : currentFrame ? (
        <>
          <HeaderBar frame={currentFrame} />
          <div className="demo-main">
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
