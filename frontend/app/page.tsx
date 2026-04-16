'use client'

import { useEffect, useState } from 'react'
import { fetchDemoInit } from '@/lib/api'
import { DemoFrame } from '@/lib/types'
import HeaderBar from '@/components/HeaderBar'
import TetrahedronPanel from '@/components/TetrahedronPanel'
import InsightPanels from '@/components/InsightPanels'

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

export default function Home() {
  const [currentFrame, setCurrentFrame] = useState<Frame | null>(null)
  const [loading, setLoading] = useState(true)
  const [isConnected, setIsConnected] = useState(false)

  // Load demo data from the correct backend endpoint
  useEffect(() => {
    const loadDemoData = async () => {
      try {
        const demoData = await fetchDemoInit()
        if (demoData.initial_frame) {
          const transformedFrame = transformFrame(demoData.initial_frame)
          setCurrentFrame(transformedFrame)
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
