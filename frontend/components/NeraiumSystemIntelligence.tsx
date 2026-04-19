'use client'

import { useEffect, useState, useRef } from 'react'
import { HeroSystemOverview } from './HeroSystemOverview'
import { SubsystemAnalysis } from './SubsystemAnalysis'
import { StateEvolutionSection } from './StateEvolutionSection'
import { IntelligenceRailSticky } from './IntelligenceRailSticky'

interface UnifiedState {
  timestamp: string
  state: string
  drift: number
  coherence: number
  stability: number
  rooms: any[]
  subsystems: any[]
  timeline_states: any[]
  no_action_projection: any[]
  critical_alerts: string[]
  insights: any
  records: any[]
}

export function NeraiumSystemIntelligence() {
  const [unifiedState, setUnifiedState] = useState<UnifiedState | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const heroRef = useRef<HTMLDivElement>(null)
  const railRef = useRef<HTMLDivElement>(null)

  // Fetch unified state
  useEffect(() => {
    const fetchState = async () => {
      try {
        setLoading(true)
        const response = await fetch(`/api/system/unified-state?use_synthetic=true`)
        if (!response.ok) throw new Error(`API error: ${response.status}`)
        const data = await response.json()
        setUnifiedState(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    fetchState()
  }, [])

  // Handle playback
  useEffect(() => {
    if (!isPlaying || !unifiedState?.records) return

    const interval = setInterval(() => {
      setFrameIndex((prev) => {
        const next = prev + 1
        if (next >= unifiedState.records.length) {
          setIsPlaying(false)
          return prev
        }
        return next
      })
    }, 750 / playbackSpeed)

    return () => clearInterval(interval)
  }, [isPlaying, playbackSpeed, unifiedState?.records])

  // Fetch frame-specific state
  useEffect(() => {
    if (!unifiedState?.records) return

    const fetchFrame = async () => {
      try {
        const response = await fetch(`/api/system/frame/${frameIndex}?use_synthetic=true`)
        if (!response.ok) throw new Error(`API error: ${response.status}`)
        const data = await response.json()
        setUnifiedState(data)
      } catch (err) {
        console.error('Failed to fetch frame:', err)
      }
    }

    if (frameIndex > 0) {
      fetchFrame()
    }
  }, [frameIndex])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-black text-white">
        <div>Loading System Intelligence...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-black text-red-500">
        <div>Error: {error}</div>
      </div>
    )
  }

  if (!unifiedState) {
    return (
      <div className="flex items-center justify-center h-screen bg-black text-white">
        <div>No data available</div>
      </div>
    )
  }

  return (
    <div className="relative w-full min-h-screen bg-black text-white overflow-x-hidden">
      {/* Playback controls */}
      <div className="fixed top-4 left-4 z-40 bg-black/80 backdrop-blur-sm border border-white/10 rounded p-4 flex gap-4 items-center">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-3 py-1 bg-white/10 hover:bg-white/20 rounded text-sm"
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={() => {
            setFrameIndex(0)
            setIsPlaying(false)
          }}
          className="px-3 py-1 bg-white/10 hover:bg-white/20 rounded text-sm"
        >
          Reset
        </button>
        <input
          type="range"
          min="0"
          max={unifiedState.records.length - 1}
          value={frameIndex}
          onChange={(e) => setFrameIndex(parseInt(e.target.value))}
          className="w-32"
        />
        <span className="text-xs text-white/60">{frameIndex + 1} / {unifiedState.records.length}</span>
        <div className="flex items-center gap-2">
          <label className="text-xs text-white/60">Speed:</label>
          <input
            type="range"
            min="0.5"
            max="2"
            step="0.1"
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
            className="w-24"
          />
          <span className="text-xs text-white/60">{playbackSpeed.toFixed(1)}x</span>
        </div>
      </div>

      {/* Main scroll surface */}
      <div className="relative">
        {/* SECTION 1: Hero System Overview */}
        <div ref={heroRef} className="relative w-full h-screen sticky top-0 z-10">
          <HeroSystemOverview state={unifiedState} railRef={railRef} />

          {/* Intelligence rail - sticky to hero section */}
          <div
            ref={railRef}
            className="absolute top-0 right-0 h-full w-80 z-20 pointer-events-auto overflow-y-auto"
            style={{
              background: 'linear-gradient(90deg, transparent 0%, rgba(0,0,0,0.4) 20%, rgba(0,0,0,0.8) 100%)',
            }}
          >
            <IntelligenceRailSticky state={unifiedState} />
          </div>
        </div>

        {/* SECTION 2: Subsystem Analysis */}
        <div className="relative w-full bg-black py-24 px-8">
          <SubsystemAnalysis subsystems={unifiedState.subsystems} />
        </div>

        {/* SECTION 3: State Evolution */}
        <div className="relative w-full bg-black py-24 px-8 border-t border-white/5">
          <StateEvolutionSection
            timeline={unifiedState.timeline_states}
            noActionProjection={unifiedState.no_action_projection}
            insights={unifiedState.insights}
          />
        </div>
      </div>
    </div>
  )
}
