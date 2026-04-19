'use client'

import { useEffect, useState, useRef } from 'react'
import { generateContinuousDemoFrame } from '@/lib/continuousDemoData'
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
  time?: number
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
  const [frameIndex, setFrameIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(true)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [totalFrames, setTotalFrames] = useState(1800) // 60 seconds at 30fps
  const railRef = useRef<HTMLDivElement>(null)
  const frameTimeoutRef = useRef<NodeJS.Timeout>()

  // Initialize with first frame
  useEffect(() => {
    const firstFrame = generateContinuousDemoFrame(0, totalFrames)

    // Transform to UnifiedState format
    const systemState: UnifiedState = {
      timestamp: firstFrame.timestamp,
      state: firstFrame.drift > 0.6 ? 'critical' : firstFrame.drift > 0.4 ? 'instability' : firstFrame.drift > 0.2 ? 'drift' : 'stable',
      drift: firstFrame.drift,
      coherence: firstFrame.coherence,
      stability: 1.0,
      time: firstFrame.time,
      rooms: [
        { room_id: 'climate', room_name: 'Climate', status: 'nominal' },
        { room_id: 'airflow', room_name: 'Airflow', status: 'nominal' },
        { room_id: 'irrigation', room_name: 'Irrigation', status: 'nominal' },
      ],
      subsystems: [
        {
          subsystem_id: 'climate',
          subsystem_name: 'Climate',
          drift_contribution_pct: firstFrame.subsystems.climate.drift * 100,
          confidence: firstFrame.subsystems.climate.confidence,
          fragility_pct: firstFrame.fragility * 100,
          behavioral_state: 'Monitoring',
        },
        {
          subsystem_id: 'airflow',
          subsystem_name: 'Airflow',
          drift_contribution_pct: firstFrame.subsystems.airflow.drift * 100,
          confidence: firstFrame.subsystems.airflow.confidence,
          fragility_pct: firstFrame.fragility * 100,
          behavioral_state: 'Monitoring',
        },
        {
          subsystem_id: 'irrigation',
          subsystem_name: 'Irrigation',
          drift_contribution_pct: firstFrame.subsystems.irrigation.drift * 100,
          confidence: firstFrame.subsystems.irrigation.confidence,
          fragility_pct: firstFrame.fragility * 100,
          behavioral_state: 'Monitoring',
        },
        {
          subsystem_id: 'plant',
          subsystem_name: 'Plant Response',
          drift_contribution_pct: firstFrame.subsystems.plant.drift * 100,
          confidence: firstFrame.subsystems.plant.confidence,
          fragility_pct: firstFrame.fragility * 100,
          behavioral_state: 'Monitoring',
        },
      ],
      timeline_states: [],
      no_action_projection: [],
      critical_alerts: firstFrame.drift > 0.5 ? ['System under stress'] : [],
      insights: {
        current_state_insight: 'System initializing...',
        operator_focus_insight: 'Monitor for changes',
        recoverability_context: '',
      },
      records: [],
    }

    setUnifiedState(systemState)
    setLoading(false)
  }, [])

  // Handle playback and frame generation
  useEffect(() => {
    if (!isPlaying) return

    frameTimeoutRef.current = setTimeout(() => {
      setFrameIndex((prev) => {
        const next = prev + 1
        if (next >= totalFrames) {
          setIsPlaying(false)
          return prev
        }
        return next
      })
    }, (33.33 / playbackSpeed) * 0.8) // Adjust timing for smoother playback

    return () => {
      if (frameTimeoutRef.current) clearTimeout(frameTimeoutRef.current)
    }
  }, [isPlaying, playbackSpeed, frameIndex, totalFrames])

  // Generate frame data from continuous demo
  useEffect(() => {
    const frame = generateContinuousDemoFrame(frameIndex, totalFrames)

    // Determine system state
    let sysState = 'stable'
    if (frame.drift > 0.6) sysState = 'critical'
    else if (frame.drift > 0.4) sysState = 'instability'
    else if (frame.drift > 0.2) sysState = 'drift'

    // Build insights based on current state
    let currentStateInsight = ''
    let operatorFocusInsight = ''
    let recoverabilityContext = ''

    if (frame.drift < 0.15) {
      currentStateInsight = 'System operating nominally'
      operatorFocusInsight = 'Continue routine monitoring'
    } else if (frame.drift < 0.35) {
      currentStateInsight = 'Subtle asymmetry detected in subsystems'
      operatorFocusInsight = 'Watch for escalation in Airflow coupling'
    } else if (frame.drift < 0.55) {
      currentStateInsight = 'Drift beginning to spread through system coupling'
      operatorFocusInsight = 'Prepare for intervention if trend continues'
      recoverabilityContext = 'Recovery window narrowing—action may still prevent escalation'
    } else if (frame.drift < 0.75) {
      currentStateInsight = 'System deformation accelerating—instability evident'
      operatorFocusInsight = 'Intervention strongly recommended'
      recoverabilityContext = 'Window closing—immediate action required'
    } else {
      currentStateInsight = 'Critical system stress—deformation irreversible'
      operatorFocusInsight = 'System failure imminent'
      recoverabilityContext = 'Recovery pathway closing immediately'
    }

    setUnifiedState((prev) =>
      prev
        ? {
            ...prev,
            timestamp: frame.timestamp,
            state: sysState,
            drift: frame.drift,
            coherence: frame.coherence,
            stability: 1.0 - frame.drift,
            time: frame.time,
            subsystems: [
              {
                subsystem_id: 'climate',
                subsystem_name: 'Climate',
                drift_contribution_pct: frame.subsystems.climate.drift * 100,
                confidence: frame.subsystems.climate.confidence,
                fragility_pct: frame.fragility * 100,
                behavioral_state: frame.subsystems.climate.drift > 0.2 ? 'Coupling spreading' : 'Stable',
              },
              {
                subsystem_id: 'airflow',
                subsystem_name: 'Airflow',
                drift_contribution_pct: frame.subsystems.airflow.drift * 100,
                confidence: frame.subsystems.airflow.confidence,
                fragility_pct: frame.fragility * 100,
                behavioral_state: frame.subsystems.airflow.drift > 0.2 ? 'Primary driver' : 'Stable',
              },
              {
                subsystem_id: 'irrigation',
                subsystem_name: 'Irrigation',
                drift_contribution_pct: frame.subsystems.irrigation.drift * 100,
                confidence: frame.subsystems.irrigation.confidence,
                fragility_pct: frame.fragility * 100,
                behavioral_state: frame.subsystems.irrigation.drift > 0.2 ? 'Coupled response' : 'Stable',
              },
              {
                subsystem_id: 'plant',
                subsystem_name: 'Plant Response',
                drift_contribution_pct: frame.subsystems.plant.drift * 100,
                confidence: frame.subsystems.plant.confidence,
                fragility_pct: frame.fragility * 100,
                behavioral_state: frame.subsystems.plant.drift > 0.2 ? 'Stress response' : 'Nominal',
              },
            ],
            critical_alerts: frame.drift > 0.5 ? ['System under structural stress'] : [],
            insights: {
              current_state_insight: currentStateInsight,
              operator_focus_insight: operatorFocusInsight,
              recoverability_context: recoverabilityContext,
            },
          }
        : prev
    )
  }, [frameIndex, totalFrames])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-black text-white">
        <div>Initializing System Intelligence...</div>
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
      {/* Playback controls - minimal and unobtrusive */}
      <div className="fixed top-4 right-4 z-40 bg-black/80 backdrop-blur-sm border border-white/10 rounded p-3 flex gap-2 items-center text-xs">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-2 py-1 bg-white/10 hover:bg-white/20 rounded text-xs"
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button
          onClick={() => {
            setFrameIndex(0)
            setIsPlaying(false)
          }}
          className="px-2 py-1 bg-white/10 hover:bg-white/20 rounded text-xs"
        >
          ⟲
        </button>
        <input
          type="range"
          min="0"
          max={totalFrames - 1}
          value={frameIndex}
          onChange={(e) => {
            setFrameIndex(parseInt(e.target.value))
            setIsPlaying(false)
          }}
          className="w-24"
        />
        <span className="text-white/50">
          {frameIndex} / {totalFrames}
        </span>
      </div>

      {/* Main scroll surface */}
      <div className="relative">
        {/* SECTION 1: Hero System Overview */}
        <div className="relative w-full h-screen sticky top-0 z-10">
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
        <div className="relative w-full bg-black py-24 px-8 border-t border-white/5">
          <SubsystemAnalysis subsystems={unifiedState.subsystems} />
        </div>

        {/* SECTION 3: State Evolution */}
        <div className="relative w-full bg-black py-24 px-8 border-t border-white/5">
          <StateEvolutionSection
            timeline={[]}
            noActionProjection={[]}
            insights={unifiedState.insights}
          />
        </div>
      </div>
    </div>
  )
}
