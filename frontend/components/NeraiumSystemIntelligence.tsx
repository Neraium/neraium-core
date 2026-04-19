'use client'

import { useEffect, useState, useRef } from 'react'
import { generateContinuousDemoFrame } from '@/lib/continuousDemoData'
import { PerceptualStateManager, TextUpdateManager } from '@/lib/perceptualSmoothing'
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
  const [totalFrames, setTotalFrames] = useState(2700) // 90 seconds at 30fps
  const railRef = useRef<HTMLDivElement>(null)

  // Perceptual smoothing managers
  const smoothingRef = useRef(new PerceptualStateManager())
  const textManagerRef = useRef(new TextUpdateManager())
  const lastDisplayUpdateRef = useRef(0)
  const internalSimLoopRef = useRef<NodeJS.Timeout>()
  const displayUpdateLoopRef = useRef<NodeJS.Timeout>()

  // Initialize with first frame
  useEffect(() => {
    const firstFrame = generateContinuousDemoFrame(0, 2700)
    const smoothing = smoothingRef.current

    // Initialize smoothed values
    smoothing.updateTargetValue('coherence', firstFrame.coherence)
    smoothing.updateTargetValue('drift', firstFrame.drift)
    smoothing.updateTargetValue('fragility', firstFrame.fragility)
    smoothing.updateTargetValue('confidence', firstFrame.confidence)
    smoothing.updateTargetValue('climate_drift', firstFrame.subsystems.climate.drift)
    smoothing.updateTargetValue('airflow_drift', firstFrame.subsystems.airflow.drift)
    smoothing.updateTargetValue('irrigation_drift', firstFrame.subsystems.irrigation.drift)
    smoothing.updateTargetValue('plant_drift', firstFrame.subsystems.plant.drift)

    const systemState: UnifiedState = {
      timestamp: firstFrame.timestamp,
      state: 'stable',
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
          drift_contribution_pct: 0,
          confidence: 0.9,
          fragility_pct: 0,
          behavioral_state: 'Nominal',
        },
        {
          subsystem_id: 'airflow',
          subsystem_name: 'Airflow',
          drift_contribution_pct: 0,
          confidence: 0.9,
          fragility_pct: 0,
          behavioral_state: 'Nominal',
        },
        {
          subsystem_id: 'irrigation',
          subsystem_name: 'Irrigation',
          drift_contribution_pct: 0,
          confidence: 0.9,
          fragility_pct: 0,
          behavioral_state: 'Nominal',
        },
        {
          subsystem_id: 'plant',
          subsystem_name: 'Plant Response',
          drift_contribution_pct: 0,
          confidence: 0.9,
          fragility_pct: 0,
          behavioral_state: 'Nominal',
        },
      ],
      timeline_states: [],
      no_action_projection: [],
      critical_alerts: [],
      insights: {
        current_state_insight: 'System operating nominally',
        operator_focus_insight: 'Continue routine monitoring',
        recoverability_context: '',
      },
      records: [],
    }

    setUnifiedState(systemState)
    setLoading(false)
  }, [])

  // Internal simulation loop (high frequency)
  useEffect(() => {
    const smoothing = smoothingRef.current

    internalSimLoopRef.current = setInterval(() => {
      // Generate new frame continuously
      const frame = generateContinuousDemoFrame(frameIndex, 2700)

      // Update all target values (happens frequently)
      smoothing.updateTargetValue('coherence', frame.coherence)
      smoothing.updateTargetValue('drift', frame.drift)
      smoothing.updateTargetValue('fragility', frame.fragility)
      smoothing.updateTargetValue('confidence', frame.confidence)
      smoothing.updateTargetValue('climate_drift', frame.subsystems.climate.drift)
      smoothing.updateTargetValue('airflow_drift', frame.subsystems.airflow.drift)
      smoothing.updateTargetValue('irrigation_drift', frame.subsystems.irrigation.drift)
      smoothing.updateTargetValue('plant_drift', frame.subsystems.plant.drift)

      // Apply exponential smoothing
      smoothing.updateSmoothedValues(16.67) // ~60fps
    }, 16.67) // Run at ~60fps

    return () => {
      if (internalSimLoopRef.current) clearInterval(internalSimLoopRef.current)
    }
  }, [frameIndex])

  // Geometry update loop (follows frame directly, high frequency)
  // This updates the tetrahedron without delay
  useEffect(() => {
    const updateGeometry = () => {
      const frame = generateContinuousDemoFrame(frameIndex, 2700)

      setUnifiedState((prev) =>
        prev
          ? {
              ...prev,
              // Geometry updates immediately (no smoothing)
              drift: frame.drift,
              coherence: frame.coherence,
              stability: 1.0 - frame.drift,
              time: frame.time,
              timestamp: frame.timestamp,
              // Subsystems for tetrahedron geometry
              subsystems: [
                {
                  subsystem_id: 'climate',
                  subsystem_name: 'Climate',
                  drift_contribution_pct: frame.subsystems.climate.drift * 100,
                  confidence: frame.subsystems.climate.confidence,
                  fragility_pct: frame.fragility * 100,
                  behavioral_state: 'Monitoring',
                },
                {
                  subsystem_id: 'airflow',
                  subsystem_name: 'Airflow',
                  drift_contribution_pct: frame.subsystems.airflow.drift * 100,
                  confidence: frame.subsystems.airflow.confidence,
                  fragility_pct: frame.fragility * 100,
                  behavioral_state: 'Monitoring',
                },
                {
                  subsystem_id: 'irrigation',
                  subsystem_name: 'Irrigation',
                  drift_contribution_pct: frame.subsystems.irrigation.drift * 100,
                  confidence: frame.subsystems.irrigation.confidence,
                  fragility_pct: frame.fragility * 100,
                  behavioral_state: 'Monitoring',
                },
                {
                  subsystem_id: 'plant',
                  subsystem_name: 'Plant Response',
                  drift_contribution_pct: frame.subsystems.plant.drift * 100,
                  confidence: frame.subsystems.plant.confidence,
                  fragility_pct: frame.fragility * 100,
                  behavioral_state: 'Monitoring',
                },
              ],
            }
          : prev
      )
    }

    updateGeometry()
  }, [frameIndex])

  // Display update loop (low frequency, throttled)
  useEffect(() => {
    const smoothing = smoothingRef.current
    const textManager = textManagerRef.current

    displayUpdateLoopRef.current = setInterval(() => {
      // Check if we should update UI
      if (!smoothing.shouldUpdateUI()) return

      // Build display state from smoothed values
      const displayDrift = smoothing.getDisplayValue('drift')
      const displayCoherence = smoothing.getDisplayValue('coherence')
      const displayFragility = smoothing.getDisplayValue('fragility')
      const displayConfidence = smoothing.getDisplayValue('confidence')

      const climateDrift = smoothing.getDisplayValue('climate_drift')
      const airflowDrift = smoothing.getDisplayValue('airflow_drift')
      const irrigationDrift = smoothing.getDisplayValue('irrigation_drift')
      const plantDrift = smoothing.getDisplayValue('plant_drift')

      // Determine system state (smoothly)
      let sysState = 'stable'
      if (displayDrift > 0.6) sysState = 'critical'
      else if (displayDrift > 0.4) sysState = 'instability'
      else if (displayDrift > 0.2) sysState = 'drift'

      // Build insights with text fade transitions
      let currentStateInsight = 'System operating nominally'
      let operatorFocusInsight = 'Continue routine monitoring'
      let recoverabilityContext = ''

      if (displayDrift < 0.15) {
        currentStateInsight = 'System operating nominally'
        operatorFocusInsight = 'Continue routine monitoring'
      } else if (displayDrift < 0.35) {
        currentStateInsight = 'Subtle asymmetry detected in subsystems'
        operatorFocusInsight = 'Monitor for escalation'
      } else if (displayDrift < 0.55) {
        currentStateInsight = 'Drift beginning to spread through system coupling'
        operatorFocusInsight = 'Prepare for intervention'
        recoverabilityContext = 'Recovery window narrowing—action may still prevent escalation'
      } else if (displayDrift < 0.75) {
        currentStateInsight = 'System deformation accelerating—instability evident'
        operatorFocusInsight = 'Intervention strongly recommended'
        recoverabilityContext = 'Window closing—immediate action required'
      } else {
        currentStateInsight = 'Critical system stress—deformation irreversible'
        operatorFocusInsight = 'System failure imminent'
        recoverabilityContext = 'Recovery pathway closing immediately'
      }

      // Apply text fade transitions
      const stateText = textManager.updateText('state', currentStateInsight)
      const focusText = textManager.updateText('focus', operatorFocusInsight)
      const recoverText = textManager.updateText('recoverability', recoverabilityContext)

      setUnifiedState((prev) =>
        prev
          ? {
              ...prev,
              state: sysState,
              critical_alerts: displayDrift > 0.5 ? ['System under structural stress'] : [],
              insights: {
                current_state_insight: currentStateInsight,
                operator_focus_insight: operatorFocusInsight,
                recoverability_context: recoverabilityContext,
              },
            }
          : prev
      )
    }, 150) // Update UI every 150ms (6-7fps)

    return () => {
      if (displayUpdateLoopRef.current) clearInterval(displayUpdateLoopRef.current)
    }
  }, [])

  // Playback advancement loop
  useEffect(() => {
    if (!isPlaying) return

    const interval = setInterval(() => {
      setFrameIndex((prev) => {
        const next = prev + 1
        if (next >= 2700) {
          setIsPlaying(false)
          return prev
        }
        return next
      })
    }, (33.33 / playbackSpeed) * 0.5) // Advance frames at controlled rate

    return () => clearInterval(interval)
  }, [isPlaying, playbackSpeed])

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
        <span className="text-white/50 text-xs">
          {Math.round(frameIndex / 30)}s / 90s
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
