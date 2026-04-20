'use client'

import { useEffect, useState, useRef, useMemo } from 'react'
import { generateContinuousDemoFrame } from '@/lib/continuousDemoData'
import { PerceptualStateManager } from '@/lib/perceptualSmoothing'
import { HeroSection } from './HeroSection'
import { OperatorDecisionPanel } from './OperatorDecisionPanel'
import { HistoricalValidation } from './HistoricalValidation'
import { SubsystemCardsSection } from './SubsystemCardsSection'
import { MetricsGridSection } from './MetricsGridSection'
import { StateEvolutionSection } from './StateEvolutionSection'
import { InsightsSection } from './InsightsSection'

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
  const [totalFrames] = useState(2700)
  const [displayTimeSeconds, setDisplayTimeSeconds] = useState(0)

  const smoothingRef = useRef(new PerceptualStateManager())
  const internalSimLoopRef = useRef<NodeJS.Timeout>()
  const displayUpdateLoopRef = useRef<NodeJS.Timeout>()

  // Calculate lead time from timeline data
  const leadTimeEvent = useMemo(() => {
    if (!unifiedState?.timeline_states || unifiedState.timeline_states.length < 3) {
      return null
    }

    const timeline = unifiedState.timeline_states
    const stateThresholds = [
      { label: 'Stable', max: 0.2 },
      { label: 'Drift', max: 0.4 },
      { label: 'Instability', max: 0.6 },
      { label: 'Critical', max: 1.0 },
    ]

    // Find drift onset
    let driftOnsetIdx = -1
    for (let i = 1; i < timeline.length; i++) {
      const prev = timeline[i - 1]
      const curr = timeline[i]
      const acceleration = curr.drift - prev.drift
      if (curr.drift >= 0.08 && acceleration > 0.008) {
        driftOnsetIdx = i
        break
      }
    }

    if (driftOnsetIdx < 0) return null

    // Find first transition after onset
    let lastState = stateThresholds.findIndex((s) => timeline[0].drift <= s.max)
    for (let i = 1; i < timeline.length; i++) {
      const currentState = stateThresholds.findIndex((s) => timeline[i].drift <= s.max)
      if (currentState > lastState && i > driftOnsetIdx) {
        return {
          driftOnsetIndex: driftOnsetIdx,
          transitionIndex: i,
          leadTimeCycles: i - driftOnsetIdx,
        }
      }
      lastState = currentState
    }

    return null
  }, [unifiedState?.timeline_states])

  // Initialize
  useEffect(() => {
    const firstFrame = generateContinuousDemoFrame(0, 2700)
    const smoothing = smoothingRef.current

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
      rooms: [],
      subsystems: [],
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

  // Internal simulation loop
  useEffect(() => {
    const smoothing = smoothingRef.current

    internalSimLoopRef.current = setInterval(() => {
      const frame = generateContinuousDemoFrame(frameIndex, 2700)

      smoothing.updateTargetValue('coherence', frame.coherence)
      smoothing.updateTargetValue('drift', frame.drift)
      smoothing.updateTargetValue('fragility', frame.fragility)
      smoothing.updateTargetValue('confidence', frame.confidence)
      smoothing.updateTargetValue('climate_drift', frame.subsystems.climate.drift)
      smoothing.updateTargetValue('airflow_drift', frame.subsystems.airflow.drift)
      smoothing.updateTargetValue('irrigation_drift', frame.subsystems.irrigation.drift)
      smoothing.updateTargetValue('plant_drift', frame.subsystems.plant.drift)

      smoothing.updateSmoothedValues(16.67)
    }, 16.67)

    return () => {
      if (internalSimLoopRef.current) clearInterval(internalSimLoopRef.current)
    }
  }, [frameIndex])

  // Geometry updates (tetrahedron follows frame immediately)
  useEffect(() => {
    const frame = generateContinuousDemoFrame(frameIndex, 2700)
    const timeSeconds = Math.floor(frameIndex / 30)
    setDisplayTimeSeconds(timeSeconds)

    let sysState = 'stable'
    if (frame.drift > 0.6) sysState = 'critical'
    else if (frame.drift > 0.4) sysState = 'instability'
    else if (frame.drift > 0.2) sysState = 'drift'

    setUnifiedState((prev) =>
      prev
        ? {
            ...prev,
            drift: frame.drift,
            coherence: frame.coherence,
            stability: 1.0 - frame.drift,
            state: sysState,
            subsystems: [
              {
                subsystem_id: 'climate',
                subsystem_name: 'Climate',
                drift_contribution_pct: frame.subsystems.climate.drift * 100,
                confidence: frame.subsystems.climate.confidence,
                fragility_pct: frame.fragility * 100,
                behavioral_state: frame.subsystems.climate.drift > 0.2 ? 'Coupling' : 'Stable',
              },
              {
                subsystem_id: 'airflow',
                subsystem_name: 'Airflow',
                drift_contribution_pct: frame.subsystems.airflow.drift * 100,
                confidence: frame.subsystems.airflow.confidence,
                fragility_pct: frame.fragility * 100,
                behavioral_state: frame.subsystems.airflow.drift > 0.2 ? 'Primary' : 'Stable',
              },
              {
                subsystem_id: 'irrigation',
                subsystem_name: 'Irrigation',
                drift_contribution_pct: frame.subsystems.irrigation.drift * 100,
                confidence: frame.subsystems.irrigation.confidence,
                fragility_pct: frame.fragility * 100,
                behavioral_state: frame.subsystems.irrigation.drift > 0.2 ? 'Coupled' : 'Stable',
              },
              {
                subsystem_id: 'plant',
                subsystem_name: 'Plant',
                drift_contribution_pct: frame.subsystems.plant.drift * 100,
                confidence: frame.subsystems.plant.confidence,
                fragility_pct: frame.fragility * 100,
                behavioral_state: 'Responsive',
              },
            ],
            critical_alerts: frame.drift > 0.5 ? ['System under stress'] : [],
            insights: {
              current_state_insight:
                frame.drift < 0.15
                  ? 'System operating nominally'
                  : frame.drift < 0.35
                    ? 'Subtle asymmetry detected'
                    : frame.drift < 0.55
                      ? 'Drift spreading through coupling'
                      : frame.drift < 0.75
                        ? 'System deformation accelerating'
                        : 'Critical stress detected',
              operator_focus_insight:
                frame.drift < 0.15
                  ? 'Continue routine monitoring'
                  : frame.drift < 0.35
                    ? 'Monitor for escalation'
                    : frame.drift < 0.55
                      ? 'Prepare for intervention'
                      : 'Intervention required',
              recoverability_context: frame.drift > 0.65 ? 'Window closing—action required' : '',
            },
          }
        : prev
    )
  }, [frameIndex])

  // Display update loop (throttled)
  useEffect(() => {
    const smoothing = smoothingRef.current

    displayUpdateLoopRef.current = setInterval(() => {
      if (!smoothing.shouldUpdateUI()) return
      // UI already updated via geometry updates above
    }, 150)

    return () => {
      if (displayUpdateLoopRef.current) clearInterval(displayUpdateLoopRef.current)
    }
  }, [])

  // Playback advancement
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
    }, (33.33 / playbackSpeed) * 0.5)

    return () => clearInterval(interval)
  }, [isPlaying, playbackSpeed])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-black">
        <div className="text-lg font-light text-white/60 tracking-widest">
          Initializing System Intelligence…
        </div>
      </div>
    )
  }

  if (!unifiedState) {
    return (
      <div className="flex items-center justify-center h-screen bg-black">
        <div className="text-lg font-light text-white/60">No data available</div>
      </div>
    )
  }

  return (
    <div className="w-full min-h-screen bg-black text-white font-sans">
      {/* Global playback controls - fixed at top */}
      <div className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-b from-black via-black/80 to-transparent pointer-events-none">
        <div className="pointer-events-auto flex items-center justify-between px-8 py-6">
          {/* Title */}
          <div>
            <h1 className="text-2xl font-light tracking-widest text-white">NERAIUM</h1>
            <p className="text-xs font-light text-white/40 tracking-wide mt-1">System Intelligence</p>
          </div>

          {/* Time and controls */}
          <div className="flex items-center gap-8">
            {/* Time display */}
            <div className="text-right">
              <div className="text-3xl font-light tabular-nums text-white">
                {String(displayTimeSeconds).padStart(2, '0')}s
                <span className="text-lg text-white/30 ml-1">/ 90s</span>
              </div>
            </div>

            {/* Control buttons */}
            <div className="flex gap-3">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-12 h-12 flex items-center justify-center border border-white/20 hover:border-white/40 rounded-lg transition-all hover:bg-white/5"
                title={isPlaying ? 'Pause' : 'Play'}
              >
                {isPlaying ? '⏸' : '▶'}
              </button>
              <button
                onClick={() => {
                  setFrameIndex(0)
                  setIsPlaying(false)
                }}
                className="w-12 h-12 flex items-center justify-center border border-white/20 hover:border-white/40 rounded-lg transition-all hover:bg-white/5"
                title="Reset"
              >
                ⟲
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="relative pt-32">
        {/* HERO SECTION - Full viewport height */}
        <HeroSection state={unifiedState} />

        {/* OPERATOR DECISION PANEL - Critical action guidance */}
        <div className="relative bg-black py-20 px-12 border-t border-white/5">
          <OperatorDecisionPanel state={unifiedState} leadTimeEvent={leadTimeEvent} />
        </div>

        {/* HISTORICAL VALIDATION - Trust layer */}
        {leadTimeEvent && (
          <div className="relative bg-black py-12 px-12 border-t border-white/5">
            <HistoricalValidation leadTimeEvent={leadTimeEvent} />
          </div>
        )}

        {/* SUBSYSTEM ANALYSIS - Below fold */}
        <div className="relative bg-black py-20 px-12 border-t border-white/5">
          <SubsystemCardsSection subsystems={unifiedState.subsystems} />
        </div>

        {/* METRICS GRID */}
        <div className="relative bg-black py-20 px-12 border-t border-white/5">
          <MetricsGridSection state={unifiedState} />
        </div>

        {/* STATE EVOLUTION */}
        <div className="relative bg-black py-20 px-12 border-t border-white/5">
          <StateEvolutionSection
            timeline={unifiedState.timeline_states}
            noActionProjection={unifiedState.no_action_projection}
            insights={unifiedState.insights}
          />
        </div>

        {/* INSIGHTS BLOCK */}
        <div className="relative bg-black py-20 px-12 border-t border-white/5">
          <InsightsSection state={unifiedState} />
        </div>

        {/* Bottom padding */}
        <div className="h-20" />
      </div>
    </div>
  )
}
