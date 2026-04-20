'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { TetrahedronField } from './TetrahedronField'
import { NarrativeLayer } from './NarrativeLayer'
import { TrajectoryLayer } from './TrajectoryLayer'
import { PlaybackControls } from './SystemPlaybackControls'
import { ConsequenceIndicator } from './ConsequenceIndicator'
import { usePhaseController } from '@/lib/phaseController'
import { useSystemInterpolation } from '@/lib/systemInterpolation'
import { computeConsequenceState, getEscalationNarrative } from '@/lib/decisionGravity'
import { computeDiagnosticLegibility } from '@/lib/diagnosticLegibility'
import { computeOutcomeConfidence } from '@/lib/outcomeConfidence'

interface SystemData {
  drift: number
  stability: number
  coherence: number
  confidence: number
  timestamp: string
  systemState: 'Stable' | 'Drift forming' | 'Instability forming' | 'Critical'
}

interface SystemIntelligenceInterfaceProps {
  systemData?: SystemData
  onTogglePlay?: (isPlaying: boolean) => void
  onSpeedChange?: (speed: number) => void
}

export function SystemIntelligenceInterface({
  systemData,
  onTogglePlay,
  onSpeedChange,
}: SystemIntelligenceInterfaceProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [narrativeText, setNarrativeText] = useState('')
  const [previousPhase, setPreviousPhase] = useState('Stable')
  const [thresholdDwellActive, setThresholdDwellActive] = useState(false)
  const [primaryDriver, setPrimaryDriver] = useState<string | null>(null) // System memory
  const [driftHistory, setDriftHistory] = useState<number[]>([])
  const [stabilityHistory, setStabilityHistory] = useState<number[]>([])

  // Phase system: manage discrete system states with 8-15 second transitions
  const { phase, phaseProgress } = usePhaseController(isPlaying, speed && !thresholdDwellActive ? 1 : 0)

  // Smooth interpolation of all numeric values
  const interpolatedData = useSystemInterpolation(systemData, phase)

  // Build history for diagnostic analysis
  useEffect(() => {
    setDriftHistory(prev => [...prev.slice(-10), interpolatedData.interpolatedDrift])
    setStabilityHistory(prev => [...prev.slice(-10), interpolatedData.interpolatedStability])
  }, [interpolatedData.interpolatedDrift, interpolatedData.interpolatedStability])

  // Compute diagnostic legibility
  const diagnostics = computeDiagnosticLegibility(
    phase,
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    driftHistory,
    stabilityHistory
  )

  // Update primary driver with system memory (preserve unless stronger one emerges)
  useEffect(() => {
    if (diagnostics.dominantDriver) {
      const isStrongerDriver = diagnostics.confidence === 'high' || !primaryDriver
      if (isStrongerDriver) {
        setPrimaryDriver(diagnostics.dominantDriver)
      }
    }
  }, [diagnostics.dominantDriver, diagnostics.confidence, primaryDriver])

  // Consequence state: time-to-impact, escalation language, operator focus
  const consequenceState = computeConsequenceState(
    phase,
    phaseProgress,
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    previousPhase
  )

  // Outcome confidence: what happens if they act or don't act
  const outcomeState = computeOutcomeConfidence(
    diagnostics.dominantDriver || 'multi-mode',
    phase,
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    diagnostics.confidence
  )

  // Track phase changes for threshold detection and dwell
  useEffect(() => {
    if (phase !== previousPhase) {
      // If crossing into instability+, activate dwell
      if ((previousPhase === 'Stable' || previousPhase === 'Drift forming') &&
          (phase === 'Instability forming' || phase === 'Critical')) {
        setThresholdDwellActive(true)
        // Dwell for 1.5-2.5 seconds
        const dwellTime = 1500 + Math.random() * 1000
        setTimeout(() => {
          setThresholdDwellActive(false)
        }, dwellTime)
      }
      setPreviousPhase(phase)
    }
  }, [phase, previousPhase])

  // Generate narrative with escalation language and decision gravity
  useEffect(() => {
    // Use escalation narrative (directive, not descriptive)
    const text = getEscalationNarrative(phase, phaseProgress)
    setNarrativeText(text)
  }, [phase, phaseProgress])

  const handleTogglePlay = () => {
    setIsPlaying(!isPlaying)
    onTogglePlay?.(!isPlaying)
  }

  const handleSpeedChange = (newSpeed: number) => {
    setSpeed(newSpeed)
    onSpeedChange?.(newSpeed)
  }

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        background: '#0a0e1a',
        color: '#e2e8f0',
        overflow: 'hidden',
        position: 'relative',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        fontSize: '14px',
      }}
    >
      {/* Subtle scanline texture overlay */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.15) 2px, rgba(0,0,0,0.15) 4px)',
          pointerEvents: 'none',
          zIndex: 99,
          opacity: 0.2,
        }}
      />

      {/* Consequence Indicator */}
      <ConsequenceIndicator
        timeToImpactLabel={consequenceState.timeToImpactLabel}
        operatorFocus={consequenceState.operatorFocus}
        escalationLevel={consequenceState.escalationLevel}
        hasThresholdCrossed={consequenceState.hasThresholdCrossed}
        confidence={diagnostics.confidence}
        actionOutcome={outcomeState.actionOutcome}
        noActionConsequence={outcomeState.noActionConsequence}
      />

      {/* Main scrollable container */}
      <div
        style={{
          width: '100%',
          height: '100%',
          overflowY: 'auto',
          overflowX: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          scrollBehavior: 'smooth',
        }}
      >
        {/* System Field - Primary Visualization */}
        <div
          style={{
            width: '100%',
            height: '85vh',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            flexShrink: 0,
          }}
        >
          <TetrahedronField
            data={interpolatedData}
            phaseProgress={phaseProgress}
            phase={phase}
            escalationLevel={consequenceState.escalationLevel}
            hasThresholdCrossed={consequenceState.hasThresholdCrossed}
            isApproachingFailure={consequenceState.isApproachingFailure}
          />
        </div>

        {/* Narrative Layer */}
        <div
          style={{
            width: '100%',
            height: 'auto',
            padding: '40px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            flexShrink: 0,
          }}
        >
          <NarrativeLayer
            text={narrativeText}
            phase={phase}
            phaseProgress={phaseProgress}
            diagnostics={{
              origin: diagnostics.origin,
              propagationPath: diagnostics.propagationPath,
              whyNow: diagnostics.whyNow,
              currentRiskZone: diagnostics.currentRiskZone,
            }}
          />
        </div>

        {/* Subsystem Signals - Removed, encoded into tetrahedron */}

        {/* Trajectory Layer */}
        <div
          style={{
            width: '100%',
            height: '320px',
            padding: '40px 0',
            position: 'relative',
            flexShrink: 0,
          }}
        >
          <TrajectoryLayer data={interpolatedData} escalationLevel={consequenceState.escalationLevel} />
        </div>

        {/* Bottom spacing */}
        <div style={{ height: '80px', flexShrink: 0 }} />
      </div>

      {/* Playback Controls - Fixed at bottom right */}
      <div
        style={{
          position: 'fixed',
          bottom: '40px',
          right: '40px',
          zIndex: 100,
        }}
      >
        <PlaybackControls
          isPlaying={isPlaying}
          speed={speed}
          onTogglePlay={handleTogglePlay}
          onSpeedChange={handleSpeedChange}
        />
      </div>
    </div>
  )
}
