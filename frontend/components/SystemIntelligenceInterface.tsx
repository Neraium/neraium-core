'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { TetrahedronField } from './TetrahedronField'
import { NarrativeLayer } from './NarrativeLayer'
import { SubsystemSignals } from './SubsystemSignals'
import { TrajectoryLayer } from './TrajectoryLayer'
import { PlaybackControls } from './SystemPlaybackControls'
import { usePhaseController } from '@/lib/phaseController'
import { useSystemInterpolation } from '@/lib/systemInterpolation'

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

  // Phase system: manage discrete system states with 8-15 second transitions
  const { phase, phaseProgress } = usePhaseController(isPlaying, speed)

  // Smooth interpolation of all numeric values
  const interpolatedData = useSystemInterpolation(systemData, phase)

  // Generate narrative based on phase and data
  useEffect(() => {
    const narratives: Record<string, string[]> = {
      Stable: [
        'System coherence stable',
        'Drift minimal across couplings',
        'Recovery pathways open',
      ],
      'Drift forming': [
        'Drift spreading through airflow coupling',
        'Stability margins narrowing',
        'Coherence structure wavering',
      ],
      'Instability forming': [
        'Recovery pathway narrowing',
        'System coherence degrading',
        'Multiple coupling points destabilizing',
      ],
      Critical: [
        'Critical threshold approaching',
        'Coherence collapse imminent',
        'System reorganization likely',
      ],
    }

    const texts = narratives[phase] || narratives.Stable
    const text = texts[Math.floor(Math.random() * texts.length)]
    setNarrativeText(text)
  }, [phase])

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
            height: '80vh',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            flexShrink: 0,
          }}
        >
          <TetrahedronField data={interpolatedData} phaseProgress={phaseProgress} />
        </div>

        {/* Narrative Layer */}
        <div
          style={{
            width: '100%',
            height: 'auto',
            padding: '60px 40px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            flexShrink: 0,
          }}
        >
          <NarrativeLayer text={narrativeText} />
        </div>

        {/* Subsystem Signals */}
        <div
          style={{
            width: '100%',
            height: 'auto',
            padding: '40px',
            position: 'relative',
            flexShrink: 0,
          }}
        >
          <SubsystemSignals data={interpolatedData} />
        </div>

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
          <TrajectoryLayer data={interpolatedData} />
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
