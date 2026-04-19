'use client'

import React, { useState } from 'react'
import { LiveSystem } from '@/lib/simulation'
import { FacilityHeader } from './FacilityHeader'
import { SystemCoherenceCore } from './SystemCoherenceCore'
import { FacilityIntelligencePanel } from './FacilityIntelligencePanel'
import { SubsystemPanel } from './SubsystemPanel'
import { NarrativeTimeline } from './NarrativeTimeline'

interface Props {
  system: LiveSystem
  onSelectRoom: (id: string) => void
  isPlaying?: boolean
  onTogglePlay?: () => void
  onStep?: (delta: number) => void
}

export function FacilityOverview({ system, onSelectRoom, isPlaying, onTogglePlay, onStep }: Props) {
  const [hoveredRoomId, setHoveredRoomId] = useState<string | null>(null)
  const [hoveredStage, setHoveredStage] = useState<string | null>(null)

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      width: '100%', height: '100%',
      background: '#030405',
      color: '#e8e9eb',
      fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',
      overflow: 'hidden',
    }}>
      {/* 1. FACILITY COMMAND STRIP */}
      <FacilityHeader
        system={system}
        isPlaying={isPlaying}
        onTogglePlay={onTogglePlay}
        onStep={onStep}
      />

      {/* Main body */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        {/* 2. HERO SYSTEM FIELD — dominant center */}
        <div style={{
          flex: '1 1 0%',
          position: 'relative', overflow: 'hidden',
          display: 'flex', flexDirection: 'column',
        }}>
          {/* System Coherence Core — takes most space */}
          <div style={{ flex: '1 1 0%', position: 'relative', minHeight: 0 }}>
            <SystemCoherenceCore
              system={system}
              onSelectRoom={onSelectRoom}
              hoveredRoomId={hoveredRoomId}
              onHoverRoom={setHoveredRoomId}
            />
          </div>

          {/* 5. SUBSYSTEM BEHAVIOR PANELS — bottom of hero area */}
          <div style={{
            flexShrink: 0,
            maxHeight: 220,
            borderTop: '1px solid rgba(255,255,255,0.04)',
            background: 'linear-gradient(180deg, transparent 0%, #080a0d 100%)',
          }}>
            <SubsystemPanel
              system={system}
              onSelectRoom={onSelectRoom}
              hoveredRoomId={hoveredRoomId}
              onHoverRoom={setHoveredRoomId}
            />
          </div>
        </div>

        {/* 3. INTELLIGENCE RAIL — right side */}
        <div style={{
          flex: '0 0 340px',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
          borderLeft: '1px solid rgba(255,255,255,0.04)',
        }}>
          <FacilityIntelligencePanel
            system={system}
            hoveredRoomId={hoveredRoomId}
          />
        </div>
      </div>

      {/* 6. TIMELINE — bottom */}
      <div style={{
        flexShrink: 0,
        borderTop: '1px solid rgba(255,255,255,0.04)',
        background: '#080a0d',
      }}>
        <NarrativeTimeline
          system={system}
          onHoverStage={setHoveredStage}
        />
      </div>
    </div>
  )
}
