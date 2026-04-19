'use client'

import React from 'react'
import { LiveSystem } from '@/lib/simulation'
import { FacilityHeader } from './FacilityHeader'
import { FacilityMap } from './FacilityMap'
import { FacilityIntelligencePanel } from './FacilityIntelligencePanel'
import { NarrativeTimeline } from './NarrativeTimeline'

interface Props {
  system: LiveSystem
  onSelectRoom: (id: string) => void
  isPlaying?: boolean
  onTogglePlay?: () => void
  onStep?: (delta: number) => void
}

export function FacilityOverview({ system, onSelectRoom, isPlaying, onTogglePlay, onStep }: Props) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      width: '100%', height: '100%',
      background: '#030405',
      color: '#e8e9eb',
      fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',
      overflow: 'hidden',
    }}>
      <FacilityHeader
        system={system}
        isPlaying={isPlaying}
        onTogglePlay={onTogglePlay}
        onStep={onStep}
      />

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        <div style={{
          flex: '0 0 62%',
          position: 'relative', overflow: 'hidden',
          borderRight: '1px solid rgba(255,255,255,0.04)',
        }}>
          <FacilityMap system={system} onSelectRoom={onSelectRoom} />
        </div>
        <div style={{
          flex: '0 0 38%',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}>
          <FacilityIntelligencePanel system={system} />
        </div>
      </div>

      <div style={{ flexShrink: 0, borderTop: '1px solid rgba(255,255,255,0.04)', background: '#080a0d' }}>
        <NarrativeTimeline system={system} />
      </div>
    </div>
  )
}
