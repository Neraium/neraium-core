'use client'

import React from 'react'
import { useLiveSimulation } from '@/lib/simulation'
import { SystemField } from './SystemField'
import { SubsystemPanel } from './SubsystemPanel'
import { IntelligencePanel } from './IntelligencePanel'
import { NarrativeTimeline } from './NarrativeTimeline'
import { RoomSelector } from './RoomSelector'
import { RoomDetail } from './RoomDetail'

const C = {
  text: '#e8e9eb',
  muted: '#3a3f4a',
  neon: '#7cb342',
  surface: '#080a0d',
  border: 'rgba(255,255,255,0.04)',
}

function fmt2d(n: number): string {
  return n.toString().padStart(2, '0')
}

export function LouddpaxDashboard() {
  const {
    live,
    isPlaying,
    setIsPlaying,
    selectedRoomId,
    setSelectedRoomId,
    stepScenario,
  } = useLiveSimulation()

  const now = live.timestamp
  const timeStr = `${fmt2d(now.getHours())}:${fmt2d(now.getMinutes())}`

  const col =
    live.severity === 'HIGH' ? '#e05c5c'
    : live.severity === 'ELEVATED' ? '#ff9e6d'
    : live.severity === 'MODERATE' ? '#5c9dff'
    : '#7cb342'

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      width: '100vw', height: '100vh',
      background: '#030405',
      color: C.text,
      fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',
      overflow: 'hidden',
      position: 'relative',
    }}>
      {/* ═══ HEADER ═══════════════════════════════════════════════════════ */}
      <header style={{
        flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 64, padding: '0 20px',
        borderBottom: `1px solid ${C.border}`,
        background: 'rgba(3,4,5,0.98)',
      }}>
        {/* brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <img src="/louddpax-logo.jpg" alt="Louddpax" style={{ height: 42, width: 42, objectFit: 'cover', borderRadius: 8 }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            <span style={{ fontSize: 14, fontWeight: 800, letterSpacing: '0.08em', color: '#fff' }}>
              LOUDDPAX
            </span>
            <span style={{ fontSize: 8, color: C.muted, letterSpacing: '0.14em', fontWeight: 700, textTransform: 'uppercase' }}>
              Grow Facility
            </span>
          </div>
          <span style={{ width: 1, height: 26, background: 'rgba(255,255,255,0.05)', margin: '0 4px' }} />
          <img src="/neraium-logo.jpg" alt="Neraium" style={{ height: 30, width: 30, objectFit: 'cover', borderRadius: 6, opacity: 0.55 }} />
        </div>

        {/* controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button onClick={() => stepScenario(-1)} style={controlBtnStyle}>‹</button>
          <button
            onClick={() => setIsPlaying(p => !p)}
            style={{ ...controlBtnStyle, width: 34, height: 34, borderRadius: 8, color: isPlaying ? C.neon : C.text, borderColor: isPlaying ? `${C.neon}40` : C.border }}
          >
            {isPlaying ? '‖' : '▶'}
          </button>
          <button onClick={() => stepScenario(1)} style={controlBtnStyle}>›</button>
        </div>

        {/* right */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: col, boxShadow: `0 0 10px ${col}` }} />
            <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.12em', color: col }}>
              {live.stage.replace(/_/g, ' ')}
            </span>
          </div>
          <span style={{ width: 1, height: 18, background: 'rgba(255,255,255,0.05)' }} />
          <span style={{ fontSize: 10, color: C.muted, fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
            {timeStr}
          </span>
          <span style={{
            fontSize: 7, color: C.neon, letterSpacing: '0.16em', fontWeight: 800,
            border: `1px solid ${C.neon}25`, borderRadius: 4,
            padding: '3px 8px', background: `${C.neon}06`,
          }}>
            LIVE
          </span>
        </div>
      </header>

      {/* ═══ ROOM SELECTOR ════════════════════════════════════════════════ */}
      <RoomSelector
        system={live}
        selectedId={selectedRoomId}
        onSelect={setSelectedRoomId}
      />

      {/* ═══ MAIN ═════════════════════════════════════════════════════════ */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0, position: 'relative' }}>

        {/* LEFT — Hero System Field */}
        <div style={{
          flex: '0 0 52%',
          borderRight: `1px solid ${C.border}`,
          position: 'relative', overflow: 'hidden',
        }}>
          <SystemField system={live} />
        </div>

        {/* RIGHT — Intelligence + Subsystems */}
        <div style={{
          flex: '0 0 48%',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
          background: C.surface,
        }}>
          <div style={{ flex: '0 0 auto', maxHeight: '52%', overflowY: 'auto', borderBottom: `1px solid ${C.border}` }}>
            <IntelligencePanel system={live} />
          </div>
          <div style={{ flex: '1 1 auto', overflowY: 'auto' }}>
            <SubsystemPanel system={live} />
          </div>
        </div>

        {/* ROOM DETAIL */}
        {selectedRoomId && (
          <RoomDetail
            system={live}
            roomId={selectedRoomId}
            onClose={() => setSelectedRoomId(null)}
          />
        )}
      </div>

      {/* ═══ BOTTOM — Timeline ════════════════════════════════════════════ */}
      <div style={{ flexShrink: 0, borderTop: `1px solid ${C.border}`, background: C.surface }}>
        <NarrativeTimeline system={live} />
      </div>
    </div>
  )
}

const controlBtnStyle: React.CSSProperties = {
  width: 28, height: 28, borderRadius: 6,
  background: '#0f1116', border: `1px solid ${C.border}`,
  color: '#6b7080', fontSize: 14, cursor: 'pointer',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
}
