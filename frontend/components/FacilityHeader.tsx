'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { LiveSystem } from '@/lib/simulation'
import { computeFacilityContext } from '@/lib/facilityContext'

interface Props {
  system: LiveSystem
  isPlaying?: boolean
  onTogglePlay?: () => void
  onStep?: (delta: number) => void
}

const C = {
  text: '#e8e9eb',
  secondary: '#8a8f9a',
  muted: '#4a4f5a',
  neon: '#7cb342',
  blue: '#5c9dff',
  orange: '#ff9e6d',
  red: '#e05c5c',
  surface: '#080a0d',
  surfaceHi: '#0f1116',
  border: 'rgba(255,255,255,0.04)',
}

function fmt2d(n: number): string {
  return n.toString().padStart(2, '0')
}

export function FacilityHeader({ system, isPlaying, onTogglePlay, onStep }: Props) {
  const ctx = computeFacilityContext(system)
  const now = system.timestamp
  const timeStr = `${fmt2d(now.getHours())}:${fmt2d(now.getMinutes())}:${fmt2d(now.getSeconds())}`

  const attentionCol = ctx.roomsNeedingAttention > 1 ? C.red : ctx.roomsNeedingAttention > 0 ? C.orange : C.neon

  return (
    <header style={{
      flexShrink: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      height: 72, padding: '0 24px',
      borderBottom: `1px solid ${C.border}`,
      background: 'linear-gradient(180deg, rgba(5,6,8,0.98) 0%, rgba(3,4,5,0.98) 100%)',
      position: 'relative',
      zIndex: 50,
    }}>
      {/* Brand & Facility */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{
          width: 40, height: 40, borderRadius: 10,
          background: 'linear-gradient(135deg, #1a1f1a 0%, #0d110d 100%)',
          border: `1px solid ${C.neon}20`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: `0 0 20px ${C.neon}08`,
        }}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <rect x="4" y="4" width="4" height="16" rx="1" fill={C.neon} opacity="0.9" />
            <rect x="10" y="4" width="4" height="16" rx="1" fill={C.neon} opacity="0.7" />
            <rect x="16" y="8" width="4" height="12" rx="1" fill={C.neon} opacity="0.5" />
          </svg>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span style={{ fontSize: 15, fontWeight: 800, letterSpacing: '0.1em', color: '#fff', lineHeight: 1.1 }}>
            LOUDDPAX
          </span>
          <span style={{ fontSize: 10, color: C.secondary, fontWeight: 600, letterSpacing: '0.06em' }}>
            North Las Vegas · Facility 7
          </span>
        </div>

        <span style={{ width: 1, height: 32, background: 'rgba(255,255,255,0.06)', margin: '0 8px' }} />

        {/* Facility State */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span style={{ fontSize: 8, color: C.muted, letterSpacing: '0.14em', fontWeight: 700, textTransform: 'uppercase' }}>
            Facility State
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <motion.div
              style={{ width: 6, height: 6, borderRadius: '50%', background: attentionCol }}
              animate={ctx.roomsNeedingAttention > 0 ? {
                boxShadow: [`0 0 0px ${attentionCol}`, `0 0 10px ${attentionCol}`, `0 0 0px ${attentionCol}`],
              } : {}}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            />
            <span style={{ fontSize: 12, color: ctx.roomsNeedingAttention > 0 ? attentionCol : C.neon, fontWeight: 800, letterSpacing: '0.02em' }}>
              {ctx.facilityStateLabel}
            </span>
          </div>
        </div>
      </div>

      {/* Center: Room Command Strip */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {system.rooms.map((room) => {
          const col = room.status === 'critical' ? C.red : room.status === 'warning' ? C.orange : C.neon
          const isActive = room.status !== 'optimal'
          return (
            <motion.button
              key={room.id}
              whileHover={{ scale: 1.03, y: -1 }}
              whileTap={{ scale: 0.98 }}
              style={{
                display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
                gap: 3,
                padding: '8px 14px',
                borderRadius: 8,
                background: isActive ? `${col}08` : 'rgba(255,255,255,0.02)',
                border: `1px solid ${isActive ? `${col}25` : C.border}`,
                cursor: 'pointer',
                minWidth: 110,
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              {isActive && (
                <motion.div
                  style={{
                    position: 'absolute', inset: 0,
                    background: `radial-gradient(circle at 50% 0%, ${col}10 0%, transparent 70%)`,
                  }}
                  animate={{ opacity: [0.3, 0.6, 0.3] }}
                  transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
                />
              )}
              <div style={{ position: 'relative', zIndex: 1, width: '100%' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                  <span style={{ fontSize: 10, fontWeight: 800, color: C.text, letterSpacing: '0.06em' }}>
                    {room.shortName}
                  </span>
                  <span style={{
                    fontSize: 7, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase',
                    color: col, padding: '1px 5px', borderRadius: 3,
                    background: `${col}12`, border: `1px solid ${col}20`,
                  }}>
                    {room.growStage}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
                  <span style={{ fontSize: 8, color: col, fontWeight: 700 }}>
                    {room.status === 'optimal' ? 'Stable' : room.status === 'warning' ? 'Drift' : 'Critical'}
                  </span>
                  <span style={{ fontSize: 7, color: C.muted }}>·</span>
                  <span style={{ fontSize: 8, color: C.secondary, fontVariantNumeric: 'tabular-nums' }}>
                    {(room.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 3 }}>
                  <span style={{
                    width: 4, height: 4, borderRadius: '50%',
                    background: room.cycle === 'day' ? C.neon : C.blue,
                    opacity: room.cycle === 'day' ? 1 : 0.5,
                  }} />
                  <span style={{ fontSize: 7, color: C.muted, fontWeight: 600, letterSpacing: '0.04em' }}>
                    {room.cycle === 'day' ? 'Lights on' : 'Night recovery'}
                  </span>
                </div>
              </div>
            </motion.button>
          )
        })}
      </div>

      {/* Right: Time, Latency, Live + Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
        {/* Facility summary micro-bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '6px 14px', borderRadius: 8, background: 'rgba(255,255,255,0.02)', border: `1px solid ${C.border}` }}>
          <MicroStat label="Coherence" value={`${(system.coherence * 100).toFixed(0)}%`} color={system.coherence > 0.6 ? C.neon : system.coherence > 0.3 ? C.orange : C.red} />
          <span style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.05)' }} />
          <MicroStat label="Alerts" value={`${ctx.roomsNeedingAttention}`} color={ctx.roomsNeedingAttention > 0 ? C.orange : C.neon} />
          <span style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.05)' }} />
          <MicroStat label="Changes" value={ctx.recentChange.includes('No') ? '—' : 'Active'} color={ctx.recentChange.includes('No') ? C.muted : C.blue} />
        </div>

        {onTogglePlay && onStep && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <button onClick={() => onStep(-1)} style={controlBtnStyle}>‹</button>
            <button
              onClick={onTogglePlay}
              style={{ ...controlBtnStyle, width: 32, height: 32, borderRadius: 8, color: isPlaying ? C.neon : C.text, borderColor: isPlaying ? `${C.neon}40` : C.border }}
            >
              {isPlaying ? '‖' : '▶'}
            </button>
            <button onClick={() => onStep(1)} style={controlBtnStyle}>›</button>
          </div>
        )}

        <span style={{ width: 1, height: 24, background: 'rgba(255,255,255,0.05)' }} />

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 1 }}>
          <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            Feed Latency
          </span>
          <span style={{ fontSize: 10, color: C.secondary, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>
            {ctx.facilityLatency.toFixed(1)}s
          </span>
        </div>

        <span style={{ fontSize: 11, color: C.secondary, fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
          {timeStr}
        </span>

        <motion.span
          style={{
            fontSize: 7, color: C.neon, letterSpacing: '0.18em', fontWeight: 800,
            border: `1px solid ${C.neon}25`, borderRadius: 4,
            padding: '4px 10px', background: `${C.neon}06`,
          }}
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
        >
          LIVE
        </motion.span>
      </div>
    </header>
  )
}

function MicroStat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 1 }}>
      <span style={{ fontSize: 7, color: C.muted, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</span>
      <span style={{ fontSize: 11, color, fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}

const controlBtnStyle: React.CSSProperties = {
  width: 28, height: 28, borderRadius: 7,
  background: '#0f1116', border: `1px solid ${C.border}`,
  color: '#6b7080', fontSize: 14, cursor: 'pointer',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
}
