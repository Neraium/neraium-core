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
  secondary: '#6b7080',
  muted: '#3a3f4a',
  neon: '#7cb342',
  blue: '#5c9dff',
  orange: '#ff9e6d',
  red: '#e05c5c',
  surface: '#080a0d',
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
      height: 64, padding: '0 20px',
      borderBottom: `1px solid ${C.border}`,
      background: 'rgba(3,4,5,0.98)',
    }}>
      {/* Brand & Facility */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <img src="/louddpax-logo.jpg" alt="Louddpax" style={{ height: 42, width: 42, objectFit: 'cover', borderRadius: 8 }} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <span style={{ fontSize: 14, fontWeight: 800, letterSpacing: '0.08em', color: '#fff' }}>
            LOUDDPAX
          </span>
          <span style={{ fontSize: 10, color: C.text, fontWeight: 600, letterSpacing: '0.04em' }}>
            North Las Vegas
          </span>
        </div>
        <span style={{ width: 1, height: 26, background: 'rgba(255,255,255,0.05)', margin: '0 4px' }} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <span style={{ fontSize: 8, color: C.muted, letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase' }}>
            Facility State
          </span>
          <span style={{ fontSize: 11, color: ctx.roomsNeedingAttention > 0 ? attentionCol : C.neon, fontWeight: 700 }}>
            {ctx.facilityStateLabel}
          </span>
        </div>
      </div>

      {/* Center: Metrics + Playback controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
        <MetricPill label="Active Rooms" value={`${ctx.activeRoomCount}`} color={C.text} />
        <MetricPill
          label="Need Attention"
          value={`${ctx.roomsNeedingAttention}`}
          color={attentionCol}
          pulse={ctx.roomsNeedingAttention > 0}
        />
        <MetricPill label="Cycle" value={ctx.cycleContext} color={C.blue} />

        {onTogglePlay && onStep && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 8 }}>
            <button onClick={() => onStep(-1)} style={controlBtnStyle}>‹</button>
            <button
              onClick={onTogglePlay}
              style={{ ...controlBtnStyle, width: 30, height: 30, borderRadius: 7, color: isPlaying ? C.neon : C.text, borderColor: isPlaying ? `${C.neon}40` : C.border }}
            >
              {isPlaying ? '‖' : '▶'}
            </button>
            <button onClick={() => onStep(1)} style={controlBtnStyle}>›</button>
          </div>
        )}
      </div>

      {/* Right: Time, Latency, Live */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 1 }}>
          <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            Feed Latency
          </span>
          <span style={{ fontSize: 10, color: C.secondary, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>
            {ctx.facilityLatency.toFixed(1)}s
          </span>
        </div>
        <span style={{ width: 1, height: 18, background: 'rgba(255,255,255,0.05)' }} />
        <span style={{ fontSize: 10, color: C.secondary, fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
          {timeStr}
        </span>
        <motion.span
          style={{
            fontSize: 7, color: C.neon, letterSpacing: '0.16em', fontWeight: 800,
            border: `1px solid ${C.neon}25`, borderRadius: 4,
            padding: '3px 8px', background: `${C.neon}06`,
          }}
          animate={{ opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        >
          LIVE
        </motion.span>
      </div>
    </header>
  )
}

function MetricPill({ label, value, color, pulse = false }: { label: string; value: string; color: string; pulse?: boolean }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
      minWidth: 80,
    }}>
      <span style={{ fontSize: 7, color: C.muted, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
        {label}
      </span>
      <span style={{
        fontSize: 12, color, fontWeight: 800, letterSpacing: '0.02em',
        textShadow: pulse ? `0 0 10px ${color}40` : 'none',
      }}>
        {value}
      </span>
      {pulse && (
        <motion.div
          style={{ width: 4, height: 4, borderRadius: '50%', background: color, marginTop: 2 }}
          animate={{ opacity: [0.3, 1, 0.3], scale: [1, 1.2, 1] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}
    </div>
  )
}

const controlBtnStyle: React.CSSProperties = {
  width: 26, height: 26, borderRadius: 6,
  background: '#0f1116', border: `1px solid ${C.border}`,
  color: '#6b7080', fontSize: 13, cursor: 'pointer',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
}
