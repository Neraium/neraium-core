'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { LiveSystem, LiveRoom } from '@/lib/simulation'

interface Props {
  system: LiveSystem
  onSelectRoom?: (id: string) => void
  hoveredRoomId?: string | null
  onHoverRoom?: (id: string | null) => void
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

function sparkline(data: number[], w: number, h: number): string {
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  return data.map((v, i) => {
    const x = (i / Math.max(data.length - 1, 1)) * w
    const y = h - ((v - min) / range) * h
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

export function SubsystemPanel({ system, onSelectRoom, hoveredRoomId, onHoverRoom }: Props) {
  return (
    <div style={{
      padding: '0 20px 16px',
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
      flex: 1,
      overflowY: 'auto',
    }}>
      <div style={{
        fontSize: 8, letterSpacing: '0.18em', textTransform: 'uppercase',
        color: C.muted, fontWeight: 700, marginBottom: 4,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <span>Subsystem Behavior</span>
        <span style={{ fontSize: 7, color: C.secondary, letterSpacing: '0.06em' }}>
          {system.rooms.length} zones monitored
        </span>
      </div>

      {system.rooms.map((room, index) => (
        <SubsystemCard
          key={room.id}
          room={room}
          index={index}
          isHovered={hoveredRoomId === room.id}
          onHover={onHoverRoom}
          onClick={onSelectRoom}
        />
      ))}
    </div>
  )
}

function SubsystemCard({
  room,
  index,
  isHovered,
  onHover,
  onClick,
}: {
  room: LiveRoom
  index: number
  isHovered: boolean
  onHover?: (id: string | null) => void
  onClick?: (id: string) => void
}) {
  const col = room.status === 'critical' ? C.red : room.status === 'warning' ? C.orange : C.neon
  const hist = room.histories.temperature_f
  const spark = sparkline(hist.slice(-24), 60, 22)

  const stateLabel =
    room.driftContribution > 0.7 ? 'Critical Divergence'
    : room.driftContribution > 0.4 ? 'Instability Forming'
    : room.driftContribution > 0.2 ? 'Early Drift'
    : 'Stable'

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.05 }}
      onMouseEnter={() => onHover?.(room.id)}
      onMouseLeave={() => onHover?.(null)}
      onClick={() => onClick?.(room.id)}
      style={{
        position: 'relative',
        padding: '12px 14px',
        background: isHovered ? `linear-gradient(180deg, ${col}08 0%, ${C.surfaceHi} 100%)` : C.surfaceHi,
        borderRadius: 8,
        border: `1px solid ${isHovered ? `${col}30` : C.border}`,
        borderLeft: `2px solid ${col}`,
        overflow: 'hidden',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.3s ease',
        boxShadow: isHovered ? `0 4px 20px ${col}10` : 'none',
      }}
    >
      {/* Ambient gradient wash */}
      <div style={{
        position: 'absolute', left: 0, top: 0, bottom: 0, width: 60,
        background: `linear-gradient(90deg, ${col}06 0%, transparent 100%)`,
        pointerEvents: 'none',
      }} />

      <div style={{ position: 'relative', zIndex: 1 }}>
        {/* Top row */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 11, fontWeight: 800, color: C.text, letterSpacing: '0.04em' }}>
              {room.shortName}
            </span>
            <span style={{
              fontSize: 7, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase',
              color: col, padding: '2px 6px', borderRadius: 3,
              background: `${col}10`, border: `1px solid ${col}20`,
            }}>
              {stateLabel}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <svg width={60} height={22} style={{ opacity: 0.75 }}>
              <path d={spark} fill="none" stroke={col} strokeWidth={1.3} />
              <circle
                cx={60}
                cy={22 - ((hist[hist.length - 1] - Math.min(...hist)) / (Math.max(...hist) - Math.min(...hist) || 1)) * 22}
                r={2}
                fill={col}
              />
            </svg>
          </div>
        </div>

        {/* Behavior explanation */}
        <div style={{
          fontSize: 9, color: C.secondary, fontWeight: 500, lineHeight: 1.4, marginBottom: 8,
        }}>
          {room.behavioralState}
        </div>

        {/* Metrics row */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 14,
          paddingTop: 8,
          borderTop: `1px solid ${C.border}`,
        }}>
          <Metric label="Drift" value={`${(room.driftContribution * 100).toFixed(0)}%`} color={col} />
          <Metric label="Confidence" value={`${(room.confidence * 100).toFixed(0)}%`} color={C.neon} />
          <div style={{ flex: 1 }} />
          <TrendIndicator trend={room.trend} magnitude={room.trendMagnitude} />
          <span style={{
            fontSize: 7, color: C.muted, fontWeight: 600, letterSpacing: '0.04em',
            display: 'flex', alignItems: 'center', gap: 4,
          }}>
            <span style={{
              width: 4, height: 4, borderRadius: '50%',
              background: room.cycle === 'day' ? C.neon : C.blue,
              opacity: room.cycle === 'day' ? 1 : 0.5,
            }} />
            {room.cycle === 'day' ? 'Lights on' : 'Recovery'}
          </span>
        </div>
      </div>
    </motion.div>
  )
}

function Metric({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <span style={{ fontSize: 7, color: C.muted, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</span>
      <span style={{ fontSize: 10, color, fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}

function TrendIndicator({ trend, magnitude }: { trend: 'rising' | 'falling' | 'stable'; magnitude: number }) {
  const col = trend === 'rising' ? C.orange : trend === 'falling' ? C.blue : C.neon
  const arrow = trend === 'rising' ? '↑' : trend === 'falling' ? '↓' : '→'
  const label = trend === 'rising' ? 'Warming' : trend === 'falling' ? 'Cooling' : 'Stable'
  const intensity = magnitude > 1.5 ? 'strong' : magnitude > 0.5 ? 'moderate' : 'weak'

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ fontSize: 9, color: col, fontWeight: 800 }}>{arrow}</span>
      <span style={{ fontSize: 8, color: C.secondary, fontWeight: 600, letterSpacing: '0.04em' }}>
        {label} · {intensity}
      </span>
    </div>
  )
}
