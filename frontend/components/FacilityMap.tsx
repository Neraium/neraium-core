'use client'

import React, { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { LiveSystem, LiveRoom } from '@/lib/simulation'
import { computeFacilityContext, roomStateLabel } from '@/lib/facilityContext'

interface Props {
  system: LiveSystem
  onSelectRoom: (id: string) => void
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
  surfaceHi: '#0f1116',
  border: 'rgba(255,255,255,0.04)',
}

const ROOM_LAYOUT: Array<{
  id: string
  left: number
  top: number
  floor: number
}> = [
  { id: 'veg-a',  left: 12, top: 58, floor: 1 },
  { id: 'veg-b',  left: 12, top: 22, floor: 1 },
  { id: 'flow-a', left: 68, top: 18, floor: 2 },
  { id: 'flow-b', left: 68, top: 50, floor: 2 },
  { id: 'flow-c', left: 68, top: 78, floor: 2 },
]

function healthColor(status: LiveRoom['status']) {
  return status === 'critical' ? C.red : status === 'warning' ? C.orange : C.neon
}

export function FacilityMap({ system, onSelectRoom }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [tick, setTick] = useState(0)
  const lastRef = useRef(0)

  useEffect(() => {
    let raf: number
    const loop = (t: number) => {
      if (t - lastRef.current > 80) {
        lastRef.current = t
        setTick(x => x + 1)
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  const time = tick * 0.03
  const ctx = computeFacilityContext(system)

  // Core position (center)
  const coreX = 50 // %
  const coreY = 50 // %

  return (
    <div ref={containerRef} style={{
      position: 'relative',
      width: '100%', height: '100%',
      background: `radial-gradient(ellipse at 50% 50%, ${C.neon}06 0%, transparent 60%)`,
      overflow: 'hidden',
    }}>
      {/* SVG Blueprint Layer */}
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        <defs>
          <filter id="fm-glow-neon" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="fm-glow-red" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>

        {/* Grid */}
        <g opacity={0.06}>
          {Array.from({ length: 12 }, (_, i) => (
            <line key={`v${i}`} x1={(i + 1) * 7.69} y1={0} x2={(i + 1) * 7.69} y2={100} stroke={C.muted} strokeWidth={0.5} />
          ))}
          {Array.from({ length: 8 }, (_, i) => (
            <line key={`h${i}`} x1={0} y1={(i + 1) * 12.5} x2={100} y2={(i + 1) * 12.5} stroke={C.muted} strokeWidth={0.5} />
          ))}
        </g>

        {/* Floor labels */}
        <text x={6} y={50} fill={C.muted} fontSize={3} fontWeight={700} letterSpacing="0.14em" textAnchor="middle" transform="rotate(-90, 6, 50)">
          FLOOR 1 — VEG
        </text>
        <text x={94} y={50} fill={C.muted} fontSize={3} fontWeight={700} letterSpacing="0.14em" textAnchor="middle" transform="rotate(90, 94, 50)">
          FLOOR 2 — FLOWER
        </text>

        {/* Connection lines from core to rooms */}
        {ROOM_LAYOUT.map(layout => {
          const room = system.rooms.find(r => r.id === layout.id)
          if (!room) return null
          const col = healthColor(room.status)
          const isCritical = room.status === 'critical'
          const x1 = coreX
          const y1 = coreY
          const x2 = layout.left
          const y2 = layout.top
          // Animated dash offset for flow
          const dashOffset = isCritical ? time * 20 : time * 8
          return (
            <g key={layout.id}>
              <line x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={col} strokeWidth={0.3} opacity={0.15}
                strokeDasharray={isCritical ? '1 1' : '2 1'}
                strokeDashoffset={-dashOffset}
                style={{ transition: 'stroke 0.8s ease' }}
              />
              <line x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={col} strokeWidth={0.8} opacity={0.05} filter={isCritical ? 'url(#fm-glow-red)' : 'url(#fm-glow-neon)'}
              />
            </g>
          )
        })}

        {/* Core */}
        <g transform={`translate(${coreX}, ${coreY})`}>
          <circle r={8} fill="none" stroke={ctx.coherenceTrend === 'weakening' ? C.orange : C.neon} strokeWidth={0.3} opacity={0.25} />
          <circle r={5} fill={`${ctx.coherenceTrend === 'weakening' ? C.orange : C.neon}10`} stroke={ctx.coherenceTrend === 'weakening' ? C.orange : C.neon} strokeWidth={0.8} />
          <text textAnchor="middle" dominantBaseline="central" fill={ctx.coherenceTrend === 'weakening' ? C.orange : C.neon} fontSize={3} fontWeight={800} letterSpacing="0.1em">
            CORE
          </text>
        </g>
      </svg>

      {/* Room Modules */}
      {ROOM_LAYOUT.map(layout => {
        const room = system.rooms.find(r => r.id === layout.id)
        if (!room) return null
        return (
          <RoomModule
            key={room.id}
            room={room}
            left={`${layout.left}%`}
            top={`${layout.top}%`}
            onClick={() => onSelectRoom(room.id)}
          />
        )
      })}
    </div>
  )
}

function RoomModule({ room, left, top, onClick }: {
  room: LiveRoom
  left: string
  top: string
  onClick: () => void
}) {
  const col = healthColor(room.status)
  const label = roomStateLabel(room)
  const isCritical = room.status === 'critical'
  const isWarning = room.status === 'warning'

  return (
    <div style={{
      position: 'absolute',
      left, top,
      transform: 'translate(-50%, -50%)',
    }}>
      <motion.button
        onClick={onClick}
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        whileHover={{ scale: 1.04 }}
        whileTap={{ scale: 0.98 }}
        transition={{ type: 'spring', stiffness: 300, damping: 25 }}
        style={{
          width: 200,
          background: `linear-gradient(180deg, ${C.surfaceHi} 0%, ${C.surface} 100%)`,
          border: `1px solid ${C.border}`,
          borderLeft: `2px solid ${col}`,
          borderRadius: 10,
          padding: '14px 16px',
          cursor: 'pointer',
          textAlign: 'left',
          boxShadow: isCritical ? `0 0 24px ${col}15` : '0 8px 24px rgba(0,0,0,0.4)',
          overflow: 'hidden',
        }}
      >
      {/* Ambient pulse for critical / warning */}
      {(isCritical || isWarning) && (
        <motion.div
          style={{
            position: 'absolute', inset: 0,
            background: `radial-gradient(circle at 30% 30%, ${col}08 0%, transparent 70%)`,
          }}
          animate={{ opacity: [0.4, 0.8, 0.4] }}
          transition={{ duration: isCritical ? 1.2 : 2.5, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}

      <div style={{ position: 'relative', zIndex: 1 }}>
        {/* Top row: name + status dot */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 800, color: C.text, letterSpacing: '0.04em' }}>
              {room.shortName}
            </div>
            <div style={{ fontSize: 8, color: C.muted, fontWeight: 600, marginTop: 1 }}>
              {room.stage}
            </div>
          </div>
          <motion.div
            style={{ width: 6, height: 6, borderRadius: '50%', background: col }}
            animate={isCritical ? { boxShadow: [`0 0 0px ${col}`, `0 0 10px ${col}`, `0 0 0px ${col}`] } : {}}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
          />
        </div>

        {/* State label */}
        <div style={{
          display: 'inline-block',
          fontSize: 8, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase',
          color: col, padding: '3px 8px', borderRadius: 4,
          background: `${col}10`, border: `1px solid ${col}20`,
          marginBottom: 10,
        }}>
          {label}
        </div>

        {/* Drift & Confidence mini bars */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <MiniBar label="Drift" value={room.driftContribution} color={col} />
          <MiniBar label="Confidence" value={room.confidence} color={C.neon} />
        </div>

        {/* Bottom meta */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginTop: 10, paddingTop: 8,
          borderTop: `1px solid ${C.border}`,
        }}>
          <span style={{ fontSize: 8, color: C.muted, fontWeight: 600 }}>
            {room.cycle === 'day' ? 'Lights on' : 'Night recovery'}
          </span>
          <span style={{ fontSize: 8, color: C.secondary, fontWeight: 700 }}>
            {room.plantCount}/{room.capacity}
          </span>
        </div>
      </div>
    </motion.button>
    </div>
  )
}

function MiniBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
        <span style={{ fontSize: 7, color: C.muted, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</span>
        <span style={{ fontSize: 8, color: C.text, fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>
          {Math.round(value * 100)}%
        </span>
      </div>
      <div style={{ height: 3, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
        <motion.div
          style={{ height: '100%', background: color, borderRadius: 2 }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(1, Math.max(0, value)) * 100}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        />
      </div>
    </div>
  )
}
