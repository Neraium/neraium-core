'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { LiveSystem, LiveRoom, LiveSensors } from '@/lib/simulation'
import { roomStateLabel } from '@/lib/facilityContext'
import { IntelligenceRail } from './IntelligenceRail'
import { NarrativeTimeline } from './NarrativeTimeline'

interface Props {
  system: LiveSystem
  roomId: string
  onBack: () => void
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

const SENSORS: Array<{
  key: keyof LiveSensors
  label: string
  unit: string
  min: number
  max: number
}> = [
  { key: 'temperature_f', label: 'Temperature', unit: '°F', min: 65, max: 92 },
  { key: 'humidity_rh', label: 'Humidity', unit: '%', min: 30, max: 80 },
  { key: 'co2_ppm', label: 'CO₂', unit: 'ppm', min: 300, max: 2000 },
  { key: 'vpd_kpa', label: 'VPD', unit: 'kPa', min: 0.3, max: 3.2 },
  { key: 'ph', label: 'pH', unit: '', min: 5.0, max: 7.5 },
  { key: 'ec_ms', label: 'EC', unit: 'mS', min: 0.5, max: 3.5 },
  { key: 'ppfd_umol', label: 'PPFD', unit: 'µmol', min: 0, max: 900 },
  { key: 'irrigation_ml', label: 'Irrigation', unit: 'mL', min: 0, max: 400 },
]

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

function sensorDeviation(sensor: keyof LiveSensors, value: number): number {
  const meta = SENSORS.find(s => s.key === sensor)
  if (!meta) return 0
  const mid = (meta.min + meta.max) / 2
  const range = meta.max - meta.min
  return Math.abs(value - mid) / (range / 2)
}

function getContributors(room: LiveRoom) {
  const list = SENSORS.map(meta => {
    const val = room.sensors[meta.key]
    const dev = sensorDeviation(meta.key, val)
    const isHigh = val > meta.max * 0.92
    const isLow = val < meta.min * 1.08
    const status: 'critical' | 'warning' | 'optimal' = isHigh || isLow ? 'critical' : dev > 0.5 ? 'warning' : 'optimal'
    return {
      name: meta.label,
      percentage: Math.min(100, Math.round(dev * 100)),
      status,
    }
  })
  return list.sort((a, b) => b.percentage - a.percentage).slice(0, 3)
}

function roomExplanation(room: LiveRoom, system: LiveSystem): string {
  const label = roomStateLabel(room)
  if (room.status === 'critical') {
    return `${room.name} is in critical divergence. ${room.behavioralState}. Immediate intervention required to prevent cascade failure.`
  }
  if (room.status === 'warning') {
    return `${room.name} shows ${label.toLowerCase()}. ${room.behavioralState}. Monitor closely and verify climate setpoints.`
  }
  return `${room.name} is stable. ${room.behavioralState}. Subsystems operating within nominal parameters.`
}

function cycleModeLabel(room: LiveRoom): string {
  if (room.sensors.irrigation_ml > 0) return 'Post-irrigation'
  if (room.cycle === 'day') {
    if (room.cycleProgress < 0.2) return 'Lights ramping'
    if (room.cycleProgress > 0.8) return 'Late flower / wind-down'
    return 'Lights on'
  }
  return 'Night recovery'
}

export function RoomDetailView({ system, roomId, onBack }: Props) {
  const room = system.rooms.find(r => r.id === roomId)
  if (!room) return null

  const col = room.status === 'critical' ? C.red : room.status === 'warning' ? C.orange : C.neon
  const label = roomStateLabel(room)

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      width: '100%', height: '100%',
      background: '#030405',
      color: C.text,
      fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',
      overflow: 'hidden',
    }}>
      {/* ═══ TOP BAR ═══════════════════════════════════════════════════════ */}
      <div style={{
        flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 64, padding: '0 24px',
        borderBottom: `1px solid ${C.border}`,
        background: 'linear-gradient(180deg, rgba(5,6,8,0.98) 0%, rgba(3,4,5,0.98) 100%)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <motion.button
            onClick={onBack}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{
              width: 32, height: 32, borderRadius: 8,
              background: C.surfaceHi, border: `1px solid ${C.border}`,
              color: C.secondary, fontSize: 16, cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >
            ←
          </motion.button>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 10, color: C.muted, fontWeight: 700, letterSpacing: '0.08em' }}>FACILITY</span>
            <span style={{ fontSize: 10, color: C.muted }}>/</span>
            <span style={{ fontSize: 13, fontWeight: 800, color: C.text, letterSpacing: '0.04em' }}>{room.shortName}</span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <span style={{
            fontSize: 9, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase',
            color: col, padding: '4px 12px', borderRadius: 4,
            background: `${col}10`, border: `1px solid ${col}25`,
          }}>
            {label}
          </span>
          <span style={{ fontSize: 9, color: C.secondary, fontWeight: 600, letterSpacing: '0.04em' }}>
            {cycleModeLabel(room)}
          </span>
        </div>
      </div>

      {/* ═══ HERO ══════════════════════════════════════════════════════════ */}
      <div style={{
        flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '22px 28px',
        borderBottom: `1px solid ${C.border}`,
        background: `linear-gradient(90deg, ${col}06 0%, transparent 60%)`,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          {/* Status Ring */}
          <div style={{
            width: 72, height: 72, borderRadius: '50%',
            border: `2px solid ${col}`,
            boxShadow: `0 0 24px ${col}18`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            position: 'relative',
          }}>
            <motion.div
              style={{ position: 'absolute', inset: -5, borderRadius: '50%', border: `1px solid ${col}` }}
              animate={{ opacity: [0.15, 0.4, 0.15], scale: [1, 1.06, 1] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
            />
            <span style={{ fontSize: 10, fontWeight: 800, color: col, letterSpacing: '0.06em' }}>
              {room.status.toUpperCase()}
            </span>
          </div>

          <div>
            <div style={{ fontSize: 22, fontWeight: 900, color: C.text, letterSpacing: '-0.02em' }}>
              {room.name}
            </div>
            <div style={{ fontSize: 10, color: C.muted, fontWeight: 600, marginTop: 3, display: 'flex', alignItems: 'center', gap: 10 }}>
              <span>{room.stage}</span>
              <span style={{ width: 3, height: 3, borderRadius: '50%', background: C.muted }} />
              <span>Floor {room.floor}</span>
              <span style={{ width: 3, height: 3, borderRadius: '50%', background: C.muted }} />
              <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <span style={{
                  width: 5, height: 5, borderRadius: '50%',
                  background: room.cycle === 'day' ? C.neon : C.blue,
                  opacity: room.cycle === 'day' ? 1 : 0.5,
                }} />
                {cycleModeLabel(room)}
              </span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <HeroStat label="Plants" value={`${room.plantCount}`} sub={`/ ${room.capacity}`} color={C.text} />
          <HeroStat label="Coherence" value={`${(room.confidence * 100).toFixed(0)}%`} color={room.confidence > 0.6 ? C.neon : room.confidence > 0.3 ? C.orange : C.red} />
          <HeroStat label="Drift" value={`${(room.driftContribution * 100).toFixed(0)}%`} color={col} />
          <HeroStat label="Trajectory" value={system.trajectory} color={system.trajectory === 'Degrading' ? C.orange : system.trajectory === 'Uncertain' ? C.blue : C.neon} />
        </div>
      </div>

      {/* ═══ MAIN ══════════════════════════════════════════════════════════ */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        {/* LEFT — Intelligence Rail */}
        <div style={{
          flex: '0 0 340px',
          borderRight: `1px solid ${C.border}`,
          overflowY: 'auto',
          background: C.surface,
        }}>
          <IntelligenceRail
            stateLabel={label}
            driftScore={room.driftContribution}
            confidence={room.confidence}
            driftOnsetTime={system.onsetMinutes ? `~${system.onsetMinutes} min ago` : undefined}
            topContributors={getContributors(room)}
            explanation={roomExplanation(room, system)}
          />
        </div>

        {/* RIGHT — Sensor Grid */}
        <div style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: 12,
          padding: '18px 22px',
          overflowY: 'auto',
          background: '#030405',
        }}>
          {SENSORS.map((meta, sIdx) => {
            const value = room.sensors[meta.key]
            const hist = room.histories[meta.key]
            const spark = sparkline(hist.slice(-30), 200, 44)
            const isHigh = value > meta.max * 0.92
            const isLow = value < meta.min * 1.08
            const valCol = isHigh || isLow ? C.red : C.text
            const statusLabel = isHigh ? 'HIGH' : isLow ? 'LOW' : 'NOMINAL'
            const statusColor = isHigh || isLow ? C.red : C.neon

            return (
              <motion.div
                key={meta.key}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: sIdx * 0.04 }}
                style={{
                  background: `linear-gradient(180deg, ${C.surfaceHi} 0%, ${C.surface} 100%)`,
                  border: `1px solid ${C.border}`,
                  borderLeft: `2px solid ${statusColor}`,
                  borderRadius: 8,
                  padding: '16px 18px',
                  display: 'flex', flexDirection: 'column', gap: 10,
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <div style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0, width: 50,
                  background: `linear-gradient(90deg, ${statusColor}06 0%, transparent 100%)`,
                  pointerEvents: 'none',
                }} />
                <div style={{ position: 'relative', zIndex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                      {meta.label}
                    </span>
                    <span style={{
                      fontSize: 8, fontWeight: 800, letterSpacing: '0.06em', textTransform: 'uppercase',
                      color: statusColor, padding: '2px 8px', borderRadius: 4,
                      background: `${statusColor}10`, border: `1px solid ${statusColor}20`,
                    }}>
                      {statusLabel}
                    </span>
                  </div>

                  <div style={{ fontSize: 24, fontWeight: 900, color: valCol, fontVariantNumeric: 'tabular-nums' }}>
                    {typeof value === 'number' ? value.toFixed(1) : value}{meta.unit}
                  </div>

                  <svg width={200} height={44} style={{ opacity: 0.85 }}>
                    <path d={spark} fill="none" stroke={isHigh || isLow ? C.red : col} strokeWidth={1.4} opacity={0.7} />
                    <circle
                      cx={200}
                      cy={44 - ((hist[hist.length - 1] - Math.min(...hist)) / (Math.max(...hist) - Math.min(...hist) || 1)) * 44}
                      r={3}
                      fill={isHigh || isLow ? C.red : col}
                    />
                  </svg>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: C.muted, fontWeight: 600 }}>
                    <span>Min {Math.min(...hist).toFixed(1)}</span>
                    <span>Max {Math.max(...hist).toFixed(1)}</span>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* ═══ BOTTOM — Path + Timeline ═════════════════════════════════════ */}
      <div style={{ flexShrink: 0, borderTop: `1px solid ${C.border}`, background: C.surface, padding: '12px 24px' }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 14,
          marginBottom: 10,
        }}>
          <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            Projected Path
          </span>
          <span style={{ fontSize: 11, color: C.text, fontWeight: 600 }}>
            {system.pathOutlook}
          </span>
        </div>
        <NarrativeTimeline system={system} />
      </div>
    </div>
  )
}

function HeroStat({ label, value, sub, color }: { label: string; value: string; sub?: string; color: string }) {
  return (
    <div style={{ textAlign: 'right' }}>
      <div style={{ fontSize: 7, color: C.muted, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 3 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 900, color, fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 8, color: C.muted, fontWeight: 600, marginTop: 1 }}>
          {sub}
        </div>
      )}
    </div>
  )
}
