'use client'

import React from 'react'
import { LiveSystem } from '@/lib/simulation'

interface Props {
  system: LiveSystem
  roomId: string
  onClose: () => void
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

const SENSORS: Array<{ key: keyof LiveSystem['rooms'][0]['sensors']; label: string; unit: string; min: number; max: number }> = [
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

export function RoomDetail({ system, roomId, onClose }: Props) {
  const room = system.rooms.find(r => r.id === roomId)
  if (!room) return null

  const col = room.status === 'critical' ? C.red : room.status === 'warning' ? C.orange : C.neon

  return (
    <div style={{
      position: 'absolute', top: 0, right: 0, bottom: 0, width: 400,
      background: `linear-gradient(180deg, ${C.surface} 0%, #030405 100%)`,
      borderLeft: `1px solid ${C.border}`,
      boxShadow: '-12px 0 40px rgba(0,0,0,0.7)',
      zIndex: 100,
      display: 'flex', flexDirection: 'column',
      animation: 'rdIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards',
    }}>
      <style>{`@keyframes rdIn { from { transform: translateX(100%) } to { transform: translateX(0) } }`}</style>

      {/* header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '16px 20px', borderBottom: `1px solid ${C.border}`,
      }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 800, color: C.text }}>{room.name}</div>
          <div style={{ fontSize: 9, color: C.muted, marginTop: 2, fontWeight: 600 }}>
            {room.stage} · Floor {room.floor} · {room.cycle === 'day' ? 'Lights on' : 'Night recovery'}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontSize: 9, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase',
            color: col, padding: '3px 10px', borderRadius: 4,
            background: `${col}10`, border: `1px solid ${col}25`,
          }}>
            {room.status}
          </span>
          <button onClick={onClose} style={{
            width: 26, height: 26, borderRadius: 6, background: C.surfaceHi,
            border: `1px solid ${C.border}`, color: C.secondary,
            fontSize: 16, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>×</button>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {/* stats */}
        <div style={{ display: 'flex', gap: 12 }}>
          <div style={{ flex: 1, padding: '12px', background: C.surfaceHi, borderRadius: 8, border: `1px solid ${C.border}` }}>
            <div style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 4 }}>Plants</div>
            <div style={{ fontSize: 20, fontWeight: 900, color: C.text }}>{room.plantCount.toLocaleString()}</div>
            <div style={{ fontSize: 8, color: C.muted, marginTop: 2 }}>/ {room.capacity.toLocaleString()}</div>
          </div>
          <div style={{ flex: 1, padding: '12px', background: C.surfaceHi, borderRadius: 8, border: `1px solid ${C.border}` }}>
            <div style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 4 }}>Coherence</div>
            <div style={{ fontSize: 20, fontWeight: 900, color: room.confidence > 0.6 ? C.neon : room.confidence > 0.3 ? C.orange : C.red }}>
              {(room.confidence * 100).toFixed(0)}%
            </div>
            <div style={{ fontSize: 8, color: C.muted, marginTop: 2 }}>subsystem confidence</div>
          </div>
        </div>

        {/* behavioral state */}
        <div style={{
          padding: '10px 12px', background: `${col}08`, borderRadius: 6,
          border: `1px solid ${col}15`,
        }}>
          <div style={{ fontSize: 8, color: col, fontWeight: 800, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 3 }}>Behavioral State</div>
          <div style={{ fontSize: 11, color: C.text, fontWeight: 600 }}>{room.behavioralState}</div>
        </div>

        {/* sensor sparklines */}
        <div style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 2 }}>Sensor Telemetry</div>

        {SENSORS.map(meta => {
          const value = room.sensors[meta.key]
          const hist = room.histories[meta.key]
          const spark = sparkline(hist.slice(-24), 110, 26)
          const isHigh = value > meta.max * 0.92
          const isLow = value < meta.min * 1.08
          const valCol = isHigh || isLow ? C.red : C.text

          return (
            <div key={meta.key} style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '8px 12px', background: C.surfaceHi, borderRadius: 7, border: `1px solid ${C.border}`,
            }}>
              <div style={{ width: 70, flexShrink: 0 }}>
                <div style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase' }}>{meta.label}</div>
                <div style={{ fontSize: 13, fontWeight: 800, color: valCol, marginTop: 1 }}>{value}{meta.unit}</div>
              </div>
              <svg width={110} height={26} style={{ flexShrink: 0 }}>
                <path d={spark} fill="none" stroke={isHigh || isLow ? C.red : col} strokeWidth={1.3} opacity={0.75} />
                <circle cx={110} cy={26 - ((hist[hist.length - 1] - Math.min(...hist)) / (Math.max(...hist) - Math.min(...hist) || 1)) * 26} r={2} fill={isHigh || isLow ? C.red : col} />
              </svg>
              <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                <span style={{ fontSize: 8, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: isHigh || isLow ? C.red : C.muted }}>
                  {isHigh ? 'HIGH' : isLow ? 'LOW' : 'NOMINAL'}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
