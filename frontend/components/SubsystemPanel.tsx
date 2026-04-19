'use client'

import React from 'react'
import { LiveSystem } from '@/lib/simulation'

interface Props {
  system: LiveSystem
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

export function SubsystemPanel({ system }: Props) {
  return (
    <div style={{ padding: '0 18px', display: 'flex', flexDirection: 'column', gap: 6, flex: 1, overflowY: 'auto' }}>
      <div style={{
        fontSize: 8, letterSpacing: '0.16em', textTransform: 'uppercase',
        color: C.muted, fontWeight: 700, marginBottom: 4,
      }}>
        Subsystem Behavior
      </div>

      {system.rooms.map(room => {
        const col = room.status === 'critical' ? C.red : room.status === 'warning' ? C.orange : C.neon
        const hist = room.histories.temperature_f
        const spark = sparkline(hist.slice(-20), 50, 18)

        return (
          <div key={room.id} style={{
            position: 'relative',
            padding: '10px 14px',
            background: C.surfaceHi,
            borderRadius: 8,
            border: `1px solid ${C.border}`,
            borderLeft: `1.5px solid ${col}`,
            overflow: 'hidden',
            transition: 'all 0.35s ease',
          }}>
            <div style={{
              position: 'absolute', left: 0, top: 0, bottom: 0, width: 50,
              background: `linear-gradient(90deg, ${col}08 0%, transparent 100%)`,
              pointerEvents: 'none',
            }} />

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 10, fontWeight: 800, color: C.text, letterSpacing: '0.04em' }}>
                  {room.shortName}
                </span>
                <span style={{ fontSize: 8, color: C.muted, fontWeight: 600, letterSpacing: '0.04em' }}>
                  {room.cycle === 'day' ? 'Lights on' : 'Recovery'}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <svg width={50} height={18} style={{ opacity: 0.7 }}>
                  <path d={spark} fill="none" stroke={col} strokeWidth={1.2} />
                </svg>
                <span style={{
                  fontSize: 8, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase',
                  color: col, padding: '2px 7px', borderRadius: 4,
                  background: `${col}10`, border: `1px solid ${col}20`,
                }}>
                  {room.status}
                </span>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <span style={{ fontSize: 9, color: C.secondary, fontWeight: 600 }}>
                  {room.behavioralState}
                </span>
                <span style={{ fontSize: 8, color: C.muted, fontWeight: 500 }}>
                  Drift {(room.driftContribution * 100).toFixed(0)}% · Conf {(room.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <span style={{
                fontSize: 8, color: room.trend === 'rising' ? C.orange : room.trend === 'falling' ? C.blue : C.neon,
                fontWeight: 700, letterSpacing: '0.06em',
              }}>
                {room.trend === 'rising' ? '↑ Warming' : room.trend === 'falling' ? '↓ Cooling' : '→ Stable'}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
