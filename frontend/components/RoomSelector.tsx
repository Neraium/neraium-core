'use client'

import React from 'react'
import { LiveSystem } from '@/lib/simulation'

interface Props {
  system: LiveSystem
  selectedId: string | null
  onSelect: (id: string | null) => void
}

const C = {
  text: '#e8e9eb',
  muted: '#3a3f4a',
  neon: '#7cb342',
  orange: '#ff9e6d',
  red: '#e05c5c',
  surface: '#080a0d',
  surfaceHi: '#0f1116',
  border: 'rgba(255,255,255,0.04)',
}

export function RoomSelector({ system, selectedId, onSelect }: Props) {
  return (
    <div style={{
      display: 'flex', gap: 8,
      padding: '10px 20px',
      borderBottom: `1px solid ${C.border}`,
      background: C.surface,
      overflowX: 'auto',
    }}>
      {system.rooms.map(room => {
        const isSelected = selectedId === room.id
        const col = room.status === 'critical' ? C.red : room.status === 'warning' ? C.orange : C.neon

        return (
          <button
            key={room.id}
            onClick={() => onSelect(isSelected ? null : room.id)}
            style={{
              flexShrink: 0,
              display: 'flex', flexDirection: 'column', gap: 3,
              padding: '9px 14px',
              minWidth: 130,
              background: isSelected ? `${col}10` : C.surfaceHi,
              border: `1px solid ${isSelected ? `${col}35` : C.border}`,
              borderLeft: `2px solid ${col}`,
              borderRadius: 7,
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.25s ease',
              position: 'relative',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 10, fontWeight: 800, color: C.text, letterSpacing: '0.04em' }}>
                {room.shortName}
              </span>
              <span style={{ width: 5, height: 5, borderRadius: '50%', background: col, boxShadow: `0 0 5px ${col}` }} />
            </div>
            <span style={{ fontSize: 8, color: C.muted, fontWeight: 600, letterSpacing: '0.03em' }}>
              {room.cycle === 'day' ? 'Lights on' : 'Night recovery'}
            </span>
            <span style={{ fontSize: 8, fontWeight: 700, color: col, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              {room.status}
            </span>
          </button>
        )
      })}
    </div>
  )
}
