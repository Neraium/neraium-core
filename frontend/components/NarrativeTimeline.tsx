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
  border: 'rgba(255,255,255,0.04)',
}

const STAGES = [
  { key: 'baseline', label: 'Baseline' },
  { key: 'early_shift', label: 'Drift' },
  { key: 'emerging', label: 'Emerging' },
  { key: 'persistent', label: 'Persistent' },
  { key: 'accelerated', label: 'Accelerated' },
  { key: 'failure_approach', label: 'Critical' },
] as const

export function NarrativeTimeline({ system }: Props) {
  const currentIdx = STAGES.findIndex(s => s.key === system.stage)
  const total = STAGES.length
  const segWidth = 100 / total

  const narrativeProgress = (system.scenarioIdx + system.scenarioProgress) / 6

  return (
    <div style={{ padding: '10px 22px 14px' }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 10,
      }}>
        <span style={{
          fontSize: 8, color: C.muted, textTransform: 'uppercase',
          letterSpacing: '0.16em', fontWeight: 700,
        }}>
          State Evolution
        </span>
        <span style={{ fontSize: 8, color: C.secondary, letterSpacing: '0.08em', fontWeight: 600 }}>
          {STAGES[currentIdx]?.label} · Drift {(system.structuralDrift * 100).toFixed(0)}%
        </span>
      </div>

      {/* track */}
      <div style={{
        position: 'relative',
        height: 32,
        borderRadius: 6,
        overflow: 'hidden',
        background: C.surface,
        border: `1px solid ${C.border}`,
      }}>
        {/* zone fills */}
        {STAGES.map((s, i) => {
          const col =
            i <= 1 ? C.neon
            : i <= 3 ? C.blue
            : i <= 4 ? C.orange
            : C.red
          return (
            <div key={s.key} style={{
              position: 'absolute', left: `${i * segWidth}%`, top: 0, bottom: 0,
              width: `${segWidth}%`,
              background: i <= currentIdx ? `${col}10` : 'transparent',
              borderRight: i < total - 1 ? `1px solid ${C.border}` : 'none',
              transition: 'background 0.8s ease',
            }} />
          )
        })}

        {/* narrative trail */}
        <div style={{
          position: 'absolute', left: 0, top: 0, bottom: 0,
          width: `${narrativeProgress * 100}%`,
          background: `linear-gradient(90deg, ${C.neon}08 0%, ${currentIdx >= 3 ? C.orange : C.blue}12 100%)`,
          transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)',
          pointerEvents: 'none',
        }} />

        {/* labels */}
        {STAGES.map((s, i) => (
          <div key={s.key} style={{
            position: 'absolute', left: `${i * segWidth}%`, top: 0, bottom: 0,
            width: `${segWidth}%`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{
              fontSize: 8, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
              color: i === currentIdx ? C.text
                : i < currentIdx ? C.secondary
                : `${C.muted}50`,
              transition: 'color 0.5s ease',
            }}>
              {s.label}
            </span>
          </div>
        ))}

        {/* current indicator */}
        <div style={{
          position: 'absolute',
          left: `${(currentIdx + 0.5) * segWidth}%`,
          top: '50%',
          transform: 'translate(-50%, -50%)',
          width: 8, height: 8,
          borderRadius: '50%',
          background: C.text,
          boxShadow: `0 0 10px ${C.text}40`,
          zIndex: 2,
          transition: 'left 1s cubic-bezier(0.4,0,0.2,1)',
        }} />
      </div>

      {/* event markers */}
      <div style={{ display: 'flex', gap: 16, marginTop: 7, paddingLeft: 2 }}>
        {currentIdx >= 1 && (
          <span style={{ fontSize: 7, color: C.blue, fontWeight: 700, letterSpacing: '0.06em' }}>
            ● Drift emerged
          </span>
        )}
        {currentIdx >= 3 && (
          <span style={{ fontSize: 7, color: C.orange, fontWeight: 700, letterSpacing: '0.06em' }}>
            ● Coherence break
          </span>
        )}
        {currentIdx >= 5 && (
          <span style={{ fontSize: 7, color: C.red, fontWeight: 700, letterSpacing: '0.06em' }}>
            ● Critical regime
          </span>
        )}
      </div>
    </div>
  )
}
