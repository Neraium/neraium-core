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

function severityCol(sev: string) {
  return sev === 'HIGH' ? C.red : sev === 'ELEVATED' ? C.orange : sev === 'MODERATE' ? C.blue : C.neon
}

export function IntelligencePanel({ system }: Props) {
  const col = severityCol(system.severity)

  return (
    <div style={{ padding: '0 18px 16px' }}>
      <div style={{
        fontSize: 8, letterSpacing: '0.16em', textTransform: 'uppercase',
        color: C.muted, fontWeight: 700, marginBottom: 12,
      }}>
        Operator Intelligence
      </div>

      <div style={{
        background: `linear-gradient(180deg, #0d1014 0%, ${C.surface} 100%)`,
        border: `1px solid ${C.border}`,
        borderRadius: 10,
        padding: '16px 18px',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* top accent */}
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: 1.5,
          background: `linear-gradient(90deg, transparent 5%, ${col} 50%, transparent 95%)`,
          opacity: 0.5,
        }} />

        {/* current state */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 8, color: C.muted, letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase', marginBottom: 5 }}>
            Current State
          </div>
          <div style={{
            fontSize: 24, fontWeight: 900, color: col, letterSpacing: '-0.02em',
            lineHeight: 1.1,
            textShadow: `0 0 24px ${col}30`,
          }}>
            {system.stage.replace(/_/g, ' ')}
          </div>
        </div>

        {/* grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 8, color: C.muted, letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>
              Onset
            </div>
            <div style={{ fontSize: 12, color: C.text, fontWeight: 700 }}>
              {system.onsetMinutes ? `~${system.onsetMinutes} min ago` : 'No deviation'}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 8, color: C.muted, letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>
              Coherence
            </div>
            <div style={{ fontSize: 12, color: system.coherence > 0.6 ? C.neon : system.coherence > 0.3 ? C.orange : C.red, fontWeight: 700 }}>
              {(system.coherence * 100).toFixed(0)}%
            </div>
          </div>
          <div>
            <div style={{ fontSize: 8, color: C.muted, letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>
              Primary Driver
            </div>
            <div style={{ fontSize: 12, color: col, fontWeight: 700 }}>
              {system.primaryDriver}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 8, color: C.muted, letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>
              Confidence
            </div>
            <div style={{ fontSize: 12, color: C.text, fontWeight: 700 }}>
              {system.confidenceLabel}
            </div>
          </div>
        </div>

        {/* explanation */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 8, color: C.muted, letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase', marginBottom: 5 }}>
            Explanation
          </div>
          <div style={{ fontSize: 11, color: C.secondary, lineHeight: 1.55, fontWeight: 500 }}>
            {system.explanation}
          </div>
        </div>

        {/* operator focus */}
        <div style={{
          background: `${col}08`,
          border: `1px solid ${col}18`,
          borderRadius: 6,
          padding: '10px 12px',
        }}>
          <div style={{ fontSize: 8, color: col, letterSpacing: '0.12em', fontWeight: 800, textTransform: 'uppercase', marginBottom: 4 }}>
            Operator Focus
          </div>
          <div style={{ fontSize: 11, color: C.text, fontWeight: 600, lineHeight: 1.45 }}>
            {system.operatorFocus}
          </div>
        </div>

        {/* path outlook */}
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${C.border}` }}>
          <div style={{ fontSize: 8, color: C.muted, letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase', marginBottom: 4 }}>
            Path Outlook
          </div>
          <div style={{ fontSize: 10, color: C.secondary, fontWeight: 500, lineHeight: 1.4 }}>
            {system.pathOutlook}
          </div>
        </div>
      </div>
    </div>
  )
}
