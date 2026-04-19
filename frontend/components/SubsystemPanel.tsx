'use client'

import React from 'react'
import { DecisionUIState } from '@/lib/decisionToUI'
import { PALETTE, deriveSubsystems } from '@/lib/dashboardHelpers'

interface Props {
  state: DecisionUIState
}

export function SubsystemPanel({ state }: Props) {
  const subsystems = deriveSubsystems(state)

  return (
    <div style={{ padding: '0 16px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{
        fontSize: 9, letterSpacing: '0.14em', textTransform: 'uppercase',
        color: PALETTE.textMuted, fontWeight: 800, marginBottom: 2,
      }}>
        Subsystem Behavior
      </div>

      {subsystems.map(sub => {
        const col =
          sub.health === 'critical' ? PALETTE.red
          : sub.health === 'warning' ? PALETTE.orange
          : PALETTE.neon

        const glow =
          sub.health === 'critical' ? PALETTE.redGlow
          : sub.health === 'warning' ? PALETTE.orangeGlow
          : PALETTE.neonGlow

        return (
          <div key={sub.id} style={{
            position: 'relative',
            padding: '10px 14px',
            background: PALETTE.surfaceHighlight,
            borderRadius: 8,
            border: `1px solid ${PALETTE.border}`,
            borderLeft: `2px solid ${col}`,
            overflow: 'hidden',
            transition: 'all 0.4s ease',
          }}>
            {/* subtle glow on left */}
            <div style={{
              position: 'absolute', left: 0, top: 0, bottom: 0, width: 60,
              background: `linear-gradient(90deg, ${glow} 0%, transparent 100%)`,
              opacity: 0.15, pointerEvents: 'none',
            }} />

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 14, lineHeight: 1 }}>{sub.icon}</span>
                <span style={{ fontSize: 11, fontWeight: 700, color: PALETTE.textPrimary, letterSpacing: '0.04em' }}>
                  {sub.name}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{
                  fontSize: 9, fontWeight: 700, letterSpacing: '0.06em',
                  color: col, textTransform: 'uppercase',
                }}>
                  {sub.state}
                </span>
                <span style={{
                  fontSize: 8, fontWeight: 800, letterSpacing: '0.08em',
                  padding: '2px 7px', borderRadius: 4,
                  background: `${col}15`, color: col, border: `1px solid ${col}30`,
                }}>
                  {sub.contribution}
                </span>
              </div>
            </div>

            <div style={{ fontSize: 10, color: PALETTE.textSecondary, lineHeight: 1.4, fontWeight: 500 }}>
              {sub.explanation}
            </div>
          </div>
        )
      })}
    </div>
  )
}
