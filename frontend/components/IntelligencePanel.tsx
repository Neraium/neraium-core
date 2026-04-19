'use client'

import React from 'react'
import { DecisionUIState } from '@/lib/decisionToUI'
import { PALETTE, deriveIntelligence, severityColor } from '@/lib/dashboardHelpers'

interface Props {
  state: DecisionUIState
}

export function IntelligencePanel({ state }: Props) {
  const intel = deriveIntelligence(state)
  const col = severityColor(intel.severity)

  return (
    <div style={{ padding: '0 16px 16px' }}>
      <div style={{
        fontSize: 9, letterSpacing: '0.14em', textTransform: 'uppercase',
        color: PALETTE.textMuted, fontWeight: 800, marginBottom: 10,
      }}>
        Operator Intelligence
      </div>

      <div style={{
        background: `linear-gradient(135deg, ${PALETTE.surfaceHighlight} 0%, ${PALETTE.surface} 100%)`,
        border: `1px solid ${PALETTE.border}`,
        borderRadius: 10,
        padding: '14px 16px',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* top accent line */}
        <div style={{
          position: 'absolute', top: 0, left: 16, right: 16, height: 2,
          background: `linear-gradient(90deg, transparent 0%, ${col} 50%, transparent 100%)`,
          opacity: 0.6,
        }} />

        {/* current state — big */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 9, color: PALETTE.textMuted, letterSpacing: '0.1em', fontWeight: 700, marginBottom: 4, textTransform: 'uppercase' }}>
            Current State
          </div>
          <div style={{
            fontSize: 22, fontWeight: 900, color: col, letterSpacing: '-0.01em',
            textShadow: `0 0 20px ${col}40`,
            lineHeight: 1.1,
          }}>
            {intel.currentState}
          </div>
        </div>

        {/* grid: onset + driver */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 14 }}>
          <div>
            <div style={{ fontSize: 9, color: PALETTE.textMuted, letterSpacing: '0.1em', fontWeight: 700, marginBottom: 3, textTransform: 'uppercase' }}>
              Onset
            </div>
            <div style={{ fontSize: 12, color: PALETTE.textPrimary, fontWeight: 700 }}>
              {intel.onset}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 9, color: PALETTE.textMuted, letterSpacing: '0.1em', fontWeight: 700, marginBottom: 3, textTransform: 'uppercase' }}>
              Primary Driver
            </div>
            <div style={{ fontSize: 12, color: col, fontWeight: 700 }}>
              {intel.primaryDriver}
            </div>
          </div>
        </div>

        {/* explanation */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 9, color: PALETTE.textMuted, letterSpacing: '0.1em', fontWeight: 700, marginBottom: 4, textTransform: 'uppercase' }}>
            Explanation
          </div>
          <div style={{ fontSize: 11, color: PALETTE.textSecondary, lineHeight: 1.5, fontWeight: 500 }}>
            {intel.explanation}
          </div>
        </div>

        {/* operator focus — highlighted */}
        <div style={{
          background: `${col}10`,
          border: `1px solid ${col}25`,
          borderRadius: 6,
          padding: '10px 12px',
        }}>
          <div style={{ fontSize: 9, color: col, letterSpacing: '0.1em', fontWeight: 800, marginBottom: 3, textTransform: 'uppercase' }}>
            Operator Focus
          </div>
          <div style={{ fontSize: 11, color: PALETTE.textPrimary, fontWeight: 600, lineHeight: 1.4 }}>
            {intel.operatorFocus}
          </div>
        </div>
      </div>
    </div>
  )
}
