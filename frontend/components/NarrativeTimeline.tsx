'use client'

import React from 'react'
import { DecisionUIState, DegradationStage } from '@/lib/decisionToUI'
import { PALETTE, stageIndex } from '@/lib/dashboardHelpers'

interface Props {
  state: DecisionUIState
}

const STAGES = [
  { stage: DegradationStage.BASELINE,        label: 'Baseline',        zone: 'baseline' },
  { stage: DegradationStage.EARLY_SHIFT,     label: 'Drift',           zone: 'drift' },
  { stage: DegradationStage.EMERGING,        label: 'Emerging',        zone: 'drift' },
  { stage: DegradationStage.PERSISTENT,      label: 'Persistent',      zone: 'instability' },
  { stage: DegradationStage.ACCELERATED,     label: 'Accelerated',     zone: 'escalation' },
  { stage: DegradationStage.FAILURE_APPROACH,label: 'Critical',        zone: 'critical' },
] as const

const ZONE_COLORS: Record<string, { bg: string; glow: string }> = {
  baseline:   { bg: `${PALETTE.neon}10`,  glow: `${PALETTE.neon}20` },
  drift:      { bg: `${PALETTE.blue}10`,  glow: `${PALETTE.blue}20` },
  instability:{ bg: `${PALETTE.orange}08`, glow: `${PALETTE.orange}20` },
}

export function NarrativeTimeline({ state }: Props) {
  const { timeline, driftChart } = state
  const currentIdx = timeline.currentIndex
  const currentStage = STAGES[currentIdx]?.stage ?? DegradationStage.BASELINE

  /* compute marker positions based on drift chart events */
  const total = STAGES.length
  const segWidth = 100 / total

  return (
    <div style={{ padding: '10px 24px 14px' }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 10,
      }}>
        <span style={{
          fontSize: 9, color: PALETTE.textMuted, textTransform: 'uppercase',
          letterSpacing: '0.14em', fontWeight: 800,
        }}>
          System Narrative
        </span>
        <span style={{ fontSize: 9, color: PALETTE.textSecondary, letterSpacing: '0.06em' }}>
          {STAGES[currentIdx]?.label ?? 'Unknown'}
        </span>
      </div>

      {/* timeline track */}
      <div style={{
        position: 'relative',
        height: 36,
        borderRadius: 6,
        overflow: 'hidden',
        background: PALETTE.surface,
        border: `1px solid ${PALETTE.border}`,
      }}>
        {/* zone backgrounds */}
        {STAGES.map((s, i) => {
          const zone = s.zone
          const col =
            zone === 'baseline' ? PALETTE.neon
            : zone === 'drift' ? PALETTE.blue
            : zone === 'instability' ? PALETTE.orange
            : PALETTE.red

          return (
            <div key={s.stage} style={{
              position: 'absolute', left: `${i * segWidth}%`, top: 0, bottom: 0,
              width: `${segWidth}%`,
              background: i <= currentIdx ? `${col}12` : 'transparent',
              borderRight: i < total - 1 ? `1px solid ${PALETTE.border}` : 'none',
              transition: 'background 0.6s ease',
            }} />
          )
        })}

        {/* stage labels */}
        {STAGES.map((s, i) => (
          <div key={s.stage} style={{
            position: 'absolute', left: `${i * segWidth}%`, top: 0, bottom: 0,
            width: `${segWidth}%`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{
              fontSize: 8, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
              color: i === currentIdx ? PALETTE.textPrimary
                : i < currentIdx ? PALETTE.textSecondary
                : `${PALETTE.textMuted}60`,
              transition: 'color 0.4s ease',
            }}>
              {s.label}
            </span>
          </div>
        ))}

        {/* current position indicator */}
        <div style={{
          position: 'absolute',
          left: `${currentIdx * segWidth + segWidth / 2}%`,
          top: '50%',
          transform: 'translate(-50%, -50%)',
          width: 10, height: 10,
          borderRadius: '50%',
          background: PALETTE.textPrimary,
          boxShadow: `0 0 12px ${PALETTE.textPrimary}60`,
          zIndex: 2,
          transition: 'left 0.8s cubic-bezier(0.4,0,0.2,1)',
        }} />

        {/* passed glow trail */}
        <div style={{
          position: 'absolute', left: 0, top: 0, bottom: 0,
          width: `${(currentIdx + 0.5) * segWidth}%`,
          background: `linear-gradient(90deg, ${PALETTE.neon}08 0%, ${currentIdx >= 3 ? PALETTE.orange : PALETTE.blue}10 100%)`,
          pointerEvents: 'none',
          transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)',
        }} />
      </div>

      {/* markers row */}
      <div style={{ display: 'flex', gap: 20, marginTop: 8, paddingLeft: 4 }}>
        {currentIdx >= 1 && (
          <span style={{ fontSize: 8, color: PALETTE.blue, fontWeight: 700, letterSpacing: '0.06em' }}>
            ● Drift detected
          </span>
        )}
        {currentIdx >= 3 && (
          <span style={{ fontSize: 8, color: PALETTE.orange, fontWeight: 700, letterSpacing: '0.06em' }}>
            ● Instability onset
          </span>
        )}
        {currentIdx >= 4 && (
          <span style={{ fontSize: 8, color: PALETTE.red, fontWeight: 700, letterSpacing: '0.06em' }}>
            ● Intervention required
          </span>
        )}
      </div>
    </div>
  )
}
