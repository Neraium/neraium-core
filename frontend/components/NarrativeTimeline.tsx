'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { LiveSystem } from '@/lib/simulation'

interface Props {
  system: LiveSystem
  onHoverStage?: (stageKey: string | null) => void
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

const STAGES = [
  { key: 'baseline', label: 'Baseline', desc: 'All systems nominal' },
  { key: 'early_shift', label: 'Drift', desc: 'First structural separation' },
  { key: 'emerging', label: 'Emerging', desc: 'Pattern forming' },
  { key: 'persistent', label: 'Persistent', desc: 'Degradation anchored' },
  { key: 'accelerated', label: 'Accelerated', desc: 'Cascade building' },
  { key: 'failure_approach', label: 'Critical', desc: 'Regime break imminent' },
] as const

export function NarrativeTimeline({ system, onHoverStage }: Props) {
  const currentIdx = STAGES.findIndex(s => s.key === system.stage)
  const total = STAGES.length
  const segWidth = 100 / total

  const narrativeProgress = (system.scenarioIdx + system.scenarioProgress) / 6

  return (
    <div style={{ padding: '14px 24px 18px' }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 12,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontSize: 8, color: C.muted, textTransform: 'uppercase',
            letterSpacing: '0.18em', fontWeight: 700,
          }}>
            State Evolution
          </span>
          <span style={{
            fontSize: 8, color: C.secondary, letterSpacing: '0.08em', fontWeight: 600,
            padding: '2px 8px', borderRadius: 4, background: 'rgba(255,255,255,0.03)',
            border: `1px solid ${C.border}`,
          }}>
            {STAGES[currentIdx]?.label}
          </span>
        </div>
        <span style={{ fontSize: 8, color: C.secondary, letterSpacing: '0.08em', fontWeight: 600 }}>
          Drift {(system.structuralDrift * 100).toFixed(0)}% · {(narrativeProgress * 100).toFixed(0)}% narrative
        </span>
      </div>

      {/* Track */}
      <div style={{
        position: 'relative',
        height: 44,
        borderRadius: 8,
        overflow: 'hidden',
        background: C.surface,
        border: `1px solid ${C.border}`,
      }}>
        {/* Zone fills with depth */}
        {STAGES.map((s, i) => {
          const col =
            i <= 1 ? C.neon
            : i <= 3 ? C.blue
            : i <= 4 ? C.orange
            : C.red
          const isPast = i < currentIdx
          const isCurrent = i === currentIdx
          return (
            <div
              key={s.key}
              style={{
                position: 'absolute', left: `${i * segWidth}%`, top: 0, bottom: 0,
                width: `${segWidth}%`,
                background: isPast ? `${col}0D` : isCurrent ? `${col}08` : 'transparent',
                borderRight: i < total - 1 ? `1px solid ${C.border}` : 'none',
                transition: 'background 0.8s ease',
                cursor: 'pointer',
              }}
              onMouseEnter={() => onHoverStage?.(s.key)}
              onMouseLeave={() => onHoverStage?.(null)}
            />
          )
        })}

        {/* Narrative trail — gradient showing journey */}
        <div style={{
          position: 'absolute', left: 0, top: 0, bottom: 0,
          width: `${narrativeProgress * 100}%`,
          background: `linear-gradient(90deg, ${C.neon}06 0%, ${currentIdx >= 3 ? C.orange : C.blue}10 100%)`,
          transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)',
          pointerEvents: 'none',
        }} />

        {/* Stage labels */}
        {STAGES.map((s, i) => {
          const isCurrent = i === currentIdx
          const isPast = i < currentIdx
          return (
            <div key={s.key} style={{
              position: 'absolute', left: `${i * segWidth}%`, top: 0, bottom: 0,
              width: `${segWidth}%`,
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              gap: 2,
              cursor: 'pointer',
            }}
              onMouseEnter={() => onHoverStage?.(s.key)}
              onMouseLeave={() => onHoverStage?.(null)}
            >
              <span style={{
                fontSize: 8, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase',
                color: isCurrent ? C.text : isPast ? C.secondary : `${C.muted}60`,
                transition: 'color 0.5s ease',
              }}>
                {s.label}
              </span>
              <span style={{
                fontSize: 7, fontWeight: 600, letterSpacing: '0.04em',
                color: isCurrent ? C.secondary : `${C.muted}40`,
                transition: 'color 0.5s ease',
                maxWidth: '90%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {s.desc}
              </span>
            </div>
          )
        })}

        {/* Current position indicator */}
        <motion.div
          style={{
            position: 'absolute',
            left: `${(currentIdx + 0.5) * segWidth}%`,
            top: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 3,
          }}
          animate={{ left: `${(currentIdx + 0.5) * segWidth}%` }}
          transition={{ duration: 1, ease: [0.4, 0, 0.2, 1] }}
        >
          <div style={{
            width: 12, height: 12, borderRadius: '50%',
            background: C.text,
            boxShadow: `0 0 16px ${C.text}50, 0 0 32px ${C.text}20`,
            position: 'relative',
          }}>
            <motion.div
              style={{
                position: 'absolute', inset: -6,
                borderRadius: '50%',
                border: `1px solid ${C.text}30`,
              }}
              animate={{ scale: [1, 1.4, 1], opacity: [0.5, 0.1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            />
          </div>
        </motion.div>

        {/* Progress line overlay */}
        <div style={{
          position: 'absolute', left: 0, top: '50%',
          width: `${narrativeProgress * 100}%`,
          height: 1,
          background: `linear-gradient(90deg, ${C.neon}30, ${currentIdx >= 3 ? C.orange : C.blue}30)`,
          transform: 'translateY(-50%)',
          pointerEvents: 'none',
          transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)',
        }} />
      </div>

      {/* Event markers */}
      <div style={{ display: 'flex', gap: 18, marginTop: 10, paddingLeft: 2 }}>
        {currentIdx >= 1 && (
          <span style={{ fontSize: 7, color: C.blue, fontWeight: 700, letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: C.blue }} />
            Drift emerged
          </span>
        )}
        {currentIdx >= 3 && (
          <span style={{ fontSize: 7, color: C.orange, fontWeight: 700, letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: C.orange }} />
            Coherence break
          </span>
        )}
        {currentIdx >= 5 && (
          <span style={{ fontSize: 7, color: C.red, fontWeight: 700, letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: C.red }} />
            Critical regime
          </span>
        )}
      </div>
    </div>
  )
}
