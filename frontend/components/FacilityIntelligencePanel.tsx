'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { LiveSystem } from '@/lib/simulation'
import { computeFacilityContext } from '@/lib/facilityContext'

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

export function FacilityIntelligencePanel({ system }: Props) {
  const ctx = computeFacilityContext(system)
  const col = system.severity === 'HIGH' ? C.red : system.severity === 'ELEVATED' ? C.orange : system.severity === 'MODERATE' ? C.blue : C.neon

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100%', overflowY: 'auto',
      background: C.surface,
      borderLeft: `1px solid ${C.border}`,
      padding: '18px 20px',
      gap: 14,
    }}>
      <div style={{
        fontSize: 8, letterSpacing: '0.16em', textTransform: 'uppercase',
        color: C.muted, fontWeight: 700,
      }}>
        Facility Intelligence
      </div>

      {/* Coherence Gauge */}
      <div style={{
        background: `linear-gradient(180deg, #0d1014 0%, ${C.surface} 100%)`,
        border: `1px solid ${C.border}`,
        borderRadius: 10,
        padding: '14px 16px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
          <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            Facility Coherence
          </span>
          <span style={{ fontSize: 18, fontWeight: 900, color: system.coherence > 0.6 ? C.neon : system.coherence > 0.3 ? C.orange : C.red }}>
            {(system.coherence * 100).toFixed(0)}%
          </span>
        </div>
        <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
          <motion.div
            style={{
              height: '100%',
              background: system.coherence > 0.6 ? C.neon : system.coherence > 0.3 ? C.orange : C.red,
              borderRadius: 2,
            }}
            animate={{ width: `${system.coherence * 100}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          />
        </div>
        <div style={{ marginTop: 6, fontSize: 9, color: C.secondary, fontWeight: 600 }}>
          {ctx.coherenceTrend === 'weakening'
            ? 'Coherence weakening — structural separation spreading'
            : ctx.coherenceTrend === 'recovering'
            ? 'Coherence recovering — subsystems realigning'
            : 'Coherence stable — facility behavior intact'}
        </div>
      </div>

      {/* Attention Priority */}
      <IntelCard
        title="Attention Priority"
        accent={col}
        content={ctx.attentionPriority}
        sub={ctx.mostUrgentRoom ? `Drift ${(ctx.mostUrgentRoom.driftContribution * 100).toFixed(0)}% · ${ctx.mostUrgentRoom.behavioralState}` : undefined}
      />

      {/* Emerging Issue */}
      <IntelCard
        title="Emerging Issue"
        accent={col}
        content={ctx.emergingIssue}
      />

      {/* Pattern Cluster */}
      {ctx.patternClusters.length > 0 && (
        <IntelCard
          title="Pattern Cluster"
          accent={C.blue}
          content={ctx.patternClusters.map(c => `${c.label}: ${c.rooms.join(', ')}`).join('\n')}
        />
      )}

      {/* Recent Change */}
      <IntelCard
        title="Recent Change"
        accent={C.neon}
        content={ctx.recentChange}
      />

      {/* System-level text from simulation */}
      <div style={{
        background: `${col}08`,
        border: `1px solid ${col}15`,
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

      <div style={{ marginTop: 'auto', paddingTop: 10, borderTop: `1px solid ${C.border}` }}>
        <div style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 4 }}>
          Path Outlook
        </div>
        <div style={{ fontSize: 10, color: C.secondary, fontWeight: 500, lineHeight: 1.4 }}>
          {system.pathOutlook}
        </div>
      </div>
    </div>
  )
}

function IntelCard({ title, accent, content, sub }: { title: string; accent: string; content: string; sub?: string }) {
  return (
    <div style={{
      background: C.surfaceHi,
      border: `1px solid ${C.border}`,
      borderLeft: `2px solid ${accent}`,
      borderRadius: 8,
      padding: '10px 12px',
      position: 'relative',
      overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', left: 0, top: 0, bottom: 0, width: 40,
        background: `linear-gradient(90deg, ${accent}08 0%, transparent 100%)`,
        pointerEvents: 'none',
      }} />
      <div style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ fontSize: 7, color: C.muted, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', marginBottom: 4 }}>
          {title}
        </div>
        <div style={{ fontSize: 11, color: C.text, fontWeight: 700, lineHeight: 1.4, whiteSpace: 'pre-line' }}>
          {content}
        </div>
        {sub && (
          <div style={{ fontSize: 9, color: C.secondary, fontWeight: 600, marginTop: 3 }}>
            {sub}
          </div>
        )}
      </div>
    </div>
  )
}
