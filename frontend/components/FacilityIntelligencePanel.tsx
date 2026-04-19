'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { LiveSystem } from '@/lib/simulation'
import { computeFacilityContext } from '@/lib/facilityContext'

interface Props {
  system: LiveSystem
  hoveredRoomId?: string | null
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

export function FacilityIntelligencePanel({ system, hoveredRoomId }: Props) {
  const ctx = computeFacilityContext(system)
  const col = system.severity === 'HIGH' ? C.red : system.severity === 'ELEVATED' ? C.orange : system.severity === 'MODERATE' ? C.blue : C.neon

  const hoveredRoom = hoveredRoomId ? system.rooms.find(r => r.id === hoveredRoomId) : null

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100%', overflowY: 'auto',
      background: 'linear-gradient(180deg, #080a0d 0%, #050607 100%)',
      borderLeft: `1px solid ${C.border}`,
      padding: '20px 22px',
      gap: 12,
    }}>
      {/* Header */}
      <div style={{
        fontSize: 8, letterSpacing: '0.18em', textTransform: 'uppercase',
        color: C.muted, fontWeight: 700, marginBottom: 2,
      }}>
        Operator Intelligence
      </div>

      {/* ── Current State ───────────────────────────────────────────── */}
      <IntelSection title="Current State" accent={col}>
        <div style={{ fontSize: 15, fontWeight: 800, color: col, lineHeight: 1.3, letterSpacing: '-0.01em' }}>
          {ctx.facilityStateLabel}
        </div>
        <div style={{ fontSize: 10, color: C.secondary, fontWeight: 500, lineHeight: 1.4, marginTop: 4 }}>
          {system.explanation}
        </div>
      </IntelSection>

      {/* ── Onset ───────────────────────────────────────────────────── */}
      {system.onsetMinutes !== null && (
        <IntelSection title="Onset" accent={C.blue}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
            <span style={{ fontSize: 18, fontWeight: 900, color: C.text, fontVariantNumeric: 'tabular-nums' }}>
              ~{system.onsetMinutes}
            </span>
            <span style={{ fontSize: 10, color: C.secondary, fontWeight: 600 }}>min since first drift signal</span>
          </div>
        </IntelSection>
      )}

      {/* ── Coherence ───────────────────────────────────────────────── */}
      <div style={{
        background: `linear-gradient(180deg, #0d1014 0%, ${C.surface} 100%)`,
        border: `1px solid ${C.border}`,
        borderRadius: 10,
        padding: '14px 16px',
        position: 'relative',
        overflow: 'hidden',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
          <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            Coherence
          </span>
          <span style={{ fontSize: 20, fontWeight: 900, color: system.coherence > 0.6 ? C.neon : system.coherence > 0.3 ? C.orange : C.red, fontVariantNumeric: 'tabular-nums' }}>
            {(system.coherence * 100).toFixed(0)}%
          </span>
        </div>
        <div style={{ height: 4, background: 'rgba(255,255,255,0.05)', borderRadius: 2, overflow: 'hidden' }}>
          <motion.div
            style={{
              height: '100%',
              background: system.coherence > 0.6 ? C.neon : system.coherence > 0.3 ? C.orange : C.red,
              borderRadius: 2,
              boxShadow: `0 0 10px ${system.coherence > 0.6 ? C.neon : system.coherence > 0.3 ? C.orange : C.red}30`,
            }}
            animate={{ width: `${system.coherence * 100}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          />
        </div>
        <div style={{ marginTop: 6, fontSize: 9, color: C.secondary, fontWeight: 600, lineHeight: 1.4 }}>
          {ctx.coherenceTrend === 'weakening'
            ? 'Structural separation spreading across facility'
            : ctx.coherenceTrend === 'recovering'
            ? 'Subsystems realigning toward baseline'
            : 'Facility behavior intact — all zones coupled'}
        </div>
      </div>

      {/* ── Primary Driver ──────────────────────────────────────────── */}
      <IntelSection title="Primary Driver" accent={C.orange}>
        <div style={{ fontSize: 13, fontWeight: 800, color: C.text, lineHeight: 1.3 }}>
          {system.primaryDriver}
        </div>
        {ctx.mostUrgentRoom && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
            <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Drift</span>
            <span style={{ fontSize: 11, color: C.orange, fontWeight: 800 }}>{(ctx.mostUrgentRoom.driftContribution * 100).toFixed(0)}%</span>
            <span style={{ width: 1, height: 10, background: 'rgba(255,255,255,0.08)' }} />
            <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' }}>State</span>
            <span style={{ fontSize: 10, color: C.secondary, fontWeight: 600 }}>{ctx.mostUrgentRoom.behavioralState}</span>
          </div>
        )}
      </IntelSection>

      {/* ── Operator Focus ──────────────────────────────────────────── */}
      <IntelSection title="Operator Focus" accent={col}>
        <div style={{ fontSize: 11, color: C.text, fontWeight: 600, lineHeight: 1.5 }}>
          {system.operatorFocus}
        </div>
      </IntelSection>

      {/* ── Path Outlook ────────────────────────────────────────────── */}
      <div style={{ marginTop: 'auto', paddingTop: 12, borderTop: `1px solid ${C.border}` }}>
        <div style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', marginBottom: 6 }}>
          Path Outlook
        </div>
        <div style={{
          fontSize: 10, color: C.secondary, fontWeight: 500, lineHeight: 1.5,
          padding: '10px 12px',
          background: 'rgba(255,255,255,0.02)',
          borderRadius: 6,
          border: `1px solid ${C.border}`,
        }}>
          {system.pathOutlook}
        </div>
      </div>

      {/* ── Hovered Room Detail (live contextual) ───────────────────── */}
      {hoveredRoom && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 8 }}
          transition={{ duration: 0.25 }}
          style={{
            position: 'absolute', bottom: 16, left: 16, right: 16,
            background: 'rgba(5,6,8,0.96)',
            border: `1px solid ${C.border}`,
            borderLeft: `2px solid ${hoveredRoom.status === 'critical' ? C.red : hoveredRoom.status === 'warning' ? C.orange : C.neon}`,
            borderRadius: 8,
            padding: '12px 14px',
            backdropFilter: 'blur(8px)',
            zIndex: 10,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 11, fontWeight: 800, color: C.text }}>{hoveredRoom.shortName}</span>
            <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              {hoveredRoom.behavioralState}
            </span>
          </div>
          <div style={{ fontSize: 9, color: C.secondary, lineHeight: 1.4 }}>
            Drift {(hoveredRoom.driftContribution * 100).toFixed(0)}% · Confidence {(hoveredRoom.confidence * 100).toFixed(0)}% · {hoveredRoom.cycle === 'day' ? 'Lights on' : 'Night recovery'}
          </div>
        </motion.div>
      )}
    </div>
  )
}

function IntelSection({ title, accent, children }: { title: string; accent: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: C.surfaceHi,
      border: `1px solid ${C.border}`,
      borderLeft: `2px solid ${accent}`,
      borderRadius: 8,
      padding: '12px 14px',
      position: 'relative',
      overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', left: 0, top: 0, bottom: 0, width: 50,
        background: `linear-gradient(90deg, ${accent}08 0%, transparent 100%)`,
        pointerEvents: 'none',
      }} />
      <div style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ fontSize: 7, color: C.muted, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', marginBottom: 6 }}>
          {title}
        </div>
        {children}
      </div>
    </div>
  )
}
