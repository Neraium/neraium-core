'use client'

import React, { useState, useEffect } from 'react'
import { DEMO_SCENARIOS } from '@/lib/demoScenarios'
import { DecisionUIState, Severity, Trajectory, DegradationStage } from '@/lib/decisionToUI'
import { PALETTE, severityColor, stageLabel, trajectoryLabel } from '@/lib/dashboardHelpers'
import { SystemField } from './SystemField'
import { DriftInstabilityViz } from './DriftInstabilityViz'
import { SubsystemPanel } from './SubsystemPanel'
import { IntelligencePanel } from './IntelligencePanel'
import { NarrativeTimeline } from './NarrativeTimeline'

function fmt2d(n: number): string {
  return n.toString().padStart(2, '0')
}

export function LouddpaxDashboard() {
  const [scenarioIdx, setScenarioIdx] = useState(0)
  const [uiState, setUiState] = useState<DecisionUIState>(DEMO_SCENARIOS[0].state)
  const [, setTick] = useState(0)

  /* auto-advance demo scenarios */
  useEffect(() => {
    const ms = DEMO_SCENARIOS[scenarioIdx].durationMs
    const t = setTimeout(() => {
      const next = (scenarioIdx + 1) % DEMO_SCENARIOS.length
      setScenarioIdx(next)
      setUiState(DEMO_SCENARIOS[next].state)
    }, ms)
    return () => clearTimeout(t)
  }, [scenarioIdx])

  /* clock tick */
  useEffect(() => {
    const t = setInterval(() => setTick(x => x + 1), 1000)
    return () => clearInterval(t)
  }, [])

  const { statusHeader } = uiState
  const color = severityColor(statusHeader.severity)
  const label = stageLabel(statusHeader.degradationStage)
  const isMoving = statusHeader.trajectory !== Trajectory.STABLE
  const traj = trajectoryLabel(statusHeader.trajectory)

  const now = new Date()
  const timeStr = `${fmt2d(now.getHours())}:${fmt2d(now.getMinutes())}`

  const panelBorder =
    statusHeader.severity === Severity.HIGH     ? 'rgba(239,68,68,0.15)'
    : statusHeader.severity === Severity.ELEVATED ? 'rgba(255,138,92,0.12)'
    : statusHeader.severity === Severity.MODERATE ? 'rgba(96,165,250,0.10)'
    : 'rgba(255,255,255,0.05)'

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      width: '100vw', height: '100vh',
      background: PALETTE.bg,
      color: PALETTE.textPrimary,
      fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',
      overflow: 'hidden',
    }}>
      {/* ═══ HEADER ═══════════════════════════════════════════════════════ */}
      <header style={{
        flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 52, padding: '0 22px',
        borderBottom: `1px solid ${PALETTE.border}`,
        background: 'rgba(5,6,7,0.98)',
        backdropFilter: 'blur(8px)',
      }}>
        {/* brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <img
            src="/louddpax-logo.jpg"
            alt="Louddpax"
            style={{ height: 28, width: 28, objectFit: 'cover', borderRadius: 5 }}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            <span style={{ fontSize: 12, fontWeight: 800, letterSpacing: '0.08em', color: '#fff' }}>
              LOUDDPAX
            </span>
            <span style={{ fontSize: 8, color: PALETTE.textMuted, letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase' }}>
              Grow Facility
            </span>
          </div>
          <span style={{ width: 1, height: 18, background: 'rgba(255,255,255,0.06)', margin: '0 2px' }} />
          <img
            src="/neraium-logo.jpg"
            alt="Neraium"
            style={{ height: 20, width: 20, objectFit: 'cover', borderRadius: 4, opacity: 0.6 }}
          />
        </div>

        {/* status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%',
            background: color,
            boxShadow: `0 0 10px ${color}`,
            animation: 'lpPulse 2.4s ease-in-out infinite',
          }} />
          <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.12em', color }}>
            {label}
          </span>
          {isMoving && (
            <>
              <span style={{ fontSize: 13, color: 'rgba(255,255,255,0.1)', margin: '0 2px' }}>→</span>
              <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.08em', color: PALETTE.textMuted }}>
                {traj}
              </span>
            </>
          )}
        </div>

        {/* right */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          <span style={{ fontSize: 9, color: PALETTE.textMuted, letterSpacing: '0.08em', fontWeight: 700 }}>
            CONF <span style={{ color: PALETTE.textSecondary }}>{statusHeader.confidence === 'HIGH' ? '0.91' : statusHeader.confidence === 'MEDIUM' ? '0.74' : '0.42'}</span>
          </span>
          <span style={{ fontSize: 10, color: PALETTE.textMuted, fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
            {timeStr}
          </span>
          <span style={{
            fontSize: 8, color: PALETTE.neon, letterSpacing: '0.14em', fontWeight: 800,
            border: `1px solid ${PALETTE.neon}30`, borderRadius: 4,
            padding: '3px 7px', background: `${PALETTE.neon}08`,
          }}>
            LIVE
          </span>
        </div>
      </header>

      {/* ═══ MAIN ═════════════════════════════════════════════════════════ */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

        {/* LEFT — System Field */}
        <div style={{
          flex: '0 0 55%',
          borderRight: `1px solid ${panelBorder}`,
          position: 'relative', overflow: 'hidden',
          transition: 'border-color 0.8s ease',
        }}>
          <SystemField state={uiState} />
        </div>

        {/* RIGHT — stacked panels */}
        <div style={{
          flex: '0 0 45%',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
          background: PALETTE.surface,
        }}>
          {/* Drift + Instability */}
          <div style={{ flex: '0 0 auto', borderBottom: `1px solid ${PALETTE.border}` }}>
            <DriftInstabilityViz state={uiState} />
          </div>

          {/* Subsystem Panel */}
          <div style={{ flex: '1 1 auto', overflowY: 'auto', borderBottom: `1px solid ${PALETTE.border}` }}>
            <SubsystemPanel state={uiState} />
          </div>

          {/* Intelligence Panel */}
          <div style={{ flex: '0 0 auto' }}>
            <IntelligencePanel state={uiState} />
          </div>
        </div>
      </div>

      {/* ═══ BOTTOM — Narrative Timeline ══════════════════════════════════ */}
      <div style={{
        flexShrink: 0,
        borderTop: `1px solid ${panelBorder}`,
        background: PALETTE.surface,
        transition: 'border-color 0.8s ease',
      }}>
        <NarrativeTimeline state={uiState} />
      </div>

      {/* global keyframes */}
      <style>{`
        @keyframes lpPulse {
          0%, 100% { box-shadow: 0 0 6px currentColor; opacity: 1; }
          50%       { box-shadow: 0 0 14px currentColor; opacity: 0.7; }
        }
      `}</style>
    </div>
  )
}
