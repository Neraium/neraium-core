'use client'

import React, { useState, useEffect } from 'react'
import { DEMO_SCENARIOS } from '@/lib/demoScenarios'
import { DecisionUIState, Severity, Trajectory, DegradationStage } from '@/lib/decisionToUI'
import { SystemModelView }      from './SystemModelView'
import { DriftTrajectoryPanel } from './DriftTrajectoryPanel'
import { StateTimelineBar }     from './StateTimelineBar'

// ─── Header helpers ───────────────────────────────────────────────────────────

function severityColor(s: Severity): string {
  return s === Severity.HIGH     ? '#ef4444' :
         s === Severity.ELEVATED ? '#f59e0b' :
         s === Severity.MODERATE ? '#fbbf24' : '#22c55e'
}

function stageLabel(stage: DegradationStage): string {
  return stage === DegradationStage.BASELINE        ? 'STABLE'    :
         stage === DegradationStage.EARLY_SHIFT     ? 'DRIFTING'  :
         stage === DegradationStage.EMERGING        ? 'UNSTABLE'  :
         stage === DegradationStage.PERSISTENT      ? 'UNSTABLE'  :
         stage === DegradationStage.ACCELERATED     ? 'CRITICAL'  : 'CRITICAL'
}

function confidenceDisplay(c: 'LOW' | 'MEDIUM' | 'HIGH'): string {
  return c === 'HIGH' ? '0.87' : c === 'MEDIUM' ? '0.73' : '0.41'
}

function fmt2d(n: number): string {
  return n.toString().padStart(2, '0')
}

// ─── Component ────────────────────────────────────────────────────────────────

export function NeraiumDashboard() {
  const [scenarioIdx, setScenarioIdx] = useState(0)
  const [uiState, setUiState]         = useState<DecisionUIState>(DEMO_SCENARIOS[0].state)
  const [, setTick]                   = useState(0)  // forces clock re-render each second

  // Auto-advance demo scenarios
  useEffect(() => {
    const ms = DEMO_SCENARIOS[scenarioIdx].durationMs
    const t  = setTimeout(() => {
      const next = (scenarioIdx + 1) % DEMO_SCENARIOS.length
      setScenarioIdx(next)
      setUiState(DEMO_SCENARIOS[next].state)
    }, ms)
    return () => clearTimeout(t)
  }, [scenarioIdx])

  // Clock tick
  useEffect(() => {
    const t = setInterval(() => setTick(x => x + 1), 1000)
    return () => clearInterval(t)
  }, [])

  const { statusHeader } = uiState
  const color      = severityColor(statusHeader.severity)
  const label      = stageLabel(statusHeader.degradationStage)
  const conf       = confidenceDisplay(statusHeader.confidence)
  const isMoving   = statusHeader.trajectory !== Trajectory.STABLE
  const trajLabel  = statusHeader.trajectory === Trajectory.DEGRADING ? 'DETERIORATING' : 'UNCERTAIN'

  const now = new Date()
  const timeStr = `${fmt2d(now.getHours())}:${fmt2d(now.getMinutes())}:${fmt2d(now.getSeconds())}`

  const panelBorderColor =
    statusHeader.severity === Severity.HIGH     ? 'rgba(239,68,68,0.18)'     :
    statusHeader.severity === Severity.ELEVATED ? 'rgba(245,158,11,0.14)'    :
    statusHeader.severity === Severity.MODERATE ? 'rgba(250,204,21,0.10)'    :
    'rgba(255,255,255,0.06)'

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      width: '100vw', height: '100vh',
      background: '#050607',
      color: '#e8ebf0',
      fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',
      overflow: 'hidden',
    }}>

      {/* ─── HEADER ──────────────────────────────────────────────────────────── */}
      <header style={{
        flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 52, padding: '0 28px',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        background: 'rgba(5,6,7,0.98)',
      }}>
        {/* Brand + system name */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <span style={{
            fontSize: 13, fontWeight: 800, letterSpacing: '0.14em',
            color: '#fff',
          }}>
            NERAIUM
          </span>
          <span style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.1)' }} />
          <span style={{ fontSize: 12, color: '#6b7280', letterSpacing: '0.04em', fontWeight: 500 }}>
            Cooling System A
          </span>
        </div>

        {/* Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.1em', color }}>
            {label}
          </span>
          {isMoving && (
            <>
              <span style={{ fontSize: 15, color: 'rgba(255,255,255,0.18)' }}>→</span>
              <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: '0.08em', color: '#6b7280' }}>
                {trajLabel}
              </span>
            </>
          )}
        </div>

        {/* Confidence + time + live */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <span style={{ fontSize: 11, color: '#4b5563', letterSpacing: '0.06em' }}>
            CONFIDENCE{' '}
            <span style={{ color: '#d1d5db', fontWeight: 700 }}>{conf}</span>
          </span>
          <span style={{ fontSize: 11, color: '#374151', fontVariantNumeric: 'tabular-nums' }}>
            {timeStr}
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%',
              background: '#22c55e',
              boxShadow: '0 0 8px rgba(34,197,94,0.75)',
              display: 'inline-block',
              animation: 'nrLivePulse 2s ease-in-out infinite',
            }} />
            <span style={{ fontSize: 11, color: '#22c55e', letterSpacing: '0.1em', fontWeight: 700 }}>
              LIVE
            </span>
          </span>
        </div>
      </header>

      {/* ─── MAIN ────────────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

        {/* LEFT 60% — System Model */}
        <div style={{
          flex: '0 0 60%',
          borderRight: `1px solid ${panelBorderColor}`,
          position: 'relative', overflow: 'hidden',
          transition: 'border-color 0.8s ease',
        }}>
          <SystemModelView state={uiState} />
        </div>

        {/* RIGHT 40% — Drift + Trajectory */}
        <div style={{
          flex: '0 0 40%',
          overflow: 'hidden',
          display: 'flex', flexDirection: 'column',
        }}>
          <DriftTrajectoryPanel state={uiState} />
        </div>
      </div>

      {/* ─── BOTTOM — State Timeline ─────────────────────────────────────────── */}
      <div style={{
        flexShrink: 0,
        borderTop: `1px solid ${panelBorderColor}`,
        transition: 'border-color 0.8s ease',
      }}>
        <StateTimelineBar state={uiState} />
      </div>

      {/* global keyframes */}
      <style>{`
        @keyframes nrLivePulse {
          0%, 100% { box-shadow: 0 0 6px rgba(34,197,94,0.6); }
          50%       { box-shadow: 0 0 14px rgba(34,197,94,0.9); }
        }
      `}</style>
    </div>
  )
}
