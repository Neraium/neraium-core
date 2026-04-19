'use client'

import React, { useState, useEffect } from 'react'
import { DEMO_SCENARIOS } from '@/lib/demoScenarios'
import { DecisionUIState, Severity, Trajectory, DegradationStage } from '@/lib/decisionToUI'
import { SystemModelView }      from './SystemModelView'
import { DriftTrajectoryPanel } from './DriftTrajectoryPanel'
import { StateTimelineBar }     from './StateTimelineBar'

/* ─── helpers ─────────────────────────────────────────────────────────────── */

function severityColor(s: Severity): string {
  return s === Severity.HIGH     ? '#C94C4C' :
         s === Severity.ELEVATED ? '#D8A35D' :
         s === Severity.MODERATE ? '#FBBF24' : '#7E9F2E'
}

function stageLabel(stage: DegradationStage): string {
  return stage === DegradationStage.BASELINE        ? 'STABLE'    :
         stage === DegradationStage.EARLY_SHIFT     ? 'DRIFTING'  :
         stage === DegradationStage.EMERGING        ? 'UNSTABLE'  :
         stage === DegradationStage.PERSISTENT      ? 'UNSTABLE'  :
         stage === DegradationStage.ACCELERATED     ? 'CRITICAL'  : 'CRITICAL'
}

function confidenceDisplay(c: 'LOW' | 'MEDIUM' | 'HIGH'): string {
  return c === 'HIGH' ? '0.91' : c === 'MEDIUM' ? '0.74' : '0.42'
}

function fmt2d(n: number): string {
  return n.toString().padStart(2, '0')
}

/* ─── Component ───────────────────────────────────────────────────────────── */

export function LouddpaxDashboard() {
  const [scenarioIdx, setScenarioIdx] = useState(0)
  const [uiState, setUiState]         = useState<DecisionUIState>(DEMO_SCENARIOS[0].state)
  const [, setTick]                   = useState(0)

  /* auto-advance demo scenarios */
  useEffect(() => {
    const ms = DEMO_SCENARIOS[scenarioIdx].durationMs
    const t  = setTimeout(() => {
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
  const color      = severityColor(statusHeader.severity)
  const label      = stageLabel(statusHeader.degradationStage)
  const conf       = confidenceDisplay(statusHeader.confidence)
  const isMoving   = statusHeader.trajectory !== Trajectory.STABLE
  const trajLabel  = statusHeader.trajectory === Trajectory.DEGRADING ? 'DETERIORATING' : 'UNCERTAIN'

  const now = new Date()
  const timeStr = `${fmt2d(now.getHours())}:${fmt2d(now.getMinutes())}`

  const panelBorderColor =
    statusHeader.severity === Severity.HIGH     ? 'rgba(201,76,76,0.18)'   :
    statusHeader.severity === Severity.ELEVATED ? 'rgba(216,163,93,0.14)'  :
    statusHeader.severity === Severity.MODERATE ? 'rgba(251,191,36,0.10)'  :
    'rgba(255,255,255,0.06)'

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      width: '100vw', height: '100vh',
      background: '#020203',
      color: '#E8EBF0',
      fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',
      overflow: 'hidden',
    }}>

      {/* ═══ HEADER ═══════════════════════════════════════════════════════ */}
      <header style={{
        flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 56, padding: '0 24px',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        background: 'rgba(2,2,3,0.98)',
      }}>
        {/* brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <img
            src="/louddpax-logo.jpg"
            alt="Louddpax"
            style={{ height: 32, width: 32, objectFit: 'cover', borderRadius: 6 }}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <span style={{
              fontSize: 13, fontWeight: 800, letterSpacing: '0.08em',
              color: '#fff',
            }}>
              LOUDDPAX
            </span>
            <span style={{ fontSize: 9, color: '#4b5563', letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase' }}>
              Grow Facility
            </span>
          </div>
          <span style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.08)', margin: '0 4px' }} />
          <img
            src="/neraium-logo.jpg"
            alt="Neraium"
            style={{ height: 22, width: 22, objectFit: 'cover', borderRadius: 4, opacity: 0.7 }}
          />
        </div>

        {/* status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: color,
            boxShadow: `0 0 10px ${color}`,
            animation: 'lpStatusPulse 2.4s ease-in-out infinite',
          }} />
          <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.12em', color }}>
            {label}
          </span>
          {isMoving && (
            <>
              <span style={{ fontSize: 14, color: 'rgba(255,255,255,0.12)', margin: '0 2px' }}>→</span>
              <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', color: '#6b7280' }}>
                {trajLabel}
              </span>
            </>
          )}
        </div>

        {/* right side */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <span style={{ fontSize: 10, color: '#374151', letterSpacing: '0.08em', fontWeight: 700 }}>
            CONFIDENCE <span style={{ color: '#d1d5db' }}>{conf}</span>
          </span>
          <span style={{ fontSize: 11, color: '#4b5563', fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
            {timeStr}
          </span>
          <span style={{
            fontSize: 9, color: '#7E9F2E', letterSpacing: '0.14em', fontWeight: 800,
            border: '1px solid rgba(126,159,46,0.3)', borderRadius: 4,
            padding: '3px 8px', background: 'rgba(126,159,46,0.08)',
          }}>
            LIVE
          </span>
        </div>
      </header>

      {/* ═══ MAIN ═════════════════════════════════════════════════════════ */}
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

      {/* ═══ BOTTOM — State Timeline ══════════════════════════════════════ */}
      <div style={{
        flexShrink: 0,
        borderTop: `1px solid ${panelBorderColor}`,
        transition: 'border-color 0.8s ease',
      }}>
        <StateTimelineBar state={uiState} />
      </div>

      {/* global keyframes */}
      <style>{`
        @keyframes lpStatusPulse {
          0%, 100% { box-shadow: 0 0 6px currentColor; opacity: 1; }
          50%       { box-shadow: 0 0 14px currentColor; opacity: 0.7; }
        }
      `}</style>
    </div>
  )
}
