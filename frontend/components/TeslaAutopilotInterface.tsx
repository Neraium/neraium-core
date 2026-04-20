'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { EnhancedTetrahedronViz } from './EnhancedTetrahedronViz'
import { usePhaseController } from '@/lib/phaseController'
import { useSystemInterpolation } from '@/lib/systemInterpolation'
import { computeConsequenceState } from '@/lib/decisionGravity'
import { computeDiagnosticLegibility } from '@/lib/diagnosticLegibility'
import { computeOutcomeConfidence } from '@/lib/outcomeConfidence'
import { evaluateActionDecision, RankedAction, ActionDecisionResult } from '@/lib/actionEvaluation'

interface SystemData {
  drift: number
  stability: number
  coherence: number
  confidence: number
  timestamp: string
  systemState: 'Stable' | 'Drift forming' | 'Instability forming' | 'Critical'
}

interface TeslaAutopilotInterfaceProps {
  systemData?: SystemData
  onTogglePlay?: (isPlaying: boolean) => void
  onSpeedChange?: (speed: number) => void
}

const getStateLabel = (phase: string): string => {
  const map: Record<string, string> = {
    'Stable': 'Stable',
    'Drift forming': 'Drift',
    'Instability forming': 'Instability',
    'Critical': 'Critical'
  }
  return map[phase] || phase
}

const getStateColor = (phase: string): string => {
  const colors: Record<string, string> = {
    'Stable': '#22c55e',
    'Drift forming': '#eab308',
    'Instability forming': '#f97316',
    'Critical': '#ef4444'
  }
  return colors[phase] || '#9ca3af'
}

export function TeslaAutopilotInterface({
  systemData,
  onTogglePlay,
  onSpeedChange,
}: TeslaAutopilotInterfaceProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [driftHistory, setDriftHistory] = useState<number[]>([])
  const [stabilityHistory, setStabilityHistory] = useState<number[]>([])
  const [actionHistory, setActionHistory] = useState<RankedAction[]>([])
  const [actionDecision, setActionDecision] = useState<ActionDecisionResult | null>(null)

  const { phase, phaseProgress } = usePhaseController(isPlaying, speed)
  const interpolatedData = useSystemInterpolation(systemData, phase)

  useEffect(() => {
    setDriftHistory(prev => [...prev.slice(-10), interpolatedData.interpolatedDrift])
    setStabilityHistory(prev => [...prev.slice(-10), interpolatedData.interpolatedStability])
  }, [interpolatedData.interpolatedDrift, interpolatedData.interpolatedStability])

  const diagnostics = computeDiagnosticLegibility(
    phase,
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    driftHistory,
    stabilityHistory
  )

  const consequenceState = computeConsequenceState(
    phase,
    phaseProgress,
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    phase
  )

  const outcomeState = computeOutcomeConfidence(
    diagnostics.dominantDriver || 'multi-mode',
    phase,
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    diagnostics.confidence
  )

  const driftAcceleration = driftHistory.length > 1
    ? driftHistory[driftHistory.length - 1] - driftHistory[Math.max(0, driftHistory.length - 3)]
    : 0

  useEffect(() => {
    const decision = evaluateActionDecision(
      phase,
      diagnostics.dominantDriver || 'multi-mode',
      interpolatedData.interpolatedDrift,
      interpolatedData.interpolatedStability,
      interpolatedData.interpolatedCoherence,
      diagnostics.confidence,
      consequenceState.timeToImpact,
      driftAcceleration,
      outcomeState.noActionConsequence,
      actionHistory
    )
    setActionDecision(decision)

    if (decision.primaryAction) {
      setActionHistory(prev => [...prev.slice(-4), {
        label: decision.primaryAction.label,
        score: decision.primaryAction.score,
        stabilizationBenefit: decision.primaryAction.stabilizationBenefit,
        riskReduction: 0,
        failureModeFit: 1,
        timeSensitivityFit: 0.8,
        confidence: 0.8,
        aggressivenessPenalty: 0.9,
        outcome: decision.primaryAction.outcome,
        timingSensitivity: decision.primaryAction.timingSensitivity
      }])
    }
  }, [phase, diagnostics.dominantDriver, interpolatedData.interpolatedDrift, interpolatedData.interpolatedStability, interpolatedData.interpolatedCoherence, diagnostics.confidence, consequenceState.timeToImpact, driftAcceleration, outcomeState.noActionConsequence])

  const handleTogglePlay = () => {
    setIsPlaying(!isPlaying)
    onTogglePlay?.(!isPlaying)
  }

  const confidentLabel = diagnostics.confidence === 'high' ? 'High' : diagnostics.confidence === 'medium' ? 'Moderate' : 'Low'

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        background: '#050607',
        color: '#e2e8f0',
        overflow: 'hidden',
        position: 'relative',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      {/* SCANLINE OVERLAY */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.15) 2px, rgba(0,0,0,0.15) 4px)',
          pointerEvents: 'none',
          zIndex: 99,
          opacity: 0.2,
        }}
      />

      {/* TOP STATUS STRIP - PERSISTENT, ALWAYS VISIBLE */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 50,
          background: 'linear-gradient(180deg, rgba(5,6,7,0.95) 0%, rgba(5,6,7,0.5) 100%)',
          backdropFilter: 'blur(4px)',
          borderBottom: `1px solid ${getStateColor(phase)}33`,
          padding: '12px 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '13px',
          letterSpacing: '0.5px',
        }}
      >
        <div style={{ display: 'flex', gap: '32px', alignItems: 'center', flex: 1 }}>
          {/* System Name */}
          <div style={{ fontWeight: 500, color: '#cbd5e1' }}>
            Grow Room A / Unit 1
          </div>

          {/* Status Separator */}
          <div style={{ width: '1px', height: '16px', background: 'rgba(203,213,225,0.2)' }} />

          {/* Mode Indicator */}
          <motion.div
            animate={{
              color: getStateColor(phase),
              textShadow: phase === 'Critical' ? `0 0 8px ${getStateColor(phase)}` : 'none'
            }}
            style={{
              fontWeight: 600,
              fontSize: '13px',
              letterSpacing: '0.5px',
              textTransform: 'uppercase',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            Mode: {getStateLabel(phase)}
          </motion.div>

          {/* Confidence */}
          <div style={{ color: '#94a3b8', fontSize: '12px' }}>
            Confidence: {confidentLabel}
          </div>

          {/* Time-to-Impact */}
          <motion.div
            animate={{
              color: consequenceState.timeToImpact < 5 ? '#ef4444' : consequenceState.timeToImpact < 10 ? '#f97316' : '#94a3b8'
            }}
            style={{
              fontSize: '12px',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            Time-to-Impact: {consequenceState.timeToImpact < 1 ? 'Imminent' : `~${Math.ceil(consequenceState.timeToImpact)} cycles`}
          </motion.div>
        </div>

        {/* Right side controls */}
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <button
            onClick={handleTogglePlay}
            style={{
              background: 'rgba(126, 159, 46, 0.2)',
              border: `1px solid ${isPlaying ? '#7e9f2e' : 'rgba(126, 159, 46, 0.4)'}`,
              color: '#7e9f2e',
              padding: '4px 12px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 500,
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(126, 159, 46, 0.3)'
              e.currentTarget.style.borderColor = '#7e9f2e'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(126, 159, 46, 0.2)'
              e.currentTarget.style.borderColor = isPlaying ? '#7e9f2e' : 'rgba(126, 159, 46, 0.4)'
            }}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
        </div>
      </motion.div>

      {/* MAIN CONTENT AREA */}
      <div
        style={{
          position: 'absolute',
          top: 52,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* CENTER SYSTEM VIEW - PRIMARY */}
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            minHeight: 0,
          }}
        >
          {/* Tetrahedron - The System Itself */}
          <div
            style={{
              width: '100%',
              height: '100%',
              position: 'relative',
            }}
          >
            <EnhancedTetrahedronViz
              position={[
                interpolatedData.interpolatedPosition?.x || 0.5,
                interpolatedData.interpolatedPosition?.y || 0.5,
                interpolatedData.interpolatedPosition?.z || 0.5,
              ]}
              severity={phase === 'Critical' ? 3 : phase === 'Instability forming' ? 2 : phase === 'Drift forming' ? 1 : 0}
              trailHistory={interpolatedData.trailHistory || []}
            />
          </div>

          {/* COMMITTED ACTION OVERLAY - ALWAYS VISIBLE */}
          {actionDecision && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
              style={{
                position: 'absolute',
                bottom: 80,
                left: '50%',
                transform: 'translateX(-50%)',
                background: 'rgba(5, 6, 7, 0.85)',
                backdropFilter: 'blur(8px)',
                border: `1px solid ${getStateColor(phase)}44`,
                borderRadius: '8px',
                padding: '16px 24px',
                maxWidth: '400px',
                textAlign: 'center',
                zIndex: 30,
              }}
            >
              <div
                style={{
                  fontSize: '14px',
                  fontWeight: 600,
                  color: '#e2e8f0',
                  marginBottom: '8px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}
              >
                {actionDecision.primaryAction?.label || 'Monitoring'}
              </div>
              <div
                style={{
                  fontSize: '12px',
                  color: '#94a3b8',
                  lineHeight: '1.5',
                }}
              >
                {actionDecision.primaryAction?.outcome || 'Coherence recovery expected'}
              </div>
            </motion.div>
          )}

          {/* SUBSYSTEM CONTEXT - INLINE, NOT BOXED */}
          <div
            style={{
              position: 'absolute',
              bottom: 20,
              left: '50%',
              transform: 'translateX(-50%)',
              display: 'flex',
              gap: '28px',
              fontSize: '12px',
              color: '#94a3b8',
              zIndex: 25,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>Airflow</span>
              <span style={{ fontSize: '16px', color: '#7e9f2e' }}>↑</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>Climate</span>
              <span style={{ fontSize: '16px', color: '#d8a35d' }}>→</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>Irrigation</span>
              <span style={{ fontSize: '16px', color: '#7e9f2e' }}>↓</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>Plant Stress</span>
              <span style={{ fontSize: '16px', color: '#c94c4c' }}>↑</span>
            </div>
          </div>
        </div>

        {/* CONSEQUENCE DISPLAY - WHEN RELEVANT */}
        {(phase === 'Instability forming' || phase === 'Critical') && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            style={{
              background: `${getStateColor(phase)}11`,
              borderTop: `1px solid ${getStateColor(phase)}44`,
              padding: '16px 24px',
              textAlign: 'center',
              fontSize: '13px',
              color: getStateColor(phase),
              fontWeight: 500,
              lineHeight: '1.6',
              zIndex: 20,
            }}
          >
            <div style={{ marginBottom: '4px' }}>No action:</div>
            <div style={{ color: 'rgba(227, 232, 240, 0.7)' }}>
              Cascade continues
            </div>
            <div style={{ color: getStateColor(phase), marginTop: '4px' }}>
              Stability collapse follows
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
