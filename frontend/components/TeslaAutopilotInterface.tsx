'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { TetrahedronField } from './TetrahedronField'
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

export interface RoomStatus {
  id: string
  shortName: string
  status: 'optimal' | 'warning' | 'critical'
  driftContribution: number
  behavioralState: string
}

export interface IntelligenceData {
  explanation: string
  operatorFocus: string
  pathOutlook: string
  primaryDriver: string
}

interface TeslaAutopilotInterfaceProps {
  systemData?: SystemData
  onTogglePlay?: (isPlaying: boolean) => void
  onSpeedChange?: (speed: number) => void
  rooms?: RoomStatus[]
  intelligence?: IntelligenceData
  onStepScenario?: (delta: number) => void
}

const getStateLabel = (phase: string): string => {
  const map: Record<string, string> = {
    'Stable': 'Stable',
    'Drift forming': 'Drift',
    'Instability forming': 'Instability',
    'Critical': 'Critical',
  }
  return map[phase] || phase
}

const getStateColor = (phase: string): string => {
  const colors: Record<string, string> = {
    'Stable': '#22c55e',
    'Drift forming': '#eab308',
    'Instability forming': '#f97316',
    'Critical': '#ef4444',
  }
  return colors[phase] || '#9ca3af'
}

const getRoomStatusColor = (status: 'optimal' | 'warning' | 'critical'): string => {
  return status === 'critical' ? '#ef4444' : status === 'warning' ? '#eab308' : '#22c55e'
}

function MetricGauge({
  label,
  value,
  color,
}: {
  label: string
  value: number
  color: string
}) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '5px' }}>
        <span style={{ fontSize: '12px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700 }}>
          {label}
        </span>
        <span style={{ fontSize: '13px', fontWeight: 700, color, fontVariantNumeric: 'tabular-nums' }}>
          {Math.round(value * 100)}
        </span>
      </div>
      <div style={{ height: '3px', background: 'rgba(255,255,255,0.06)', borderRadius: '2px', overflow: 'hidden' }}>
        <motion.div
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          style={{ height: '100%', background: color, borderRadius: '2px', boxShadow: `0 0 5px ${color}88` }}
        />
      </div>
    </div>
  )
}

export function TeslaAutopilotInterface({
  systemData,
  onTogglePlay,
  rooms,
  intelligence,
  onStepScenario,
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
        timingSensitivity: decision.primaryAction.timingSensitivity,
      }])
    }
  }, [phase, diagnostics.dominantDriver, interpolatedData.interpolatedDrift, interpolatedData.interpolatedStability, interpolatedData.interpolatedCoherence, diagnostics.confidence, consequenceState.timeToImpact, driftAcceleration, outcomeState.noActionConsequence])

  const handleTogglePlay = () => {
    setIsPlaying(!isPlaying)
    onTogglePlay?.(!isPlaying)
  }

  const confidentLabel = diagnostics.confidence === 'high' ? 'High' : diagnostics.confidence === 'moderate' ? 'Moderate' : 'Low'
  const stateColor = getStateColor(phase)

  const urgencyLevel = phase === 'Critical' ? 5
    : phase === 'Instability forming' ? 4
    : phase === 'Drift forming' ? 3
    : 1

  const urgencyColor = urgencyLevel >= 5 ? '#ef4444' : urgencyLevel >= 4 ? '#f97316' : urgencyLevel >= 3 ? '#eab308' : '#22c55e'

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
          opacity: 0.18,
        }}
      />

      {/* TOP STATUS STRIP */}
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
          background: 'linear-gradient(180deg, rgba(5,6,7,0.97) 0%, rgba(5,6,7,0.55) 100%)',
          backdropFilter: 'blur(6px)',
          borderBottom: `1px solid ${stateColor}2a`,
          padding: '0 20px',
          height: '48px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '12px',
          gap: '16px',
        }}
      >
        {/* Left: System identity + status */}
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center', minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '7px', flexShrink: 0 }}>
            <motion.div
              animate={{
                background: stateColor,
                boxShadow: `0 0 ${phase === 'Critical' ? '10px' : '5px'} ${stateColor}`,
              }}
              transition={{ duration: 0.4 }}
              style={{ width: '6px', height: '6px', borderRadius: '50%' }}
            />
            <span style={{ fontWeight: 600, color: '#cbd5e1', fontSize: '12px', letterSpacing: '0.8px', textTransform: 'uppercase' }}>
              Grow Room A / Unit 1
            </span>
          </div>

          <div style={{ width: '1px', height: '14px', background: 'rgba(203,213,225,0.12)', flexShrink: 0 }} />

          <motion.div
            animate={{ color: stateColor }}
            style={{ fontWeight: 700, fontSize: '13px', letterSpacing: '1.2px', textTransform: 'uppercase', flexShrink: 0 }}
          >
            {getStateLabel(phase)}
          </motion.div>

          <div style={{ color: '#475569', fontSize: '13px', flexShrink: 0 }}>
            Confidence: <span style={{ color: '#64748b' }}>{confidentLabel}</span>
          </div>

          <motion.div
            animate={{ color: (consequenceState.timeToImpact ?? 15) < 5 ? '#ef4444' : '#475569' }}
            style={{ fontSize: '13px', fontVariantNumeric: 'tabular-nums', flexShrink: 0 }}
          >
            T-Impact:{' '}
            <span style={{ color: '#64748b' }}>
              {(consequenceState.timeToImpact ?? 15) < 1 ? 'Imminent' : `~${Math.ceil(consequenceState.timeToImpact ?? 15)}c`}
            </span>
          </motion.div>
        </div>

        {/* Right: Controls */}
        <div style={{ display: 'flex', gap: '14px', alignItems: 'center', flexShrink: 0 }}>
          {/* Speed control */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
            <span style={{ fontSize: '12px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase' }}>Speed</span>
            <input
              type="range"
              min={0.5}
              max={3}
              step={0.5}
              value={speed}
              onChange={e => setSpeed(Number(e.target.value))}
              style={{ width: '56px', height: '3px', accentColor: '#7e9f2e', cursor: 'pointer' }}
            />
            <span style={{ fontSize: '13px', color: '#64748b', minWidth: '22px', fontVariantNumeric: 'tabular-nums' }}>{speed}x</span>
          </div>

          {/* Scenario navigation */}
          {onStepScenario && (
            <div style={{ display: 'flex', gap: '3px' }}>
              <button
                onClick={() => onStepScenario(-1)}
                style={{
                  background: 'rgba(100,116,139,0.08)',
                  border: '1px solid rgba(100,116,139,0.25)',
                  color: '#64748b',
                  width: '22px',
                  height: '22px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={e => { e.currentTarget.style.color = '#94a3b8'; e.currentTarget.style.borderColor = 'rgba(100,116,139,0.5)' }}
                onMouseLeave={e => { e.currentTarget.style.color = '#64748b'; e.currentTarget.style.borderColor = 'rgba(100,116,139,0.25)' }}
              >‹</button>
              <button
                onClick={() => onStepScenario(1)}
                style={{
                  background: 'rgba(100,116,139,0.08)',
                  border: '1px solid rgba(100,116,139,0.25)',
                  color: '#64748b',
                  width: '22px',
                  height: '22px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={e => { e.currentTarget.style.color = '#94a3b8'; e.currentTarget.style.borderColor = 'rgba(100,116,139,0.5)' }}
                onMouseLeave={e => { e.currentTarget.style.color = '#64748b'; e.currentTarget.style.borderColor = 'rgba(100,116,139,0.25)' }}
              >›</button>
            </div>
          )}

          <button
            onClick={handleTogglePlay}
            style={{
              background: isPlaying ? `${stateColor}1a` : 'rgba(126,159,46,0.12)',
              border: `1px solid ${isPlaying ? stateColor + '88' : 'rgba(126,159,46,0.35)'}`,
              color: isPlaying ? stateColor : '#7e9f2e',
              padding: '4px 14px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 600,
              letterSpacing: '0.6px',
              textTransform: 'uppercase',
              transition: 'all 0.2s',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = `${stateColor}28` }}
            onMouseLeave={e => { e.currentTarget.style.background = isPlaying ? `${stateColor}1a` : 'rgba(126,159,46,0.12)' }}
          >
            {isPlaying ? '❙❙ Pause' : '▶ Play'}
          </button>
        </div>
      </motion.div>

      {/* PHASE PROGRESS BAR */}
      <div style={{ position: 'absolute', top: 48, left: 0, right: 0, height: '2px', background: 'rgba(255,255,255,0.04)', zIndex: 49 }}>
        <motion.div
          animate={{ width: `${phaseProgress * 100}%`, backgroundColor: stateColor }}
          transition={{ duration: 0.12 }}
          style={{ height: '100%', boxShadow: `0 0 6px ${stateColor}88` }}
        />
      </div>

      {/* MAIN CONTENT AREA */}
      <div
        style={{
          position: 'absolute',
          top: 50,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>

          {/* LEFT METRIC PANEL */}
          <motion.div
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            style={{
              width: '172px',
              flexShrink: 0,
              padding: '18px 14px',
              borderRight: '1px solid rgba(255,255,255,0.04)',
              display: 'flex',
              flexDirection: 'column',
              gap: '18px',
              background: 'linear-gradient(90deg, rgba(5,6,7,0.9) 0%, rgba(5,6,7,0.2) 100%)',
            }}
          >
            <div style={{ fontSize: '12px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700 }}>
              System Metrics
            </div>

            <MetricGauge
              label="Drift"
              value={interpolatedData.interpolatedDrift}
              color={interpolatedData.interpolatedDrift > 0.6 ? '#ef4444' : interpolatedData.interpolatedDrift > 0.3 ? '#eab308' : '#22c55e'}
            />
            <MetricGauge
              label="Stability"
              value={interpolatedData.interpolatedStability}
              color={interpolatedData.interpolatedStability < 0.4 ? '#ef4444' : interpolatedData.interpolatedStability < 0.6 ? '#eab308' : '#22c55e'}
            />
            <MetricGauge
              label="Coherence"
              value={interpolatedData.interpolatedCoherence}
              color={interpolatedData.interpolatedCoherence < 0.4 ? '#ef4444' : interpolatedData.interpolatedCoherence < 0.6 ? '#eab308' : '#06b6d4'}
            />
            <MetricGauge
              label="Confidence"
              value={interpolatedData.interpolatedConfidence}
              color="#818cf8"
            />

            {/* Urgency bar */}
            <div>
              <div style={{ fontSize: '12px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '9px' }}>
                Urgency
              </div>
              <div style={{ display: 'flex', gap: '4px' }}>
                {[1, 2, 3, 4, 5].map(dot => (
                  <motion.div
                    key={dot}
                    animate={{
                      background: dot <= urgencyLevel ? urgencyColor : 'rgba(255,255,255,0.07)',
                      boxShadow: dot <= urgencyLevel && urgencyLevel >= 4 ? `0 0 5px ${urgencyColor}` : 'none',
                    }}
                    transition={{ duration: 0.3 }}
                    style={{ flex: 1, height: '5px', borderRadius: '2px' }}
                  />
                ))}
              </div>
            </div>

            {/* Dominant driver */}
            {diagnostics.dominantDriver && (
              <div>
                <div style={{ fontSize: '12px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '5px' }}>
                  Driver
                </div>
                <div style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.4 }}>
                  {diagnostics.dominantDriver}
                </div>
              </div>
            )}
          </motion.div>

          {/* CENTER: TETRAHEDRON */}
          <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
            <TetrahedronField
              data={interpolatedData}
              phaseProgress={phaseProgress}
              phase={phase}
              escalationLevel={consequenceState.escalationLevel}
              hasThresholdCrossed={consequenceState.hasThresholdCrossed}
              isApproachingFailure={consequenceState.isApproachingFailure}
            />

            {/* SUBSYSTEM CONTEXT */}
            <div
              style={{
                position: 'absolute',
                bottom: 16,
                left: '50%',
                transform: 'translateX(-50%)',
                display: 'flex',
                gap: '22px',
                fontSize: '13px',
                color: '#334155',
                zIndex: 25,
              }}
            >
              {[
                { label: 'Airflow', color: '#7e9f2e', dir: '↑' },
                { label: 'Climate', color: '#d8a35d', dir: '→' },
                { label: 'Irrigation', color: '#7e9f2e', dir: '↓' },
                { label: 'Plant Stress', color: '#c94c4c', dir: '↑' },
              ].map(s => (
                <div key={s.label} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span>{s.label}</span>
                  <span style={{ fontSize: '14px', color: s.color, lineHeight: 1 }}>{s.dir}</span>
                </div>
              ))}
            </div>
          </div>
          <motion.div
            initial={{ opacity: 0, x: 16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            style={{
              width: '196px',
              flexShrink: 0,
              padding: '18px 14px',
              borderLeft: '1px solid rgba(255,255,255,0.04)',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
              background: 'linear-gradient(270deg, rgba(5,6,7,0.9) 0%, rgba(5,6,7,0.2) 100%)',
              overflowY: 'auto',
            }}
          >
            {/* Zone status */}
            {rooms && rooms.length > 0 && (
              <div>
                <div style={{ fontSize: '12px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '10px' }}>
                  Zone Status
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                  {rooms.map(room => {
                    const rc = getRoomStatusColor(room.status)
                    return (
                      <div
                        key={room.id}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '7px',
                          padding: '5px 7px',
                          borderRadius: '4px',
                          background: `${rc}07`,
                          border: `1px solid ${rc}1a`,
                          transition: 'all 0.3s',
                        }}
                      >
                        <motion.div
                          animate={{
                            background: rc,
                            boxShadow: room.status === 'critical' ? `0 0 5px ${rc}` : 'none',
                          }}
                          style={{ width: '5px', height: '5px', borderRadius: '50%', flexShrink: 0 }}
                        />
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: '13px', fontWeight: 600, color: '#64748b' }}>{room.shortName}</div>
                          <div style={{ fontSize: '12px', color: '#334155', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {room.behavioralState}
                          </div>
                        </div>
                        <div style={{ fontSize: '12px', color: rc, fontWeight: 700, fontVariantNumeric: 'tabular-nums', flexShrink: 0 }}>
                          {Math.round(room.driftContribution * 100)}%
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* System intelligence */}
            {intelligence && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ fontSize: '12px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700 }}>
                  System Intel
                </div>
                <div style={{ fontSize: '13px', color: '#3d4e61', lineHeight: '1.65' }}>
                  {intelligence.explanation}
                </div>
                <div>
                  <div style={{ fontSize: '12px', color: '#334155', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '5px' }}>
                    Operator Focus
                  </div>
                  <div style={{
                    fontSize: '13px',
                    color: stateColor,
                    lineHeight: '1.5',
                    padding: '6px 8px',
                    borderLeft: `2px solid ${stateColor}44`,
                    borderRadius: '0 4px 4px 0',
                    background: `${stateColor}08`,
                  }}>
                    {intelligence.operatorFocus}
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </div>

        {/* CONSEQUENCE DISPLAY */}
        {(phase === 'Instability forming' || phase === 'Critical') && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            style={{
              background: `${stateColor}09`,
              borderTop: `1px solid ${stateColor}2a`,
              padding: '10px 24px',
              display: 'flex',
              justifyContent: 'center',
              gap: '28px',
              alignItems: 'center',
              fontSize: '12px',
              zIndex: 20,
            }}
          >
            <span style={{ color: '#334155' }}>No action:</span>
            <span style={{ color: 'rgba(148,163,184,0.4)' }}>Cascade continues</span>
            <span style={{ color: stateColor, fontWeight: 600 }}>Stability collapse follows</span>
          </motion.div>
        )}
      </div>

      {/* ACTION CARD — direct child of root, top-left corner */}
      {actionDecision && (
        <motion.div
          initial={{ opacity: 0, x: -16, y: -8 }}
          animate={{ opacity: 1, x: 0, y: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          style={{
            position: 'absolute',
            top: '60px',
            left: '16px',
            width: '168px',
            background: `linear-gradient(135deg, rgba(5,6,7,0.96) 0%, rgba(8,10,14,0.91) 100%)`,
            backdropFilter: 'blur(14px)',
            border: `1.5px solid ${stateColor}3a`,
            borderRadius: '10px',
            padding: '14px 16px',
            boxShadow: `0 8px 28px rgba(0,0,0,0.6), 0 0 20px ${stateColor}15, inset 0 1px 0 ${stateColor}2a`,
            zIndex: 50,
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <motion.div
                animate={{ background: stateColor, boxShadow: `0 0 6px ${stateColor}` }}
                style={{ width: '6px', height: '6px', borderRadius: '50%', flexShrink: 0 }}
              />
              <span style={{ fontSize: '12px', color: '#94a3b8', letterSpacing: '0.7px', textTransform: 'uppercase', fontWeight: 700 }}>
                Action
              </span>
            </div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: stateColor, lineHeight: 1.4, wordBreak: 'break-word' }}>
              {actionDecision.primaryAction?.label || 'Monitoring'}
            </div>
            <div style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: 1.5, wordBreak: 'break-word' }}>
              {actionDecision.primaryAction?.outcome?.primary || 'Coherence recovery expected'}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}
