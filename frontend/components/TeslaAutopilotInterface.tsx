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

type OperationalState =
  | 'Nominal'
  | 'Watchlist'
  | 'Localized deviation'
  | 'Propagating instability'
  | 'Action required'

interface OperationalAssessment {
  state: OperationalState
  propagationStrength: number
  originRoom?: RoomStatus
  affectedRooms: RoomStatus[]
}

const getOperationalColor = (state: OperationalState): string => {
  if (state === 'Nominal') return '#22c55e'
  if (state === 'Watchlist') return '#84cc16'
  if (state === 'Localized deviation') return '#eab308'
  if (state === 'Propagating instability') return '#f97316'
  return '#ef4444'
}

const getZoneStateLabel = (room: RoomStatus, assessment: OperationalAssessment): string => {
  if (assessment.originRoom?.id === room.id) return 'Drift forming · origin'
  if (assessment.affectedRooms.find(r => r.id === room.id)) return 'At risk · affected'
  if (room.driftContribution > 0.2) return 'Watchlist'
  if (room.driftContribution > 0.1) return 'Stable / linked'
  return 'Stable'
}

const deriveOperationalAssessment = (
  drift: number,
  stability: number,
  coherence: number,
  rooms: RoomStatus[] = []
): OperationalAssessment => {
  const sorted = [...rooms].sort((a, b) => b.driftContribution - a.driftContribution)
  const originRoom = sorted[0]
  const affectedRooms = sorted.filter((r, idx) => idx > 0 && r.driftContribution >= 0.35)
  const highDriftCount = sorted.filter(r => r.driftContribution >= 0.35).length
  const propagationStrength = Math.max(
    0,
    Math.min(
      1,
      (1 - coherence) * 0.55 + (1 - stability) * 0.25 + Math.min(0.2, highDriftCount * 0.08)
    )
  )

  let state: OperationalState = 'Nominal'
  if (drift < 0.2 && stability > 0.75 && coherence > 0.75) {
    state = 'Nominal'
  } else if (highDriftCount <= 1 && propagationStrength < 0.35) {
    state = drift > 0.28 ? 'Localized deviation' : 'Watchlist'
  } else if (highDriftCount <= 2 && propagationStrength < 0.62) {
    state = 'Propagating instability'
  } else {
    state = 'Action required'
  }

  return { state, propagationStrength, originRoom, affectedRooms }
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
        <span style={{ fontSize: '10px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700 }}>
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
  const assessment = deriveOperationalAssessment(
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    rooms
  )
  const operationalColor = getOperationalColor(assessment.state)
  const stateColor = getStateColor(phase)

  const geometryInterpretation = assessment.state === 'Localized deviation'
    ? `Low coherence stress is isolated in ${assessment.originRoom?.shortName ?? 'one zone'}; inspect locally, no facility-wide intervention yet.`
    : assessment.state === 'Propagating instability'
      ? `Coherence is weakening across linked vertices with ${assessment.affectedRooms.length + 1} zones involved; intervene in origin and nearest neighbors.`
      : assessment.state === 'Action required'
        ? 'Geometry collapse indicates multi-vertex coupling failure; coordinated system intervention is required now.'
        : assessment.state === 'Watchlist'
          ? `Early variance concentrated in ${assessment.originRoom?.shortName ?? 'a single zone'}; keep system in watchlist and validate trend continuity.`
          : 'Vertex coherence remains balanced; continue nominal monitoring and optimization only.'

  const doNow = assessment.state === 'Nominal'
    ? 'Maintain nominal control loop (optimization only)'
    : assessment.state === 'Watchlist'
      ? `Monitor and inspect ${assessment.originRoom?.shortName ?? 'lead zone'}`
      : assessment.state === 'Localized deviation'
        ? `Inspect ${assessment.originRoom?.shortName ?? 'origin zone'} and validate local airflow`
        : assessment.state === 'Propagating instability'
          ? `Stabilize ${assessment.originRoom?.shortName ?? 'origin zone'} and rebalance adjacent zones`
          : 'Redistribute resources across affected zones now'

  const whyNow = assessment.state === 'Action required'
    ? `Propagation strength ${(assessment.propagationStrength * 100).toFixed(0)}% with multi-zone coupling visible.`
    : assessment.state === 'Propagating instability'
      ? `Spread is directional from ${assessment.originRoom?.shortName ?? 'origin'} into ${assessment.affectedRooms.length} downstream zone(s).`
      : assessment.state === 'Localized deviation'
        ? `Single-zone drift origin with weak spread (${(assessment.propagationStrength * 100).toFixed(0)}% propagation).`
        : 'No correction threshold crossed; this is observation-first monitoring.'

  const whatHappensNext = assessment.state === 'Nominal'
    ? 'System remains in nominal envelope with minor variance.'
    : assessment.state === 'Watchlist'
      ? 'Drift may normalize or become localized deviation within the next cycle.'
      : assessment.state === 'Localized deviation'
        ? 'Without local inspection, drift can begin coupling into linked zones.'
        : assessment.state === 'Propagating instability'
          ? 'Spread likely reaches additional neighbors, reducing coherence further.'
          : 'Unchecked spread can trigger facility-wide instability and control loss.'

  const expectedOutcome = assessment.state === 'Nominal'
    ? 'Operational efficiency preserved without corrective intervention.'
    : assessment.state === 'Watchlist'
      ? 'Higher confidence classification of trend with minimal disruption.'
      : assessment.state === 'Localized deviation'
        ? 'Containment in origin zone and prevention of unnecessary system intervention.'
        : assessment.state === 'Propagating instability'
          ? 'Propagation slows and system coherence can recover in staged fashion.'
          : 'Risk reduction across coupled zones with path back to propagating state.'

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
            animate={{ color: operationalColor }}
            style={{ fontWeight: 700, fontSize: '11px', letterSpacing: '1.2px', textTransform: 'uppercase', flexShrink: 0 }}
          >
            {assessment.state}
          </motion.div>

          <div style={{ color: '#475569', fontSize: '11px', flexShrink: 0 }}>
            Confidence: <span style={{ color: '#64748b' }}>{confidentLabel}</span>
          </div>

          <motion.div
            animate={{ color: (consequenceState.timeToImpact ?? 15) < 5 ? '#ef4444' : '#475569' }}
            style={{ fontSize: '11px', fontVariantNumeric: 'tabular-nums', flexShrink: 0 }}
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
            <span style={{ fontSize: '10px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase' }}>Speed</span>
            <input
              type="range"
              min={0.5}
              max={3}
              step={0.5}
              value={speed}
              onChange={e => setSpeed(Number(e.target.value))}
              style={{ width: '56px', height: '3px', accentColor: '#7e9f2e', cursor: 'pointer' }}
            />
            <span style={{ fontSize: '11px', color: '#64748b', minWidth: '22px', fontVariantNumeric: 'tabular-nums' }}>{speed}x</span>
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
              >â€¹</button>
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
              >â€º</button>
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
              fontSize: '11px',
              fontWeight: 600,
              letterSpacing: '0.6px',
              textTransform: 'uppercase',
              transition: 'all 0.2s',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = `${stateColor}28` }}
            onMouseLeave={e => { e.currentTarget.style.background = isPlaying ? `${stateColor}1a` : 'rgba(126,159,46,0.12)' }}
          >
            {isPlaying ? 'â™â™ Pause' : 'â–¶ Play'}
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
            <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700 }}>
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
              <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '9px' }}>
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
                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '5px' }}>
                  Driver
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', lineHeight: 1.4 }}>
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

            {/* COMMITTED ACTION OVERLAY */}
            {actionDecision && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                style={{
                  position: 'absolute',
                  bottom: 18,
                  right: 18,
                  left: 'auto',
                  transform: 'none',
                  background: 'rgba(5,6,7,0.9)',
                  backdropFilter: 'blur(12px)',
                  border: `1px solid ${stateColor}33`,
                  borderTop: `2px solid ${stateColor}66`,
                  borderRadius: '8px',
                  padding: '12px 16px',
                  maxWidth: 'min(320px, 38vw)',
                  minWidth: '220px',
                  textAlign: 'center',
                  zIndex: 30,
                  pointerEvents: 'none',
                }}
              >
                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', marginBottom: '5px', fontWeight: 700 }}>
                  Decision State
                </div>
                <div style={{ fontSize: '14px', fontWeight: 700, color: operationalColor, marginBottom: '5px', letterSpacing: '0.3px' }}>
                  {assessment.state}
                </div>
                <div style={{ fontSize: '11px', color: '#475569', lineHeight: '1.5' }}>
                  {doNow}
                </div>
                {geometryInterpretation && (
                  <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: `1px solid rgba(255,255,255,0.06)`, fontSize: '10px', color: '#334155', fontStyle: 'italic', lineHeight: 1.4 }}>
                    {geometryInterpretation}
                  </div>
                )}
              </motion.div>
            )}

            {/* SUBSYSTEM CONTEXT */}
            <div
              style={{
                position: 'absolute',
                bottom: 16,
                left: '50%',
                transform: 'translateX(-50%)',
                display: 'flex',
                gap: '22px',
                fontSize: '11px',
                color: '#334155',
                zIndex: 25,
              }}
            >
              {[
                { label: 'Airflow', color: '#7e9f2e', dir: 'â†‘' },
                { label: 'Climate', color: '#d8a35d', dir: 'â†’' },
                { label: 'Irrigation', color: '#7e9f2e', dir: 'â†“' },
                { label: 'Plant Stress', color: '#c94c4c', dir: 'â†‘' },
              ].map(s => (
                <div key={s.label} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span>{s.label}</span>
                  <span style={{ fontSize: '14px', color: s.color, lineHeight: 1 }}>{s.dir}</span>
                </div>
              ))}
            </div>
          </div>

          {/* RIGHT STATUS PANEL */}
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
                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '10px' }}>
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
                          <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b' }}>{room.shortName}</div>
                          <div style={{ fontSize: '10px', color: '#334155', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {getZoneStateLabel(room, assessment)}
                          </div>
                        </div>
                        <div style={{ fontSize: '10px', color: rc, fontWeight: 700, fontVariantNumeric: 'tabular-nums', flexShrink: 0 }}>
                          {Math.round(room.driftContribution * 100)}%
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Operational intelligence */}
            {(intelligence || actionDecision) && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700 }}>
                  Action Intelligence
                </div>

                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700 }}>
                  1. Do now
                </div>
                <div style={{ fontSize: '11px', color: operationalColor, lineHeight: '1.45', fontWeight: 600 }}>{doNow}</div>

                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700 }}>
                  2. Why now
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', lineHeight: '1.5' }}>{whyNow}</div>

                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700 }}>
                  3. What we see
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', lineHeight: '1.5' }}>
                  {intelligence?.explanation || actionDecision?.decisionSummary}
                </div>

                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700 }}>
                  4. If unchanged
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', lineHeight: '1.5' }}>{whatHappensNext}</div>

                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700 }}>
                  5. Expected outcome
                </div>
                <div style={{
                  fontSize: '11px',
                  color: '#94a3b8',
                  lineHeight: '1.5',
                  padding: '6px 8px',
                  borderLeft: `2px solid ${operationalColor}55`,
                  borderRadius: '0 4px 4px 0',
                  background: `${operationalColor}0c`,
                }}>
                  {expectedOutcome}
                </div>

                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700 }}>
                  Relationship View
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', lineHeight: '1.5' }}>
                  {assessment.state === 'Localized deviation'
                    ? `Origin: ${assessment.originRoom?.shortName ?? 'N/A'} · downstream impact low`
                    : `Origin: ${assessment.originRoom?.shortName ?? 'N/A'} · affected neighbors: ${assessment.affectedRooms.map(r => r.shortName).join(', ') || 'none yet'}`}
                </div>

                <div style={{ fontSize: '10px', color: '#334155', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700 }}>
                  Geometry interpretation
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', lineHeight: '1.5' }}>
                  {geometryInterpretation}
                </div>
              </div>
            )}
          </motion.div>
        </div>

        {/* CONSEQUENCE DISPLAY */}
        {(assessment.state === 'Propagating instability' || assessment.state === 'Action required') && (
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
            <span style={{ color: 'rgba(148,163,184,0.4)' }}>
              {assessment.state === 'Action required' ? 'Coupled spread accelerates' : 'Localized spread broadens'}
            </span>
            <span style={{ color: operationalColor, fontWeight: 600 }}>
              {assessment.state === 'Action required' ? 'Facility instability risk rises' : 'More zones move to at-risk state'}
            </span>
          </motion.div>
        )}
      </div>
    </div>
  )
}
