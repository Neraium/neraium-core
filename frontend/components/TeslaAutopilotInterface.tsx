'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
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

type OperatorState = 'NORMAL' | 'WATCH' | 'INTERVENE'

const getOperatorState = (operationalState: OperationalState): OperatorState => {
  if (operationalState === 'Nominal') return 'NORMAL'
  if (operationalState === 'Watchlist' || operationalState === 'Localized deviation') return 'WATCH'
  return 'INTERVENE'
}

const getOperatorStateColor = (state: OperatorState): string => {
  if (state === 'NORMAL') return '#22c55e'
  if (state === 'WATCH') return '#eab308'
  return '#ef4444'
}

const getProgressionLabel = (
  originRoom: RoomStatus | undefined,
  affectedRooms: RoomStatus[],
  propagationStrength: number
): string => {
  if (propagationStrength > 0.7) return 'Spreading'
  if (propagationStrength > 0.3) return 'Increasing'
  return 'Stable'
}

const getFallbackZones = (): RoomStatus[] => [
  { id: 'veg-a', shortName: 'VEG-A', status: 'optimal', driftContribution: 0, behavioralState: 'normal' },
  { id: 'veg-b', shortName: 'VEG-B', status: 'optimal', driftContribution: 0, behavioralState: 'normal' },
  { id: 'flow-a', shortName: 'FLOW-A', status: 'optimal', driftContribution: 0, behavioralState: 'normal' },
  { id: 'flow-b', shortName: 'FLOW-B', status: 'warning', driftContribution: 0.4, behavioralState: 'watch' },
  { id: 'flow-c', shortName: 'FLOW-C', status: 'optimal', driftContribution: 0, behavioralState: 'normal' },
]

const getOperatorStateFromZones = (zones: RoomStatus[]): OperatorState => {
  const hasCritical = zones.some(z => z.status === 'critical')
  const hasWarning = zones.some(z => z.status === 'warning')
  if (hasCritical) return 'INTERVENE'
  if (hasWarning) return 'WATCH'
  return 'NORMAL'
}

interface RoomMetrics {
  localDrift: number
  couplingStrength: number
  influence: number
  description: string
}

const computeRoomMetrics = (room: RoomStatus, allRooms: RoomStatus[]): RoomMetrics => {
  // Local Drift: room's own drift contribution (0-100)
  const localDrift = Math.round(Math.min(100, room.driftContribution * 200))

  // Coupling Strength: how much other rooms' states affect this room
  // Higher if nearby rooms have significant drift
  const otherDrifts = allRooms
    .filter(r => r.id !== room.id)
    .map(r => r.driftContribution)
  const couplingStrength = otherDrifts.length > 0
    ? Math.round((Math.max(...otherDrifts, 0) * 0.7 + Math.min(...otherDrifts, 0) * 0.3) * 100)
    : 0

  // Influence: how much this room affects others
  // Higher drift contribution = higher influence on system
  const influence = Math.round(room.driftContribution * 100)

  // Generate descriptive text based on room state
  let description = ''
  if (room.status === 'optimal' && localDrift === 0) {
    description = 'Stable conditions'
  } else if (room.status === 'optimal' && localDrift < 15) {
    description = 'Nominal with minor variations'
  } else if (room.status === 'warning') {
    description = 'Minor deviation detected'
  } else if (room.status === 'critical') {
    if (influence > 50) {
      description = 'Driving structural instability'
    } else {
      description = 'Critical local instability'
    }
  } else {
    description = `Status: ${room.behavioralState || 'monitoring'}`
  }

  return {
    localDrift: Math.max(0, Math.min(100, localDrift)),
    couplingStrength: Math.max(0, Math.min(100, couplingStrength)),
    influence: Math.max(0, Math.min(100, influence)),
    description,
  }
}

interface TrajectoryInfo {
  label: string
  affectedRoomIds: string[]
}

const computeTrajectory = (
  drift: number,
  stability: number,
  coherence: number,
  propagationStrength: number,
  allRooms: RoomStatus[]
): TrajectoryInfo => {
  // Determine trajectory direction based on system state
  const isImproving = stability > 0.65 && drift < 0.3
  const isContained = propagationStrength < 0.3 && drift < 0.5
  const isSpreading = propagationStrength > 0.3 && propagationStrength < 0.7
  const isEscalating = propagationStrength > 0.7 || (drift > 0.6 && coherence < 0.4)
  const hasLocalIssue = drift > 0.3 && propagationStrength < 0.5

  let label = 'Stabilizing'
  let affectedRoomIds: string[] = []

  if (isImproving) {
    label = 'Stabilizing'
  } else if (isEscalating) {
    label = 'Escalating system-wide'
    // Identify rooms likely to be affected: those with highest coupling or lower stability
    affectedRoomIds = allRooms
      .sort((a, b) => b.driftContribution - a.driftContribution)
      .slice(0, Math.max(2, Math.floor(allRooms.length * 0.4)))
      .map(r => r.id)
  } else if (isSpreading && hasLocalIssue) {
    label = 'Spreading to adjacent rooms'
    // Identify rooms not yet affected but in propagation path
    const criticalRooms = allRooms.filter(r => r.status !== 'optimal').map(r => r.id)
    affectedRoomIds = allRooms
      .filter(r => !criticalRooms.includes(r.id) && r.status === 'optimal')
      .sort((a, b) => b.driftContribution - a.driftContribution)
      .slice(0, Math.ceil(allRooms.length * 0.3))
      .map(r => r.id)
  } else if (isSpreading) {
    label = 'Spreading to adjacent rooms'
  } else if (isContained) {
    label = 'Contained'
  }

  return { label, affectedRoomIds }
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
  const [showDetails, setShowDetails] = useState(false)
  const [selectedRoom, setSelectedRoom] = useState<RoomStatus | null>(null)
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

  const activeZones = rooms && rooms.length > 0 ? rooms : getFallbackZones()

  const assessment = deriveOperationalAssessment(
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    activeZones
  )

  const operatorState = getOperatorStateFromZones(activeZones)
  const operatorColor = getOperatorStateColor(operatorState)
  const progression = getProgressionLabel(assessment.originRoom, assessment.affectedRooms, assessment.propagationStrength)

  const trajectory = computeTrajectory(
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    assessment.propagationStrength,
    activeZones
  )

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
        display: 'flex',
        flexDirection: 'column',
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

      {/* LAYER 1: TOP BANNER */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        style={{
          background: 'linear-gradient(180deg, rgba(5,6,7,0.97) 0%, rgba(5,6,7,0.55) 100%)',
          backdropFilter: 'blur(6px)',
          borderBottom: `2px solid ${operatorColor}`,
          padding: '24px 32px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '32px',
          zIndex: 50,
        }}
      >
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '16px' }}>
            <motion.div
              animate={{
                background: operatorColor,
                boxShadow: `0 0 ${operatorState === 'INTERVENE' ? '12px' : '6px'} ${operatorColor}`,
              }}
              transition={{ duration: 0.4 }}
              style={{ width: '8px', height: '8px', borderRadius: '50%' }}
            />
            <span style={{ fontSize: '32px', fontWeight: 700, letterSpacing: '1.2px', textTransform: 'uppercase', color: operatorColor }}>
              SYSTEM {operatorState}
            </span>
          </div>

          {/* Zone indicators */}
          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
            {operatorState === 'WATCH' && assessment.originRoom && (
              <div style={{ fontSize: '14px', color: '#eab308', letterSpacing: '0.5px' }}>
                WATCH: <span style={{ fontWeight: 700 }}>{assessment.originRoom.shortName}</span>
              </div>
            )}
            {operatorState === 'INTERVENE' && assessment.originRoom && (
              <div style={{ fontSize: '14px', color: '#ef4444', letterSpacing: '0.5px' }}>
                INTERVENE: <span style={{ fontWeight: 700 }}>{assessment.originRoom.shortName}</span>
              </div>
            )}
            {operatorState === 'INTERVENE' && assessment.affectedRooms.length > 0 && (
              <div style={{ fontSize: '14px', color: '#ef4444', letterSpacing: '0.5px' }}>
                AFFECTED: <span style={{ fontWeight: 700 }}>{assessment.affectedRooms.map(r => r.shortName).join(', ')}</span>
              </div>
            )}
          </div>
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '11px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase' }}>Speed</span>
            <input
              type="range"
              min={0.5}
              max={3}
              step={0.5}
              value={speed}
              onChange={e => setSpeed(Number(e.target.value))}
              style={{ width: '60px', height: '3px', accentColor: '#7e9f2e', cursor: 'pointer' }}
            />
            <span style={{ fontSize: '11px', color: '#64748b', minWidth: '24px', fontVariantNumeric: 'tabular-nums' }}>{speed}x</span>
          </div>

          {onStepScenario && (
            <div style={{ display: 'flex', gap: '4px' }}>
              <button
                onClick={() => onStepScenario(-1)}
                style={{
                  background: 'rgba(100,116,139,0.08)',
                  border: '1px solid rgba(100,116,139,0.25)',
                  color: '#64748b',
                  width: '28px',
                  height: '28px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px',
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
                  width: '28px',
                  height: '28px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px',
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
              background: isPlaying ? `${operatorColor}1a` : 'rgba(126,159,46,0.12)',
              border: `1px solid ${isPlaying ? operatorColor + '88' : 'rgba(126,159,46,0.35)'}`,
              color: isPlaying ? operatorColor : '#7e9f2e',
              padding: '6px 16px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '11px',
              fontWeight: 600,
              letterSpacing: '0.6px',
              textTransform: 'uppercase',
              transition: 'all 0.2s',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = `${operatorColor}28` }}
            onMouseLeave={e => { e.currentTarget.style.background = isPlaying ? `${operatorColor}1a` : 'rgba(126,159,46,0.12)' }}
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
        </div>
      </motion.div>

      {/* LAYER 1: ZONE GRID */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        style={{
          flex: 1,
          padding: '24px 32px',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '16px',
          overflowY: 'auto',
        }}
      >
        {activeZones.map(room => {
          const roomColor = getRoomStatusColor(room.status)
          const isOrigin = assessment.originRoom?.id === room.id
          const isAffected = assessment.affectedRooms.some(r => r.id === room.id)
          const isSelected = selectedRoom?.id === room.id
          const isFutureAffected = trajectory.affectedRoomIds.includes(room.id) && !isOrigin && !isAffected

          return (
            <motion.div
              key={room.id}
              layout
              onClick={() => setSelectedRoom(isSelected ? null : room)}
              animate={{
                borderColor: isSelected
                  ? '#06b6d4'
                  : isOrigin || isAffected ? roomColor : 'rgba(255,255,255,0.08)',
                background: isSelected
                  ? '#06b6d412'
                  : isOrigin || isAffected ? `${roomColor}12` : isFutureAffected ? 'rgba(245,158,11,0.08)' : 'rgba(255,255,255,0.02)',
                scale: isSelected ? 1.05 : 1,
              }}
              transition={{ duration: 0.3 }}
              style={{
                padding: '16px',
                border: `2px solid ${roomColor}`,
                borderRadius: '6px',
                textAlign: 'center',
                cursor: 'pointer',
                boxShadow: isSelected ? '0 0 12px rgba(6,182,212,0.3)' : isFutureAffected ? '0 0 8px rgba(245,158,11,0.2)' : 'none',
                opacity: isFutureAffected ? 0.85 : 1,
              }}
              whileHover={{ scale: 1.02 }}
            >
              <div style={{ fontSize: '16px', fontWeight: 700, color: isSelected ? '#06b6d4' : roomColor, marginBottom: '4px' }}>
                {room.shortName}
              </div>
              <motion.div
                animate={{
                  height: '4px',
                  background: isSelected ? '#06b6d4' : roomColor,
                  opacity: isSelected ? 1 : 0.6,
                }}
                transition={{ duration: 0.3 }}
                style={{ borderRadius: '2px' }}
              />
            </motion.div>
          )
        })}
      </motion.div>

      {/* LAYER 2: INTERVENTION DETAILS (RED ONLY) */}
      <AnimatePresence>
        {operatorState === 'INTERVENE' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.4 }}
            style={{
              background: `linear-gradient(180deg, #ef444412 0%, transparent 100%)`,
              borderTop: '2px solid #ef4444',
              borderBottom: '1px solid #ef4444',
              padding: '20px 32px',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '24px',
              zIndex: 40,
            }}
          >
            <div>
              <div style={{ fontSize: '11px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                INTERVENE
              </div>
              <div style={{ fontSize: '14px', color: '#ef4444', fontWeight: 700 }}>
                {assessment.originRoom?.shortName || 'Primary Zone'}
              </div>
            </div>

            <div>
              <div style={{ fontSize: '11px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                Issue
              </div>
              <div style={{ fontSize: '13px', color: '#e2e8f0' }}>
                {assessment.propagationStrength > 0.7
                  ? 'Critical coherence collapse'
                  : assessment.propagationStrength > 0.4
                    ? 'Multi-zone coupling detected'
                    : 'High drift origin'}
              </div>
            </div>

            <div>
              <div style={{ fontSize: '11px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                Location
              </div>
              <div style={{ fontSize: '13px', color: '#e2e8f0' }}>
                {assessment.originRoom?.shortName}
                {assessment.affectedRooms.length > 0 && ` → ${assessment.affectedRooms.map(r => r.shortName).join(', ')}`}
              </div>
            </div>

            <div>
              <div style={{ fontSize: '11px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                Progression
              </div>
              <motion.div
                animate={{ color: progression === 'Spreading' ? '#ef4444' : progression === 'Increasing' ? '#eab308' : '#22c55e' }}
                style={{ fontSize: '13px', fontWeight: 700 }}
              >
                {progression}
              </motion.div>
            </div>

            <div>
              <div style={{ fontSize: '11px', color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                Trajectory
              </div>
              <motion.div
                animate={{ color: trajectory.label === 'Escalating system-wide' ? '#ef4444' : trajectory.label === 'Spreading to adjacent rooms' ? '#f59e0b' : trajectory.label === 'Stabilizing' ? '#22c55e' : '#64748b' }}
                style={{ fontSize: '13px', fontWeight: 700 }}
              >
                {trajectory.label}
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* LAYER 3: DETAILS DRAWER BUTTON */}
      <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.4 }}
        onClick={() => setShowDetails(!showDetails)}
        style={{
          position: 'absolute',
          bottom: '16px',
          right: '32px',
          padding: '8px 16px',
          background: 'rgba(126,159,46,0.12)',
          border: '1px solid rgba(126,159,46,0.35)',
          color: '#7e9f2e',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '11px',
          fontWeight: 600,
          letterSpacing: '0.6px',
          textTransform: 'uppercase',
          transition: 'all 0.2s',
          zIndex: 30,
        }}
        onMouseEnter={e => { e.currentTarget.style.background = 'rgba(126,159,46,0.25)'; e.currentTarget.style.borderColor = 'rgba(126,159,46,0.6)' }}
        onMouseLeave={e => { e.currentTarget.style.background = 'rgba(126,159,46,0.12)'; e.currentTarget.style.borderColor = 'rgba(126,159,46,0.35)' }}
      >
        {showDetails ? '▼ Hide Details' : '▶ View Details'}
      </motion.button>

      {/* LAYER 3: DETAILS DRAWER */}
      <AnimatePresence>
        {showDetails && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3 }}
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(5,6,7,0.95)',
              backdropFilter: 'blur(8px)',
              zIndex: 100,
              overflowY: 'auto',
              padding: '32px',
            }}
            onClick={() => setShowDetails(false)}
          >
            <motion.div
              onClick={e => e.stopPropagation()}
              style={{
                maxWidth: '800px',
                margin: '0 auto',
                background: 'rgba(15,23,42,0.8)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '8px',
                padding: '24px',
              }}
            >
              {selectedRoom ? (() => {
                const metrics = computeRoomMetrics(selectedRoom, activeZones)
                const roomColor = getRoomStatusColor(selectedRoom.status)
                return (
                  <>
                    <h2 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '8px', color: roomColor, letterSpacing: '1px', textTransform: 'uppercase' }}>
                      {selectedRoom.shortName} Analysis
                    </h2>
                    <div style={{ fontSize: '13px', color: '#cbd5e1', marginBottom: '24px', fontStyle: 'italic' }}>
                      {metrics.description}
                    </div>

                    {/* Room-specific Metrics */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px', marginBottom: '24px' }}>
                      <div>
                        <div style={{ fontSize: '11px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                          Local Drift
                        </div>
                        <div style={{ fontSize: '24px', fontWeight: 700, color: metrics.localDrift > 60 ? '#ef4444' : metrics.localDrift > 30 ? '#eab308' : '#22c55e' }}>
                          {metrics.localDrift}%
                        </div>
                        <div style={{ fontSize: '10px', color: '#64748b', marginTop: '4px' }}>
                          Room-specific
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '11px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                          Coupling Strength
                        </div>
                        <div style={{ fontSize: '24px', fontWeight: 700, color: metrics.couplingStrength > 60 ? '#f59e0b' : metrics.couplingStrength > 30 ? '#eab308' : '#22c55e' }}>
                          {metrics.couplingStrength}%
                        </div>
                        <div style={{ fontSize: '10px', color: '#64748b', marginTop: '4px' }}>
                          External influence
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '11px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                          Influence
                        </div>
                        <div style={{ fontSize: '24px', fontWeight: 700, color: metrics.influence > 60 ? '#ef4444' : metrics.influence > 30 ? '#eab308' : '#22c55e' }}>
                          {metrics.influence}%
                        </div>
                        <div style={{ fontSize: '10px', color: '#64748b', marginTop: '4px' }}>
                          System impact
                        </div>
                      </div>
                    </div>

                    {/* Room Assessment */}
                    <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '16px', marginBottom: '20px' }}>
                      <h3 style={{ fontSize: '12px', fontWeight: 700, color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', marginBottom: '12px' }}>
                        Room Status
                      </h3>
                      <div style={{ fontSize: '13px', color: '#cbd5e1', lineHeight: '1.6' }}>
                        Current State: <span style={{ color: roomColor, fontWeight: 700, textTransform: 'capitalize' }}>{selectedRoom.status}</span>
                      </div>
                      <div style={{ fontSize: '13px', color: '#cbd5e1', lineHeight: '1.6', marginTop: '8px' }}>
                        Behavioral Mode: <span style={{ color: '#e2e8f0', fontWeight: 700 }}>{selectedRoom.behavioralState}</span>
                      </div>
                    </div>

                    {/* System Trajectory */}
                    <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '16px', marginBottom: '20px' }}>
                      <h3 style={{ fontSize: '12px', fontWeight: 700, color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', marginBottom: '12px' }}>
                        System Trajectory
                      </h3>
                      <motion.div
                        animate={{ color: trajectory.label === 'Escalating system-wide' ? '#ef4444' : trajectory.label === 'Spreading to adjacent rooms' ? '#f59e0b' : trajectory.label === 'Stabilizing' ? '#22c55e' : '#64748b' }}
                        style={{ fontSize: '13px', fontWeight: 700, lineHeight: '1.6' }}
                      >
                        {trajectory.label}
                      </motion.div>
                    </div>
                  </>
                )
              })() : (
                <>
                  <h2 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '20px', color: operatorColor, letterSpacing: '1px', textTransform: 'uppercase' }}>
                    System Analytics
                  </h2>

                  {/* Global Metrics Grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px', marginBottom: '24px' }}>
                    <div>
                      <div style={{ fontSize: '11px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                        Drift
                      </div>
                      <div style={{ fontSize: '20px', fontWeight: 700, color: interpolatedData.interpolatedDrift > 0.6 ? '#ef4444' : interpolatedData.interpolatedDrift > 0.3 ? '#eab308' : '#22c55e' }}>
                        {Math.round(interpolatedData.interpolatedDrift * 100)}%
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                        Stability
                      </div>
                      <div style={{ fontSize: '20px', fontWeight: 700, color: interpolatedData.interpolatedStability < 0.4 ? '#ef4444' : interpolatedData.interpolatedStability < 0.6 ? '#eab308' : '#22c55e' }}>
                        {Math.round(interpolatedData.interpolatedStability * 100)}%
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                        Coherence
                      </div>
                      <div style={{ fontSize: '20px', fontWeight: 700, color: interpolatedData.interpolatedCoherence < 0.4 ? '#ef4444' : interpolatedData.interpolatedCoherence < 0.6 ? '#eab308' : '#06b6d4' }}>
                        {Math.round(interpolatedData.interpolatedCoherence * 100)}%
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#475569', letterSpacing: '0.8px', textTransform: 'uppercase', fontWeight: 700, marginBottom: '8px' }}>
                        Confidence
                      </div>
                      <div style={{ fontSize: '20px', fontWeight: 700, color: '#818cf8' }}>
                        {Math.round(interpolatedData.interpolatedConfidence * 100)}%
                      </div>
                    </div>
                  </div>

                  {/* Operational Assessment */}
                  <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '16px' }}>
                    <h3 style={{ fontSize: '12px', fontWeight: 700, color: '#334155', letterSpacing: '1px', textTransform: 'uppercase', marginBottom: '12px' }}>
                      Assessment
                    </h3>
                    <div style={{ fontSize: '13px', color: '#cbd5e1', lineHeight: '1.6' }}>
                      Operational State: <span style={{ color: operatorColor, fontWeight: 700 }}>{assessment.state}</span>
                    </div>
                    <div style={{ fontSize: '13px', color: '#cbd5e1', lineHeight: '1.6', marginTop: '8px' }}>
                      Propagation Strength: <span style={{ color: '#e2e8f0', fontWeight: 700 }}>{Math.round(assessment.propagationStrength * 100)}%</span>
                    </div>
                    <div style={{ fontSize: '13px', color: '#cbd5e1', lineHeight: '1.6', marginTop: '8px' }}>
                      Trajectory: <motion.span
                        animate={{ color: trajectory.label === 'Escalating system-wide' ? '#ef4444' : trajectory.label === 'Spreading to adjacent rooms' ? '#f59e0b' : trajectory.label === 'Stabilizing' ? '#22c55e' : '#64748b' }}
                        style={{ fontWeight: 700 }}
                      >
                        {trajectory.label}
                      </motion.span>
                    </div>
                  </div>
                </>
              )}

              <button
                onClick={() => setShowDetails(false)}
                style={{
                  marginTop: '20px',
                  padding: '8px 16px',
                  background: 'rgba(126,159,46,0.12)',
                  border: '1px solid rgba(126,159,46,0.35)',
                  color: '#7e9f2e',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontWeight: 600,
                  letterSpacing: '0.6px',
                  textTransform: 'uppercase',
                  transition: 'all 0.2s',
                }}
              >
                Close
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
