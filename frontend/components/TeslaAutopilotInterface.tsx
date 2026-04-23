'use client'

import React, { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { TetrahedronField } from './TetrahedronField'
import { usePhaseController } from '@/lib/phaseController'
import { useSystemInterpolation } from '@/lib/systemInterpolation'

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
  rooms?: RoomStatus[]
  intelligence?: IntelligenceData
  onStepScenario?: (delta: number) => void
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

const getStateColor = (phase: string): string => {
  const colors: Record<string, string> = {
    Stable: '#22c55e',
    'Drift forming': '#eab308',
    'Instability forming': '#f97316',
    Critical: '#ef4444',
  }
  return colors[phase] || '#9ca3af'
}

const getRoomStatusColor = (status: 'optimal' | 'warning' | 'critical'): string => {
  return status === 'critical' ? '#ef4444' : status === 'warning' ? '#eab308' : '#22c55e'
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
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: '#64748b' }}>{label}</span>
        <span style={{ fontSize: 11, color, fontWeight: 700 }}>{Math.round(value * 100)}%</span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.08)' }}>
        <motion.div
          animate={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
          transition={{ duration: 0.4 }}
          style={{ height: '100%', borderRadius: 2, background: color }}
        />
      </div>
    </div>
  )
}

export function TeslaAutopilotInterface({
  systemData,
  onTogglePlay,
  rooms = [],
  intelligence,
  onStepScenario,
}: TeslaAutopilotInterfaceProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [showLayer3, setShowLayer3] = useState(false)
  const [driftHistory, setDriftHistory] = useState<number[]>([])

  const { phase, phaseProgress } = usePhaseController(isPlaying, 1)
  const interpolatedData = useSystemInterpolation(systemData, phase)

  useEffect(() => {
    setDriftHistory(prev => [...prev.slice(-20), interpolatedData.interpolatedDrift])
  }, [interpolatedData.interpolatedDrift])

  const assessment = deriveOperationalAssessment(
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    rooms
  )

  const hasRedRoom = rooms.some(room => room.status === 'critical')
  const isRedState = phase === 'Critical' || assessment.state === 'Action required' || hasRedRoom
  const stateColor = getStateColor(phase)

  const issueSummary = useMemo(() => {
    if (!isRedState) return 'No active red issue'
    if (assessment.state === 'Action required') return 'Multi-zone structural instability detected'
    if (phase === 'Critical') return 'Critical phase threshold crossed'
    return 'Room-level critical condition detected'
  }, [assessment.state, isRedState, phase])

  const locationSummary = useMemo(() => {
    if (!isRedState) return '—'
    if (assessment.originRoom?.shortName) {
      return `${assessment.originRoom.shortName}${assessment.affectedRooms.length ? ` + ${assessment.affectedRooms.length} linked` : ''}`
    }
    const criticalRoom = rooms.find(room => room.status === 'critical')
    return criticalRoom?.shortName || 'Facility-wide'
  }, [assessment.affectedRooms.length, assessment.originRoom?.shortName, isRedState, rooms])

  const progressionSummary = useMemo(() => {
    if (!isRedState) return 'Stable trend'
    const tail = driftHistory.slice(-5)
    const rising = tail.length > 1 && tail[tail.length - 1] > tail[0]
    return rising
      ? `Escalating (${Math.round(assessment.propagationStrength * 100)}% propagation strength)`
      : `Active but non-accelerating (${Math.round(assessment.propagationStrength * 100)}% propagation strength)`
  }, [assessment.propagationStrength, driftHistory, isRedState])

  const handleTogglePlay = () => {
    const next = !isPlaying
    setIsPlaying(next)
    onTogglePlay?.(next)
  }

  return (
    <div style={{ minHeight: '100vh', background: '#050607', color: '#e2e8f0', padding: 20, fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: stateColor }}>
          System Status: {assessment.state} · Phase: {phase}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {onStepScenario && (
            <>
              <button onClick={() => onStepScenario(-1)} style={{ padding: '4px 8px' }}>Prev</button>
              <button onClick={() => onStepScenario(1)} style={{ padding: '4px 8px' }}>Next</button>
            </>
          )}
          <button onClick={handleTogglePlay} style={{ padding: '4px 10px' }}>{isPlaying ? 'Pause' : 'Play'}</button>
          <button onClick={() => setShowLayer3(prev => !prev)} style={{ padding: '4px 10px' }}>
            {showLayer3 ? 'Hide Details' : 'View Details'}
          </button>
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
          gap: 10,
          marginBottom: 16,
        }}
      >
        {rooms.map(room => (
          <div
            key={room.id}
            title={room.shortName}
            style={{
              height: 66,
              borderRadius: 6,
              background: getRoomStatusColor(room.status),
              opacity: 0.9,
            }}
          />
        ))}
      </div>

      {isRedState && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            border: '1px solid #ef444488',
            borderRadius: 8,
            background: 'rgba(127,29,29,0.25)',
            padding: 14,
            marginBottom: 16,
          }}
        >
          <div style={{ fontSize: 12, color: '#fca5a5', marginBottom: 10, fontWeight: 700 }}>Layer 2 · Auto detail (RED trigger)</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 10 }}>
            <div><div style={{ fontSize: 10, color: '#fca5a5' }}>Issue</div><div style={{ fontSize: 13 }}>{issueSummary}</div></div>
            <div><div style={{ fontSize: 10, color: '#fca5a5' }}>Location</div><div style={{ fontSize: 13 }}>{locationSummary}</div></div>
            <div><div style={{ fontSize: 10, color: '#fca5a5' }}>Progression</div><div style={{ fontSize: 13 }}>{progressionSummary}</div></div>
          </div>
        </motion.div>
      )}

      {showLayer3 && (
        <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: 16 }}>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10 }}>Layer 3 · Expanded analysis</div>

          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 12, marginBottom: 8 }}>Structural graph</div>
            <div style={{ height: 280, border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, overflow: 'hidden' }}>
              <TetrahedronField
                data={interpolatedData}
                phaseProgress={phaseProgress}
                phase={phase}
                escalationLevel={isRedState ? 3 : 1}
                hasThresholdCrossed={isRedState}
                isApproachingFailure={assessment.state === 'Propagating instability' || assessment.state === 'Action required'}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 16 }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ fontSize: 12 }}>Metrics</div>
              <MetricGauge label="Drift" value={interpolatedData.interpolatedDrift} color="#ef4444" />
              <MetricGauge label="Stability" value={interpolatedData.interpolatedStability} color="#22c55e" />
              <MetricGauge label="Coherence" value={interpolatedData.interpolatedCoherence} color="#38bdf8" />
              <MetricGauge label="Confidence" value={interpolatedData.interpolatedConfidence} color="#a78bfa" />
            </div>

            <div>
              <div style={{ fontSize: 12, marginBottom: 8 }}>Historical data</div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>
                Drift history: {driftHistory.map(v => Math.round(v * 100)).join(' · ') || 'No history yet'}
              </div>
              <div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 12 }}>Deep analysis</div>
              <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>
                {intelligence?.explanation || 'No explanatory narrative available.'} Origin focus: {assessment.originRoom?.shortName || 'unknown'}; propagation footprint:{' '}
                {assessment.affectedRooms.map(room => room.shortName).join(', ') || 'none'}.
              </div>
            </div>
          </div>
        </motion.section>
      )}
    </div>
  )
}
