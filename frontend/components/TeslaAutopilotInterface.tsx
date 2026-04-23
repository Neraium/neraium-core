'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import FacilityLinkLayer from './FacilityLinkLayer'
import { TetrahedronField } from './TetrahedronField'
import { usePhaseController } from '@/lib/phaseController'
import { useSystemInterpolation } from '@/lib/systemInterpolation'
import { computeConsequenceState } from '@/lib/decisionGravity'
import { computeDiagnosticLegibility } from '@/lib/diagnosticLegibility'
import { computeOutcomeConfidence } from '@/lib/outcomeConfidence'
import { ActionDecisionResult, evaluateActionDecision, RankedAction } from '@/lib/actionEvaluation'
import { useLiveSimulation } from '@/lib/simulation'

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

type ConsoleState = 'Stable' | 'Watch' | 'Alert'

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0
  return Math.max(0, Math.min(1, n))
}

function getStateColor(phase: string): string {
  const colors: Record<string, string> = {
    'Stable': '#22c55e',
    'Drift forming': '#eab308',
    'Instability forming': '#f97316',
    'Critical': '#ef4444',
  }
  return colors[phase] || '#94a3b8'
}

function getSystemStateLabel(phase: string): ConsoleState {
  if (phase === 'Critical') return 'Alert'
  if (phase === 'Instability forming') return 'Watch'
  return 'Stable'
}

function getDeviationBand(drift: number, stability: number, coherence: number): 'low' | 'medium' | 'high' {
  if (drift > 0.75 || stability < 0.3 || coherence < 0.3) return 'high'
  if (drift > 0.5 || stability < 0.55 || coherence < 0.55) return 'medium'
  return 'low'
}

function signalColor(label: 'drift' | 'stability' | 'coherence' | 'confidence', value01: number): string {
  if (label === 'drift') return value01 > 0.6 ? '#ef4444' : value01 > 0.3 ? '#f59e0b' : '#22c55e'
  if (label === 'stability') return value01 < 0.4 ? '#ef4444' : value01 < 0.6 ? '#f59e0b' : '#22c55e'
  if (label === 'coherence') return value01 < 0.4 ? '#ef4444' : value01 < 0.6 ? '#f59e0b' : '#06b6d4'
  return '#818cf8'
}

function SignalPill({
  label,
  value,
  color,
}: {
  label: string
  value: string
  color?: string
}) {
  return (
    <div className="pill">
      <span className="pillLabel">{label}</span>
      <span className="pillValue" style={{ color: color || '#e2e8f0' }}>
        {value}
      </span>
    </div>
  )
}

function roomStatusColor(status: RoomStatus['status']): string {
  if (status === 'critical') return '#ef4444'
  if (status === 'warning') return '#f59e0b'
  return '#22c55e'
}

function roomStatusWord(status: RoomStatus['status']): string {
  if (status === 'critical') return 'At risk'
  if (status === 'warning') return 'Transitioning'
  return 'Stable'
}

const FALLBACK_DECISION: ActionDecisionResult = {
  primaryAction: {
    label: 'Hold current state and monitor',
    confidence: 'low',
    outcome: { primary: 'Continue monitoring', continuation: 'Continue monitoring' },
    timingSensitivity: 'Monitor for changes',
    score: 0,
    stabilizationBenefit: 0,
    rationale: 'No dominant intervention candidate available from current system state.',
    urgency: 'low',
    expectedOutcome: 'Continue monitoring for clearer structural direction.',
    whatIsNext: 'Await stronger signal or broader propagation pattern.',
  },
  alternatives: [],
  noActionConsequence: 'No additional system guidance available',
  recommendationStability: 'moderate',
  decisionSummary: 'Recommendation unavailable: continuing monitoring',
  decisionRationale: 'Insufficient data for decisive action.',
}

export function TeslaAutopilotInterface({
  systemData,
  onTogglePlay,
  onSpeedChange,
  rooms,
  intelligence,
  onStepScenario,
}: TeslaAutopilotInterfaceProps) {
  const simulation = useLiveSimulation()
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [driftHistory, setDriftHistory] = useState<number[]>([])
  const [stabilityHistory, setStabilityHistory] = useState<number[]>([])
  const [actionHistory, setActionHistory] = useState<RankedAction[]>([])
  const [actionDecision, setActionDecision] = useState<ActionDecisionResult>(FALLBACK_DECISION)

  const safeTimestamp =
    (simulation.live as any)?.timestamp && typeof (simulation.live as any).timestamp?.toISOString === 'function'
      ? (simulation.live as any).timestamp.toISOString()
      : new Date().toISOString()

  const resolvedSystemData: SystemData = systemData || {
    drift: Number((simulation.live as any)?.structuralDrift ?? 0),
    stability: Number((simulation.live as any)?.stability ?? 1),
    coherence: Number((simulation.live as any)?.coherence ?? 1),
    confidence:
      (simulation.live as any)?.confidenceLabel === 'HIGH'
        ? 0.8
        : (simulation.live as any)?.confidenceLabel === 'MEDIUM'
          ? 0.5
          : 0.2,
    timestamp: safeTimestamp,
    systemState: ((simulation.live as any)?.stage as any) || 'Stable',
  }

  const resolvedRooms: RoomStatus[] =
    rooms ||
    ((simulation.live as any)?.rooms || []).map((r: any) => ({
      id: String(r?.id ?? ''),
      shortName: String(r?.shortName ?? r?.id ?? 'Zone'),
      status: (r?.status as any) || 'optimal',
      driftContribution: Number(r?.driftContribution ?? 0),
      behavioralState: String(r?.behavioralState ?? ''),
    }))

  const resolvedIntelligence: IntelligenceData = intelligence || {
    explanation: String((simulation.live as any)?.explanation ?? ''),
    operatorFocus: String((simulation.live as any)?.operatorFocus ?? ''),
    pathOutlook: String((simulation.live as any)?.pathOutlook ?? ''),
    primaryDriver: String((simulation.live as any)?.primaryDriver ?? ''),
  }

  const resolvedOnStepScenario = onStepScenario || simulation.stepScenario

  const { phase, phaseProgress } = usePhaseController(isPlaying, speed)
  const interpolatedData = useSystemInterpolation(resolvedSystemData, phase)

  useEffect(() => {
    setDriftHistory(prev => [...prev.slice(-18), interpolatedData.interpolatedDrift])
    setStabilityHistory(prev => [...prev.slice(-18), interpolatedData.interpolatedStability])
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
    phase,
    actionDecision?.primaryAction?.label || null
  )

  const outcomeState = computeOutcomeConfidence(
    diagnostics.dominantDriver || 'multi-mode',
    phase,
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    diagnostics.confidence
  )

  const driftAcceleration =
    driftHistory.length > 1
      ? driftHistory[driftHistory.length - 1] - driftHistory[Math.max(0, driftHistory.length - 3)]
      : 0

  useEffect(() => {
    try {
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
      setActionDecision(decision || FALLBACK_DECISION)

      const label = decision?.primaryAction?.label
      if (label) {
        setActionHistory(prev => [
          ...prev.slice(-4),
          {
            label,
            score: decision.primaryAction.score,
            stabilizationBenefit: decision.primaryAction.stabilizationBenefit,
            riskReduction: 0,
            failureModeFit: 1,
            timeSensitivityFit: 0.8,
            confidence: 0.8,
            aggressivenessPenalty: 0.9,
            outcome: decision.primaryAction.outcome,
            timingSensitivity: decision.primaryAction.timingSensitivity,
          },
        ])
      }
    } catch {
      setActionDecision(FALLBACK_DECISION)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    phase,
    diagnostics.dominantDriver,
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence,
    diagnostics.confidence,
    consequenceState.timeToImpact,
    driftAcceleration,
    outcomeState.noActionConsequence,
  ])

  const stateColor = getStateColor(phase)
  const systemState = getSystemStateLabel(phase)
  const confidenceLabel = diagnostics.confidence === 'high' ? 'High' : diagnostics.confidence === 'moderate' ? 'Moderate' : 'Low'
  const deviationBand = getDeviationBand(
    interpolatedData.interpolatedDrift,
    interpolatedData.interpolatedStability,
    interpolatedData.interpolatedCoherence
  )
  const projectedWindow = consequenceState.timeToImpactLabel || 'next 10-20 cycles'

  const urgencyLevel = phase === 'Critical' ? 5 : phase === 'Instability forming' ? 4 : phase === 'Drift forming' ? 3 : 1
  const urgencyColor = urgencyLevel >= 5 ? '#ef4444' : urgencyLevel >= 4 ? '#f97316' : urgencyLevel >= 3 ? '#eab308' : '#22c55e'
  const urgencyWord = urgencyLevel >= 5 ? 'Critical' : urgencyLevel >= 4 ? 'High' : urgencyLevel >= 3 ? 'Moderate' : 'Low'

  const primaryDriver = (resolvedIntelligence.primaryDriver || diagnostics.dominantDriver || 'multi-mode').toString().replace(/-/g, ' ')

  const topRoom = useMemo(() => {
    const sorted = [...(resolvedRooms || [])]
      .filter(r => r && (r.shortName || r.id))
      .sort((a, b) => clamp01(b.driftContribution) - clamp01(a.driftContribution))
    return sorted[0]?.shortName || sorted[0]?.id || 'facility'
  }, [resolvedRooms])

  const nowLine = useMemo(() => {
    if (phase === 'Critical') return `Critical instability across ${topRoom} driven by ${primaryDriver}`
    if (phase === 'Instability forming') return `Emerging instability in ${topRoom} driven by ${primaryDriver}`
    if (phase === 'Drift forming') return `Coupling drift forming near ${topRoom} driven by ${primaryDriver}`
    return `System nominal. Monitoring ${primaryDriver} for early variation.`
  }, [phase, topRoom, primaryDriver])

  const recommendation = useMemo(() => {
    const label = actionDecision?.primaryAction?.label || 'Hold current state'
    const immediateNextAction =
      actionDecision?.primaryAction?.outcome?.continuation ||
      actionDecision?.primaryAction?.outcome?.primary ||
      'Continue monitoring'
    const expectedOutcome = actionDecision?.primaryAction?.outcome?.primary || 'Continue monitoring'
    const ifIgnored = outcomeState.noActionConsequence || 'No additional system guidance available'
    const reason = actionDecision?.decisionRationale || resolvedIntelligence.operatorFocus || actionDecision?.decisionSummary || 'Recommendation unavailable'
    return {
      label,
      urgency: urgencyWord,
      reason,
      expectedOutcome,
      immediateNextAction,
      ifIgnored,
      alternatives: actionDecision?.alternatives || [],
      decisionRationale: actionDecision?.decisionRationale || '',
    }
  }, [actionDecision, outcomeState.noActionConsequence, resolvedIntelligence.operatorFocus, urgencyWord])

  const previousRoomDriftRef = useRef<Record<string, number>>({})
  const roomDeltas = useMemo(() => {
    const prev = previousRoomDriftRef.current
    const next: Record<string, number> = {}
    const arrows: Record<string, '^' | 'v' | '>'> = {}
    ;(resolvedRooms || []).forEach((r: RoomStatus) => {
      if (!r?.id) return
      const current = Number(r?.driftContribution ?? 0)
      const prior = Number(prev[r.id] ?? current)
      const delta = current - prior
      arrows[r.id] = delta > 0.01 ? '^' : delta < -0.01 ? 'v' : '>'
      next[r.id] = current
    })
    previousRoomDriftRef.current = next
    return arrows
  }, [resolvedRooms])

  const facilityModel = useMemo(() => {
    const all = [...(resolvedRooms || [])].filter(r => r && r.id)
    const affected = all
      .filter(r => r.status !== 'optimal' || clamp01(r.driftContribution) > 0.25)
      .sort((a, b) => clamp01(b.driftContribution) - clamp01(a.driftContribution))
    const origin = affected[0] || all.sort((a, b) => clamp01(b.driftContribution) - clamp01(a.driftContribution))[0]
    const pathIds = affected.slice(0, 4).map(r => r.id)
    return {
      originId: origin?.id || '',
      pathIds,
      pathSet: new Set(pathIds),
    }
  }, [resolvedRooms])

  const zoneDetailFiller = useMemo(() => {
    const list = (resolvedRooms || []).map(r => (r?.behavioralState || '').trim()).filter(Boolean)
    if (list.length < 4) return null
    const counts = new Map<string, number>()
    for (const s of list) counts.set(s, (counts.get(s) || 0) + 1)
    let top: { s: string; n: number } | null = null
    for (const [s, n] of counts.entries()) {
      if (!top || n > top.n) top = { s, n }
    }
    if (!top) return null
    const ratio = top.n / Math.max(1, (resolvedRooms || []).length)
    return ratio >= 0.6 ? top.s : null
  }, [resolvedRooms])

  const handleTogglePlay = () => {
    setIsPlaying(p => {
      const next = !p
      if (onTogglePlay) onTogglePlay(next)
      else simulation.setIsPlaying(next)
      return next
    })
  }

  const handleSpeedChange = (nextSpeed: number) => {
    const s = Number(nextSpeed) || 1
    setSpeed(s)
    if (onSpeedChange) onSpeedChange(s)
  }

  return (
    <div className="root">
      <div className="scanline" />

      {/* Sticky hero decision strip */}
      <motion.div
        initial={{ opacity: 0, y: -14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55 }}
        className="topBar"
        style={{ borderBottomColor: `${stateColor}2a` }}
      >
        <div className="topLeft">
          <motion.div
            animate={{
              background: stateColor,
              boxShadow: `0 0 ${phase === 'Critical' ? '10px' : '6px'} ${stateColor}`,
            }}
            transition={{ duration: 0.35 }}
            className="stateDot"
          />

          <div className="topText">
            <div className="topMeta">
              <span className="metaLabel">System</span>
              <span className="metaValue" style={{ color: stateColor }}>
                {systemState}
              </span>
              <span className="metaSep" />
              <span className="metaSecondary">
                Urgency: <span className="metaEmph">{urgencyWord}</span>
              </span>
              <span className="metaSecondary">
                Confidence: <span className="metaEmph">{confidenceLabel}</span>
              </span>
              <span className="metaSecondary">
                Window: <span className="metaEmph">{consequenceState.timeToImpactLabel || '-'}</span>
              </span>
            </div>

            <div className="topDoNow">
              DO NOW <span style={{ color: stateColor }}>{recommendation.label}</span>
            </div>
          </div>
        </div>

        <div className="topRight">
          <div className="ctrl">
            <span className="ctrlLabel">Speed</span>
            <input
              type="range"
              min={0.5}
              max={3}
              step={0.5}
              value={speed}
              onChange={e => handleSpeedChange(Number(e.target.value))}
              className="range"
            />
            <span className="ctrlValue">{speed}x</span>
          </div>

          {resolvedOnStepScenario && (
            <div className="scenarioNav">
              <button className="iconBtn" onClick={() => resolvedOnStepScenario(-1)} aria-label="Previous scenario">
                {'<'}
              </button>
              <button className="iconBtn" onClick={() => resolvedOnStepScenario(1)} aria-label="Next scenario">
                {'>'}
              </button>
            </div>
          )}

          <button className="playBtn" onClick={handleTogglePlay} style={{ borderColor: `${stateColor}55`, color: isPlaying ? stateColor : '#7e9f2e' }}>
            {isPlaying ? 'Pause' : 'Play'}
          </button>
        </div>
      </motion.div>

      <div className="progressTrack">
        <motion.div
          animate={{ width: `${phaseProgress * 100}%`, backgroundColor: stateColor }}
          transition={{ duration: 0.12 }}
          className="progressFill"
          style={{ boxShadow: `0 0 10px ${stateColor}66` }}
        />
      </div>

      {/* Main console */}
      <div className="scroll">
        <div className="wrap">
          <FacilityLinkLayer rooms={resolvedRooms} primaryDriver={primaryDriver} systemState={systemState} />

          <div className="mainGrid">
            <div className="centerpiece">
              <div className="centerpieceHeader">
                <div style={{ minWidth: 0 }}>
                  <div className="sectionLabel">Current system state</div>
                  <div className="sectionTitle">{`Trajectory shows ${phase === 'Stable' ? 'nominal structure' : phase.toLowerCase()}`}</div>
                </div>
                <div className="pillRow">
                  <SignalPill label="Urgency" value={urgencyWord} color={urgencyColor} />
                  <SignalPill label="Confidence" value={confidenceLabel} />
                </div>
              </div>

              <div className="vizFrame">
                <TetrahedronField
                  data={interpolatedData}
                  phaseProgress={phaseProgress}
                  phase={phase}
                  escalationLevel={consequenceState.escalationLevel}
                  hasThresholdCrossed={consequenceState.hasThresholdCrossed}
                  isApproachingFailure={consequenceState.isApproachingFailure}
                  decisionRationale={recommendation.decisionRationale}
                />
              </div>

              <div className="pillRow" style={{ marginTop: 10 }}>
                <SignalPill label="Drift" value={`${Math.round(clamp01(interpolatedData.interpolatedDrift) * 100)}`} color={signalColor('drift', interpolatedData.interpolatedDrift)} />
                <SignalPill label="Stability" value={`${Math.round(clamp01(interpolatedData.interpolatedStability) * 100)}`} color={signalColor('stability', interpolatedData.interpolatedStability)} />
                <SignalPill label="Coherence" value={`${Math.round(clamp01(interpolatedData.interpolatedCoherence) * 100)}`} color={signalColor('coherence', interpolatedData.interpolatedCoherence)} />
              </div>
            </div>

            <div className="side">
              {/* PRIMARY DIRECTIVE — dominant, unmissable */}
              <div className="card actionCard" style={{ borderTopColor: `${stateColor}cc` }}>
                <div className="cardHeader">
                  <div className="sectionLabel">Operator directive</div>
                  <div className="badge" style={{ color: urgencyColor, background: `${urgencyColor}14`, borderColor: `${urgencyColor}30` }}>
                    {urgencyWord}
                  </div>
                </div>

                <div className="actionTitle" style={{ color: stateColor }}>{recommendation.label}</div>

                <div className="actionWhy" style={{ marginTop: 10, lineHeight: 1.6 }}>
                  {recommendation.reason}
                </div>

                {/* Structured sections with clear visual separation */}
                <div className="sectionDivider" />

                <div className="guidanceGrid">
                  <div className="guidanceSection">
                    <div className="guidanceLabel">What is happening</div>
                    <div className="guidanceValue">{nowLine}</div>
                  </div>

                  <div className="guidanceSection">
                    <div className="guidanceLabel">What is driving it</div>
                    <div className="guidanceValue">{primaryDriver}</div>
                  </div>

                  <div className="guidanceSection">
                    <div className="guidanceLabel">What to do</div>
                    <div className="guidanceValue" style={{ color: stateColor, fontWeight: 900 }}>
                      {recommendation.label}
                    </div>
                    <div className="guidanceNext">Next: {recommendation.immediateNextAction}</div>
                  </div>

                  <div className="guidanceSection">
                    <div className="guidanceLabel">Impact window</div>
                    <div className="guidanceValue">
                      {consequenceState.timeToImpactLabel || 'Open'} — deviation {deviationBand} over {projectedWindow}
                    </div>
                  </div>

                  <div className="guidanceSection">
                    <div className="guidanceLabel">Expected outcome</div>
                    <div className="guidanceValue">{recommendation.expectedOutcome}</div>
                  </div>

                  <div className="guidanceSection">
                    <div className="guidanceLabel">If unchanged</div>
                    <div className="guidanceMuted">{recommendation.ifIgnored}</div>
                  </div>
                </div>

                {/* Alternatives — clearly labeled as lower confidence */}
                {recommendation.alternatives.length > 0 && (
                  <>
                    <div className="sectionDivider" />
                    <div className="sectionLabel" style={{ marginBottom: 8 }}>Alternative actions</div>
                    <div className="alternativesList">
                      {recommendation.alternatives.map((alt, i) => (
                        <div key={i} className="alternativeRow">
                          <div className="alternativeLabel">{alt.label}</div>
                          <div className="alternativeMeta">
                            <span className="alternativeDelta">−{Math.round(alt.confidenceDelta * 100)}% confidence</span>
                            <span className="alternativeOutcome">{alt.outcome.primary}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          {resolvedRooms && resolvedRooms.length > 0 && (
            <div className="zoneSection">
              <div className="zoneSectionHeader">
                <div style={{ minWidth: 0 }}>
                  <div className="sectionLabel">Zone detail</div>
                  <div className="zoneSectionTitle">State, risk, and propagation relevance</div>
                </div>
                <div className="pillRow">
                  <SignalPill
                    label="Origin"
                    value={
                      resolvedRooms.find(r => r.id === facilityModel.originId)?.shortName ||
                      resolvedRooms.find(r => r.id === facilityModel.originId)?.id ||
                      '-'
                    }
                    color={stateColor}
                  />
                  <SignalPill label="Affected" value={`${facilityModel.pathIds.length}`} />
                </div>
              </div>

              <div className="zonesWideGrid">
                {resolvedRooms
                  .slice()
                  .sort((a, b) => clamp01(b.driftContribution) - clamp01(a.driftContribution))
                  .map(room => {
                    const c = roomStatusColor(room.status)
                    const arrow = roomDeltas[room.id] || '>'
                    const pct = Math.round(clamp01(room.driftContribution) * 100)
                    const isOrigin = facilityModel.originId && room.id === facilityModel.originId
                    const isCoupled = facilityModel.pathSet.has(room.id)
                    const detail =
                      room.status !== 'optimal' || isCoupled
                        ? (room.behavioralState && room.behavioralState !== zoneDetailFiller
                          ? room.behavioralState
                          : 'Non-nominal dynamics')
                        : ''
                    return (
                      <div
                        key={room.id}
                        className="zoneRow"
                        style={{
                          borderColor: room.status !== 'optimal' || isCoupled ? `${c}26` : 'rgba(255,255,255,0.06)',
                          background: isOrigin ? `${c}10` : 'rgba(255,255,255,0.03)',
                        }}
                      >
                        <div className="zoneRowTop">
                          <div style={{ minWidth: 0 }}>
                            <div className="zoneRowName">
                              {room.shortName || room.id}
                              {isOrigin ? <span className="tag" style={{ borderColor: `${c}40`, color: c }}>Origin</span> : null}
                              {!isOrigin && isCoupled ? <span className="tag">Path</span> : null}
                            </div>
                            <div className="zoneRowState">{roomStatusWord(room.status)}</div>
                          </div>
                          <div className="zoneRowRight">
                            <div className="zoneArrow" style={{ color: c }}>{arrow}</div>
                            <div className="zonePct" style={{ color: c }}>{pct}%</div>
                          </div>
                        </div>

                        {detail ? <div className="zoneRowDetail">{detail}</div> : null}
                      </div>
                    )
                  })}
              </div>
            </div>
          )}
        </div>
      </div>

      {(phase === 'Instability forming' || phase === 'Critical') && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="bottomBanner"
          style={{ borderTopColor: `${stateColor}2a`, background: `${stateColor}09` }}
        >
          <div className="bannerItem">
            <span className="miniLabel">Consequence</span>
            <span className="bannerValue" style={{ color: stateColor }}>
              {consequenceState.operatorFocus || 'Structural deviation accelerating'}
            </span>
          </div>
          <div className="bannerItem">
            <span className="miniLabel">Operator</span>
            <span className="bannerValue">{recommendation.label}</span>
          </div>
        </motion.div>
      )}

      <style jsx>{`
        .root {
          width: 100vw;
          height: 100vh;
          background: #050607;
          color: #e2e8f0;
          overflow: hidden;
          position: relative;
          font-family: system-ui, -apple-system, Segoe UI, sans-serif;
          --r-lg: 14px;
          --r-md: 12px;
          --r-pill: 999px;
          --border: rgba(255, 255, 255, 0.06);
          --surface: rgba(5, 6, 7, 0.55);
          --surface2: rgba(255, 255, 255, 0.03);
          --shadow: 0 28px 80px rgba(0, 0, 0, 0.55);
          --inset: 0 0 0 1px rgba(255, 255, 255, 0.02) inset;
        }
        .scanline {
          position: absolute;
          inset: 0;
          background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 0, 0, 0.15) 2px,
            rgba(0, 0, 0, 0.15) 4px
          );
          pointer-events: none;
          z-index: 99;
          opacity: 0.18;
        }
        .scanline::after {
          content: '';
          position: absolute;
          inset: 0;
          background: radial-gradient(1200px 520px at 50% 18%, rgba(148, 163, 184, 0.06), transparent 70%);
          opacity: 0.7;
          pointer-events: none;
        }
        .topBar {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          z-index: 50;
          background: linear-gradient(180deg, rgba(5, 6, 7, 0.98) 0%, rgba(5, 6, 7, 0.68) 70%, rgba(5, 6, 7, 0.52) 100%);
          backdrop-filter: blur(10px);
          border-bottom: 1px solid var(--border);
          padding: 0 18px;
          height: 94px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 14px;
        }
        .topLeft {
          display: flex;
          gap: 12px;
          align-items: center;
          min-width: 0;
          flex: 1;
        }
        .stateDot {
          width: 7px;
          height: 7px;
          border-radius: var(--r-pill);
          flex-shrink: 0;
        }
        .topText {
          display: grid;
          gap: 4px;
          min-width: 0;
          flex: 1;
        }
        .topMeta {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          align-items: baseline;
          min-width: 0;
        }
        .metaLabel {
          font-weight: 950;
          color: #e2e8f0;
          font-size: 10px;
          letter-spacing: 0.8px;
          text-transform: uppercase;
        }
        .metaValue {
          font-weight: 950;
          font-size: 11px;
          letter-spacing: 1px;
          text-transform: uppercase;
        }
        .metaSep {
          width: 1px;
          height: 12px;
          background: rgba(255, 255, 255, 0.08);
        }
        .metaSecondary {
          color: #94a3b8;
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.4px;
          text-transform: uppercase;
        }
        .metaEmph {
          color: #e2e8f0;
          font-weight: 900;
        }
        .topNow {
          font-size: 16px;
          font-weight: 950;
          line-height: 1.15;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .topDoNow {
          font-size: 16px;
          font-weight: 950;
          color: #e2e8f0;
          letter-spacing: 0.2px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          line-height: 1.2;
        }
        .topRight {
          display: flex;
          gap: 12px;
          align-items: center;
          flex-shrink: 0;
        }
        .ctrl {
          display: flex;
          align-items: center;
          gap: 7px;
        }
        .ctrlLabel {
          font-size: 10px;
          color: #64748b;
          letter-spacing: 0.8px;
          text-transform: uppercase;
          font-weight: 900;
        }
        .ctrlValue {
          font-size: 11px;
          color: #94a3b8;
          min-width: 26px;
          font-variant-numeric: tabular-nums;
          font-weight: 800;
        }
        .range {
          width: 64px;
          height: 3px;
          accent-color: #7e9f2e;
          cursor: pointer;
        }
        .scenarioNav {
          display: flex;
          gap: 3px;
        }
        .iconBtn {
          background: rgba(148, 163, 184, 0.08);
          border: 1px solid rgba(148, 163, 184, 0.22);
          color: #94a3b8;
          width: 24px;
          height: 24px;
          border-radius: 10px;
          cursor: pointer;
          font-size: 14px;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.15s ease;
        }
        .iconBtn:hover {
          color: #e2e8f0;
          border-color: rgba(148, 163, 184, 0.55);
          background: rgba(148, 163, 184, 0.12);
        }
        .playBtn {
          background: rgba(126, 159, 46, 0.12);
          border: 1px solid rgba(126, 159, 46, 0.35);
          padding: 6px 14px;
          border-radius: var(--r-pill);
          cursor: pointer;
          font-size: 11px;
          font-weight: 900;
          letter-spacing: 1px;
          text-transform: uppercase;
          transition: all 0.2s ease;
        }
        .playBtn:hover {
          background: rgba(126, 159, 46, 0.18);
        }
        .progressTrack {
          position: absolute;
          top: 94px;
          left: 0;
          right: 0;
          height: 2px;
          background: rgba(255, 255, 255, 0.04);
          z-index: 49;
        }
        .progressFill {
          height: 100%;
        }
        .scroll {
          position: absolute;
          top: 96px;
          left: 0;
          right: 0;
          bottom: 0;
          overflow-y: auto;
        }
        .wrap {
          max-width: 1260px;
          margin: 0 auto;
          padding: 20px 18px 36px;
          display: grid;
          gap: 18px;
        }
        .mainGrid {
          display: grid;
          grid-template-columns: minmax(0, 1fr) 440px;
          gap: 18px;
          align-items: start;
        }
        .centerpiece {
          background: linear-gradient(180deg, rgba(255, 255, 255, 0.035) 0%, rgba(5, 6, 7, 0.46) 40%, rgba(5, 6, 7, 0.56) 100%);
          border: 1px solid var(--border);
          border-radius: var(--r-lg);
          padding: 14px 14px 12px;
          overflow: hidden;
          box-shadow: var(--inset), var(--shadow);
        }
        .centerpieceHeader {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 12px;
          padding: 2px 4px 10px;
        }
        .sectionLabel {
          font-size: 10px;
          color: #94a3b8;
          letter-spacing: 1px;
          text-transform: uppercase;
          font-weight: 950;
        }
        .sectionTitle {
          margin-top: 2px;
          font-size: 12px;
          font-weight: 900;
          color: #e2e8f0;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .pillRow {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          justify-content: flex-end;
        }
        .vizFrame {
          height: 660px;
          border-radius: var(--r-md);
          overflow: hidden;
        }
        .side {
          display: grid;
          gap: 14px;
        }
        .card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--r-lg);
          padding: 14px 14px 12px;
        }
        .actionCard {
          background: linear-gradient(180deg, rgba(15, 23, 42, 0.55) 0%, rgba(5, 6, 7, 0.55) 100%);
          border-top: 2px solid rgba(255, 255, 255, 0.12);
          box-shadow: var(--inset), 0 22px 70px rgba(0, 0, 0, 0.56);
          position: sticky;
          top: 14px;
        }
        .cardHeader {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 10px;
        }
        .badge {
          font-size: 10px;
          letter-spacing: 1px;
          text-transform: uppercase;
          font-weight: 950;
          padding: 6px 8px;
          border-radius: var(--r-pill);
          border: 1px solid rgba(255, 255, 255, 0.06);
        }
        .actionTitle {
          margin-top: 10px;
          font-size: 28px;
          font-weight: 950;
          letter-spacing: 0.1px;
          line-height: 1.12;
        }
        .actionWhy {
          margin-top: 8px;
          font-size: 13px;
          color: #cbd5e1;
          line-height: 1.6;
        }
        .miniLabel {
          font-size: 10px;
          color: #94a3b8;
          letter-spacing: 1px;
          text-transform: uppercase;
          font-weight: 950;
        }
        .sectionDivider {
          height: 1px;
          background: rgba(255, 255, 255, 0.06);
          margin: 14px 0;
        }
        .guidanceGrid {
          display: grid;
          gap: 14px;
        }
        .guidanceSection {
          display: grid;
          gap: 4px;
        }
        .guidanceLabel {
          font-size: 10px;
          color: #94a3b8;
          letter-spacing: 1px;
          text-transform: uppercase;
          font-weight: 950;
        }
        .guidanceValue {
          font-size: 13px;
          color: #e2e8f0;
          font-weight: 850;
          line-height: 1.5;
        }
        .guidanceNext {
          font-size: 12px;
          color: #94a3b8;
          font-weight: 700;
          margin-top: 2px;
        }
        .guidanceMuted {
          font-size: 13px;
          color: #cbd5e1;
          line-height: 1.5;
          opacity: 0.85;
        }
        .kvGrid {
          margin-top: 10px;
          display: grid;
          gap: 10px;
        }
        .kvValue {
          font-size: 12px;
          color: #e2e8f0;
          font-weight: 850;
          line-height: 1.45;
        }
        .kvMuted {
          font-size: 12px;
          color: #cbd5e1;
          line-height: 1.5;
          opacity: 0.92;
        }
        .pill {
          display: flex;
          align-items: baseline;
          gap: 8px;
          padding: 8px 10px;
          border-radius: var(--r-pill);
          background: rgba(255, 255, 255, 0.035);
          border: 1px solid rgba(255, 255, 255, 0.055);
        }
        .pillLabel {
          font-size: 10px;
          color: #94a3b8;
          letter-spacing: 1px;
          text-transform: uppercase;
          font-weight: 950;
        }
        .pillValue {
          font-size: 12px;
          font-weight: 950;
          font-variant-numeric: tabular-nums;
        }
        .details {
          margin-top: 12px;
        }
        .details summary {
          cursor: pointer;
          user-select: none;
          font-size: 10px;
          color: #94a3b8;
          letter-spacing: 1px;
          text-transform: uppercase;
          font-weight: 950;
        }
        .detailGrid {
          margin-top: 10px;
          display: grid;
          gap: 10px;
        }
        .alternativesList {
          display: grid;
          gap: 8px;
        }
        .alternativeRow {
          background: rgba(255, 255, 255, 0.025);
          border: 1px solid rgba(255, 255, 255, 0.04);
          border-radius: 10px;
          padding: 10px 12px;
          display: grid;
          gap: 4px;
        }
        .alternativeLabel {
          font-size: 12px;
          color: #e2e8f0;
          font-weight: 850;
        }
        .alternativeMeta {
          display: flex;
          align-items: baseline;
          gap: 10px;
          flex-wrap: wrap;
        }
        .alternativeDelta {
          font-size: 10px;
          color: #f59e0b;
          font-weight: 800;
          letter-spacing: 0.4px;
        }
        .alternativeOutcome {
          font-size: 11px;
          color: #94a3b8;
          font-weight: 700;
        }
        .intelGrid {
          margin-top: 10px;
          display: grid;
          gap: 12px;
        }
        .zoneSection {
          margin-top: 4px;
          background: rgba(5, 6, 7, 0.42);
          border: 1px solid var(--border);
          border-radius: var(--r-lg);
          padding: 14px 14px 12px;
        }
        .zoneSectionHeader {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 12px;
          padding: 2px 4px 10px;
        }
        .zoneSectionTitle {
          margin-top: 2px;
          font-size: 12px;
          font-weight: 900;
          color: #e2e8f0;
          opacity: 0.92;
        }
        .zonesWideGrid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
          gap: 10px;
        }
        .zoneRow {
          padding: 12px 12px;
          border-radius: var(--r-md);
          border: 1px solid rgba(255, 255, 255, 0.055);
          display: grid;
          gap: 8px;
        }
        .zoneRowTop {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
        }
        .zoneRowName {
          font-size: 12px;
          color: #e2e8f0;
          font-weight: 950;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          display: flex;
          gap: 8px;
          align-items: center;
        }
        .zoneRowState {
          margin-top: 3px;
          font-size: 10px;
          color: #94a3b8;
          letter-spacing: 0.4px;
          text-transform: uppercase;
          font-weight: 950;
        }
        .zoneRowRight {
          display: grid;
          justify-items: end;
          gap: 2px;
          flex-shrink: 0;
        }
        .tag {
          font-size: 10px;
          color: #cbd5e1;
          letter-spacing: 0.6px;
          text-transform: uppercase;
          font-weight: 950;
          padding: 4px 8px;
          border-radius: var(--r-pill);
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .zoneRowDetail {
          font-size: 11px;
          color: #cbd5e1;
          line-height: 1.4;
          opacity: 0.9;
        }
        .zoneTile {
          padding: 10px 10px;
          border-radius: 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.06);
          display: grid;
          gap: 6px;
        }
        .zoneTop {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
        }
        .zoneName {
          font-size: 12px;
          color: #e2e8f0;
          font-weight: 950;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .zoneState {
          margin-top: 2px;
          font-size: 10px;
          color: #94a3b8;
          letter-spacing: 0.4px;
          text-transform: uppercase;
          font-weight: 950;
        }
        .zoneRight {
          display: grid;
          justify-items: end;
          gap: 2px;
          flex-shrink: 0;
        }
        .zoneArrow {
          font-size: 16px;
          font-weight: 950;
          line-height: 1;
        }
        .zonePct {
          font-size: 11px;
          font-weight: 950;
          font-variant-numeric: tabular-nums;
        }
        .zoneDetail {
          font-size: 11px;
          color: #cbd5e1;
          line-height: 1.35;
          opacity: 0.92;
        }
        .bottomBanner {
          position: absolute;
          left: 0;
          right: 0;
          bottom: 0;
          padding: 10px 18px;
          border-top: 1px solid rgba(255, 255, 255, 0.06);
          display: flex;
          justify-content: center;
          gap: 26px;
          align-items: center;
          font-size: 12px;
          z-index: 20;
        }
        .bannerItem {
          display: flex;
          align-items: baseline;
          gap: 8px;
        }
        .bannerValue {
          font-weight: 900;
          color: #e2e8f0;
        }

        @media (max-width: 1100px) {
          .mainGrid {
            grid-template-columns: 1fr;
          }
          .vizFrame {
            height: 560px;
          }
          .actionCard {
            position: relative;
            top: 0;
          }
        }
      `}</style>
    </div>
  )
}

export default TeslaAutopilotInterface
