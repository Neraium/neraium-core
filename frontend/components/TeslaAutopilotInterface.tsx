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

function getSystemStateLabel(phase: string): ConsoleState {
  if (phase === 'Critical') return 'Alert'
  if (phase === 'Instability forming') return 'Watch'
  return 'Stable'
}

function systemTone(phase: string): { tone: 'GREEN' | 'YELLOW' | 'RED'; color: string; title: string; subtitle: string } {
  if (phase === 'Critical' || phase === 'Instability forming') {
    return { tone: 'RED', color: '#ef4444', title: 'System needs attention', subtitle: 'Read the system message for detail.' }
  }
  if (phase === 'Drift forming') {
    return { tone: 'YELLOW', color: '#eab308', title: 'System noticing early change', subtitle: 'No immediate action required.' }
  }
  return { tone: 'GREEN', color: '#22c55e', title: 'System stable', subtitle: 'All subsystems nominal.' }
}

function roomTone(status: RoomStatus['status']): { color: string; word: string } {
  if (status === 'critical') return { color: '#ef4444', word: 'Attention' }
  if (status === 'warning') return { color: '#eab308', word: 'Noticing' }
  return { color: '#22c55e', word: 'Stable' }
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
  const [selectedRoomId, setSelectedRoomId] = useState<string | null>(null)

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
  const interpolated = useSystemInterpolation(resolvedSystemData, phase)

  useEffect(() => {
    setDriftHistory(prev => [...prev.slice(-18), interpolated.interpolatedDrift])
    setStabilityHistory(prev => [...prev.slice(-18), interpolated.interpolatedStability])
  }, [interpolated.interpolatedDrift, interpolated.interpolatedStability])

  const diagnostics = computeDiagnosticLegibility(
    phase,
    interpolated.interpolatedDrift,
    interpolated.interpolatedStability,
    interpolated.interpolatedCoherence,
    driftHistory,
    stabilityHistory
  )

  const consequenceState = computeConsequenceState(
    phase,
    phaseProgress,
    interpolated.interpolatedDrift,
    interpolated.interpolatedStability,
    interpolated.interpolatedCoherence,
    phase,
    actionDecision?.primaryAction?.label || null
  )

  const outcomeState = computeOutcomeConfidence(
    diagnostics.dominantDriver || 'multi-mode',
    phase,
    interpolated.interpolatedDrift,
    interpolated.interpolatedStability,
    interpolated.interpolatedCoherence,
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
        interpolated.interpolatedDrift,
        interpolated.interpolatedStability,
        interpolated.interpolatedCoherence,
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
    interpolated.interpolatedDrift,
    interpolated.interpolatedStability,
    interpolated.interpolatedCoherence,
    diagnostics.confidence,
    consequenceState.timeToImpact,
    driftAcceleration,
    outcomeState.noActionConsequence,
  ])

  const tone = systemTone(phase)
  const systemState = getSystemStateLabel(phase)
  const confidenceLabel = diagnostics.confidence === 'high' ? 'High' : diagnostics.confidence === 'moderate' ? 'Moderate' : 'Low'
  const urgencyWord = phase === 'Critical' ? 'Critical' : phase === 'Instability forming' ? 'High' : phase === 'Drift forming' ? 'Moderate' : 'Low'
  const projectedWindow = consequenceState.timeToImpactLabel || 'next 10–20 cycles'
  const primaryDriver = (resolvedIntelligence.primaryDriver || diagnostics.dominantDriver || 'multi-mode').toString().replace(/-/g, ' ')

  const topRoom = useMemo(() => {
    const sorted = [...(resolvedRooms || [])]
      .filter(r => r && (r.shortName || r.id))
      .sort((a, b) => clamp01(b.driftContribution) - clamp01(a.driftContribution))
    return sorted[0]?.shortName || sorted[0]?.id || 'facility'
  }, [resolvedRooms])

  const nowLine = useMemo(() => {
    if (tone.tone === 'RED') return `Instability forming near ${topRoom} driven by ${primaryDriver}.`
    if (tone.tone === 'YELLOW') return `Early change detected near ${topRoom}. Monitoring ${primaryDriver}.`
    return `System stable. Monitoring ${primaryDriver}.`
  }, [tone.tone, topRoom, primaryDriver])

  const systemMessage = useMemo(() => {
    if (tone.tone === 'GREEN') return 'No anomalous behavior detected.'
    if (tone.tone === 'YELLOW') return `Early signal observed. No immediate action required.`
    return resolvedIntelligence.operatorFocus || resolvedIntelligence.explanation || nowLine
  }, [tone.tone, resolvedIntelligence.operatorFocus, resolvedIntelligence.explanation, nowLine])

  const recommendationNote = useMemo(() => {
    const reason = resolvedIntelligence.operatorFocus || resolvedIntelligence.explanation || 'No additional system note available.'
    const ifUnchanged = outcomeState.noActionConsequence || 'No additional system guidance available.'
    return { reason, ifUnchanged }
  }, [resolvedIntelligence.operatorFocus, resolvedIntelligence.explanation, outcomeState.noActionConsequence])

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

  const selectedRoom = useMemo(() => (resolvedRooms || []).find(r => r.id === selectedRoomId) || null, [resolvedRooms, selectedRoomId])

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

  const stateColor = tone.color

  return (
    <div className="root">
      <div className="frame">
        <header className="systemPanel" style={{ borderColor: `${tone.color}55`, background: `${tone.color}08` }}>
          <div className="systemTop">
            <div className="systemHeadline">
              <span className="systemDot" style={{ background: tone.color, boxShadow: `0 0 16px ${tone.color}55` }} />
              <div className="systemText">
                <div className="systemTitle">{tone.title}</div>
                <div className="systemSub">{tone.subtitle}</div>
              </div>
            </div>

            <div className="systemControls" aria-label="Demo controls">
              <button className="ctrlBtn" onClick={handleTogglePlay} style={{ borderColor: `${tone.color}55` }}>
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <div className="ctrlGroup">
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
              {resolvedOnStepScenario ? (
                <div className="ctrlGroup">
                  <button className="stepBtn" onClick={() => resolvedOnStepScenario(-1)} aria-label="Previous scenario">
                    ‹
                  </button>
                  <button className="stepBtn" onClick={() => resolvedOnStepScenario(1)} aria-label="Next scenario">
                    ›
                  </button>
                </div>
              ) : null}
            </div>
          </div>

          <div className="systemMessage" style={{ borderColor: `${tone.color}2a` }}>
            {systemMessage}
          </div>

          <div className="systemMeta">
            <div className="metaItem">
              <span className="metaKey">State</span>
              <span className="metaVal" style={{ color: tone.color }}>
                {systemState}
              </span>
            </div>
            <div className="metaItem">
              <span className="metaKey">Confidence</span>
              <span className="metaVal">{confidenceLabel}</span>
            </div>
            <div className="metaItem">
              <span className="metaKey">Urgency</span>
              <span className="metaVal">{urgencyWord}</span>
            </div>
            <div className="metaItem">
              <span className="metaKey">Window</span>
              <span className="metaVal">{projectedWindow}</span>
            </div>
          </div>
        </header>

        <main className="roomsGrid" role="list" aria-label="Subsystem status">
          {(resolvedRooms || []).map(room => {
            const rt = roomTone(room.status)
            const isOrigin = room.id === facilityModel.originId
            const isCoupled = facilityModel.pathSet.has(room.id) && room.id !== facilityModel.originId
            const arrow = roomDeltas[room.id] === '^' ? '↑' : roomDeltas[room.id] === 'v' ? '↓' : '→'
            const emphasis = isOrigin ? 'Origin' : isCoupled ? 'Coupled' : null
            return (
              <button
                key={room.id}
                type="button"
                className="roomCard"
                onClick={() => setSelectedRoomId(room.id)}
                style={{ borderColor: `${rt.color}50`, background: `${rt.color}08` }}
              >
                <div className="roomTop">
                  <div className="roomName">{room.shortName || room.id}</div>
                  <div className="roomMark" style={{ color: rt.color }}>
                    {rt.word} <span className="arrow">{arrow}</span>
                  </div>
                </div>
                {emphasis ? (
                  <div className="roomEmph" style={{ borderColor: `${rt.color}40`, color: rt.color }}>
                    {emphasis}
                  </div>
                ) : null}
                {room.status !== 'optimal' && room.behavioralState ? <div className="roomMsg">{room.behavioralState}</div> : null}
              </button>
            )
          })}
        </main>
      </div>

      {selectedRoom ? (
        <motion.div className="backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.16 }}>
          <motion.div
            className="modal"
            initial={{ opacity: 0, y: 10, scale: 0.99 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.18 }}
            role="dialog"
            aria-modal="true"
            aria-label="Subsystem details"
          >
            <div className="modalHeader">
              <div style={{ minWidth: 0 }}>
                <div className="modalTitle">{selectedRoom.shortName || selectedRoom.id}</div>
                <div className="modalState" style={{ color: roomTone(selectedRoom.status).color }}>
                  {roomTone(selectedRoom.status).word}
                </div>
              </div>
              <button className="closeBtn" onClick={() => setSelectedRoomId(null)} aria-label="Close details">
                Close
              </button>
            </div>

            <div className="modalBody">
              <div className="detailLeft">
                <div className="detailBlock">
                  <div className="detailLabel">System message</div>
                  <div className="detailText">{selectedRoom.behavioralState || systemMessage}</div>
                </div>

                <div className="detailBlock">
                  <div className="detailLabel">Context</div>
                  <div className="detailText">
                    {nowLine} Window: {projectedWindow}. Confidence: {confidenceLabel}.
                  </div>
                </div>

                <details className="detailDisclosure">
                  <summary>Supporting signals</summary>
                  <div className="signalsGrid">
                    <div className="signalCell">
                      <div className="signalKey">Drift</div>
                      <div className="signalVal">{Math.round(clamp01(interpolated.interpolatedDrift) * 100)}%</div>
                    </div>
                    <div className="signalCell">
                      <div className="signalKey">Stability</div>
                      <div className="signalVal">{Math.round(clamp01(interpolated.interpolatedStability) * 100)}%</div>
                    </div>
                    <div className="signalCell">
                      <div className="signalKey">Coherence</div>
                      <div className="signalVal">{Math.round(clamp01(interpolated.interpolatedCoherence) * 100)}%</div>
                    </div>
                    <div className="signalCell">
                      <div className="signalKey">Confidence</div>
                      <div className="signalVal">{Math.round(clamp01(resolvedSystemData.confidence) * 100)}%</div>
                    </div>
                  </div>
                </details>

                <details className="detailDisclosure">
                  <summary>System note</summary>
                  <div className="detailText">
                    {recommendationNote.reason}
                    <div style={{ marginTop: 10 }}>If unchanged: {recommendationNote.ifUnchanged}</div>
                  </div>
                </details>
              </div>

              <div className="detailRight">
                <div className="vizBox" style={{ borderColor: `${tone.color}2a` }}>
                  <div className="detailLabel">System geometry</div>
                  <div className="vizStage">
                    <TetrahedronField
                      data={{
                        interpolatedDrift: interpolated.interpolatedDrift,
                        interpolatedStability: interpolated.interpolatedStability,
                        interpolatedCoherence: interpolated.interpolatedCoherence,
                        interpolatedConfidence: resolvedSystemData.confidence,
                      }}
                      phase={phase}
                      phaseProgress={phaseProgress}
                      escalationLevel={tone.tone === 'RED' ? 5 : tone.tone === 'YELLOW' ? 3 : 1}
                      hasThresholdCrossed={tone.tone === 'RED'}
                      isApproachingFailure={tone.tone !== 'GREEN'}
                      decisionRationale={actionDecision?.decisionRationale || null}
                    />
                  </div>
                </div>

                <div className="vizBox" style={{ borderColor: `${tone.color}14` }}>
                  <div className="detailLabel">Facility structure</div>
                  <FacilityLinkLayer rooms={resolvedRooms} primaryDriver={primaryDriver} systemState={systemState} />
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      ) : null}

      <style jsx>{`
        .root {
          min-height: 100vh;
          background: #050607;
          color: #e2e8f0;
          font-family: system-ui, -apple-system, Segoe UI, sans-serif;
        }
        .frame {
          max-width: 1200px;
          margin: 0 auto;
          padding: 28px 18px 42px;
          display: grid;
          gap: 18px;
        }

        .systemPanel {
          border-radius: 16px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(5, 6, 7, 0.72);
          box-shadow: 0 34px 90px rgba(0, 0, 0, 0.58);
          padding: 20px 20px;
          display: grid;
          gap: 16px;
        }
        .systemTop {
          display: flex;
          justify-content: space-between;
          gap: 18px;
          align-items: flex-start;
          flex-wrap: wrap;
        }
        .systemHeadline {
          display: flex;
          gap: 14px;
          align-items: flex-start;
          min-width: 0;
          flex: 1;
        }
        .systemDot {
          width: 10px;
          height: 10px;
          border-radius: 999px;
          flex-shrink: 0;
          margin-top: 10px;
        }
        .systemTitle {
          font-size: 34px;
          line-height: 1.08;
          font-weight: 950;
          letter-spacing: -0.4px;
        }
        .systemSub {
          margin-top: 6px;
          font-size: 18px;
          color: rgba(226, 232, 240, 0.82);
          font-weight: 700;
        }

        .systemControls {
          display: flex;
          gap: 10px;
          align-items: center;
          flex-shrink: 0;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.06);
          border-radius: 999px;
          padding: 10px 12px;
        }
        .ctrlBtn {
          height: 34px;
          padding: 0 14px;
          border-radius: 999px;
          border: 1px solid rgba(255, 255, 255, 0.14);
          background: rgba(255, 255, 255, 0.03);
          color: #e2e8f0;
          font-weight: 900;
          cursor: pointer;
        }
        .ctrlGroup {
          display: flex;
          gap: 8px;
          align-items: center;
        }
        .ctrlLabel {
          font-size: 12px;
          color: rgba(226, 232, 240, 0.66);
          font-weight: 800;
          letter-spacing: 0.2px;
        }
        .ctrlValue {
          font-size: 12px;
          font-weight: 900;
          color: #e2e8f0;
        }
        .range {
          width: 120px;
        }
        .stepBtn {
          width: 34px;
          height: 34px;
          border-radius: 999px;
          border: 1px solid rgba(255, 255, 255, 0.12);
          background: rgba(255, 255, 255, 0.03);
          color: #e2e8f0;
          font-size: 18px;
          font-weight: 900;
          cursor: pointer;
          line-height: 1;
        }

        .systemMessage {
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(255, 255, 255, 0.03);
          padding: 14px 14px;
          font-size: 18px;
          font-weight: 800;
          line-height: 1.25;
        }

        .systemMeta {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 12px;
        }
        .metaItem {
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(255, 255, 255, 0.02);
          padding: 12px 12px;
          display: grid;
          gap: 6px;
        }
        .metaKey {
          font-size: 12px;
          color: rgba(226, 232, 240, 0.64);
          font-weight: 800;
        }
        .metaVal {
          font-size: 16px;
          font-weight: 950;
          letter-spacing: 0.2px;
        }

        .roomsGrid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 14px;
        }
        .roomCard {
          text-align: left;
          cursor: pointer;
          border-radius: 14px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(255, 255, 255, 0.02);
          padding: 16px 16px;
          display: grid;
          gap: 10px;
          box-shadow: 0 14px 44px rgba(0, 0, 0, 0.4);
          transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
        }
        .roomCard:hover {
          transform: translateY(-1px);
          border-color: rgba(226, 232, 240, 0.22);
          background: rgba(255, 255, 255, 0.03);
        }
        .roomTop {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 10px;
        }
        .roomName {
          font-size: 20px;
          font-weight: 950;
          letter-spacing: 0.2px;
        }
        .roomMark {
          font-size: 14px;
          font-weight: 950;
          letter-spacing: 0.4px;
          text-transform: uppercase;
          display: flex;
          gap: 8px;
          align-items: center;
        }
        .arrow {
          font-size: 16px;
          line-height: 1;
        }
        .roomEmph {
          display: inline-flex;
          width: fit-content;
          border-radius: 999px;
          border: 1px solid rgba(255, 255, 255, 0.14);
          padding: 6px 10px;
          font-size: 12px;
          font-weight: 950;
          letter-spacing: 0.4px;
          text-transform: uppercase;
        }
        .roomMsg {
          font-size: 16px;
          font-weight: 750;
          line-height: 1.25;
          color: rgba(226, 232, 240, 0.85);
        }

        .backdrop {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.72);
          backdrop-filter: blur(8px);
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 18px;
        }
        .modal {
          width: min(1100px, 100%);
          max-height: min(86vh, 860px);
          overflow: auto;
          border-radius: 16px;
          border: 1px solid rgba(255, 255, 255, 0.10);
          background: rgba(5, 6, 7, 0.88);
          box-shadow: 0 40px 140px rgba(0, 0, 0, 0.68);
        }
        .modalHeader {
          padding: 18px 18px;
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }
        .modalTitle {
          font-size: 22px;
          font-weight: 950;
        }
        .modalState {
          margin-top: 6px;
          font-size: 14px;
          font-weight: 950;
          letter-spacing: 0.4px;
          text-transform: uppercase;
        }
        .closeBtn {
          height: 38px;
          padding: 0 14px;
          border-radius: 999px;
          border: 1px solid rgba(255, 255, 255, 0.14);
          background: rgba(255, 255, 255, 0.03);
          color: #e2e8f0;
          font-weight: 900;
          cursor: pointer;
        }
        .modalBody {
          padding: 18px 18px 20px;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
        }
        .detailLeft,
        .detailRight {
          display: grid;
          gap: 14px;
        }
        .detailBlock {
          border-radius: 14px;
          border: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(255, 255, 255, 0.03);
          padding: 14px 14px;
          display: grid;
          gap: 8px;
        }
        .detailLabel {
          font-size: 12px;
          color: rgba(226, 232, 240, 0.65);
          font-weight: 900;
          letter-spacing: 0.4px;
          text-transform: uppercase;
        }
        .detailText {
          font-size: 16px;
          font-weight: 760;
          line-height: 1.25;
          color: rgba(226, 232, 240, 0.92);
        }
        .detailDisclosure {
          border-radius: 14px;
          border: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(255, 255, 255, 0.02);
          padding: 12px 12px;
        }
        details > summary {
          cursor: pointer;
          user-select: none;
          font-size: 13px;
          font-weight: 900;
          letter-spacing: 0.2px;
          color: rgba(226, 232, 240, 0.88);
        }
        .signalsGrid {
          margin-top: 12px;
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
        }
        .signalCell {
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(255, 255, 255, 0.02);
          padding: 12px 12px;
          display: grid;
          gap: 6px;
        }
        .signalKey {
          font-size: 12px;
          color: rgba(226, 232, 240, 0.62);
          font-weight: 800;
        }
        .signalVal {
          font-size: 18px;
          font-weight: 950;
        }
        .vizBox {
          border-radius: 14px;
          border: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(255, 255, 255, 0.02);
          padding: 14px 14px;
          display: grid;
          gap: 10px;
        }
        .vizStage {
          min-height: 340px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(0, 0, 0, 0.18);
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        @media (max-width: 980px) {
          .systemTitle {
            font-size: 28px;
          }
          .systemMeta {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
          .roomsGrid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
          .modalBody {
            grid-template-columns: 1fr;
          }
          .range {
            width: 92px;
          }
        }

        @media (max-width: 620px) {
          .roomsGrid {
            grid-template-columns: 1fr;
          }
          .systemControls {
            width: 100%;
            justify-content: space-between;
          }
        }
      `}</style>
    </div>
  )
}

export default TeslaAutopilotInterface
