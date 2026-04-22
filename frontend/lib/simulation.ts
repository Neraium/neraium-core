/**
 * Live simulation engine for the Louddpax × Neraium dashboard.
 *
 * Generates continuous, realistic facility telemetry with:
 * - light cycle simulation (18/6 veg, 12/12 flower)
 * - irrigation events and recovery windows
 * - sensor drift and noise
 * - structural coherence computation
 * - historical arrays for sparklines
 *
 * All values are deterministic given a seed + time.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { DecisionUIState, Severity, DegradationStage, Trajectory } from '@/lib/decisionToUI'
import { DEMO_SCENARIOS } from '@/lib/demoScenarios'

const SIMULATION_EPOCH_MS = Date.UTC(2026, 0, 1, 8, 30, 0)

/* ─── Types ───────────────────────────────────────────────────────────────── */

export interface LiveSensors {
  temperature_f: number
  humidity_rh: number
  co2_ppm: number
  vpd_kpa: number
  ph: number
  ec_ms: number
  ppfd_umol: number
  irrigation_ml: number
}

export interface LiveRoom {
  id: string
  name: string
  shortName: string
  floor: number
  stage: string
  plantCount: number
  capacity: number
  growStage: 'veg' | 'flower'
  cycle: 'day' | 'night'
  cycleProgress: number // 0–1
  status: 'optimal' | 'warning' | 'critical'
  sensors: LiveSensors
  histories: Record<keyof LiveSensors, number[]>
  behavioralState: string
  driftContribution: number // 0–1
  confidence: number // 0–1
  trend: 'rising' | 'falling' | 'stable'
  trendMagnitude: number
  lastIrrigationMinutes: number
  recoveryProgress: number // 0–1
}

export interface LiveSystem {
  timestamp: Date
  scenarioIdx: number
  scenarioProgress: number // 0–1 through current scenario

  // System-level
  structuralDrift: number
  coherence: number // 0–1
  stability: number // 0–1
  onsetMinutes: number | null

  severity: Severity
  stage: DegradationStage
  trajectory: Trajectory
  confidenceLabel: 'LOW' | 'MEDIUM' | 'HIGH'

  // Histories
  driftHistory: number[]
  coherenceHistory: number[]
  instabilityHistory: number[]

  // Derived
  primaryDriver: string
  explanation: string
  operatorFocus: string
  pathOutlook: string

  rooms: LiveRoom[]
}

/* ─── Deterministic noise ─────────────────────────────────────────────────── */

function hash(seed: number): number {
  let x = Math.sin(seed * 127.1 + 311.7) * 43758.5453
  return x - Math.floor(x)
}

function smoothNoise(seed: number, t: number): number {
  const i = Math.floor(t)
  const f = t - i
  const a = hash(seed + i)
  const b = hash(seed + i + 1)
  return a + (b - a) * f * f * (3 - 2 * f) // smoothstep
}

function boundedNoise(seed: number, t: number, min: number, max: number): number {
  return min + smoothNoise(seed, t) * (max - min)
}

/* ─── Cycle computation ───────────────────────────────────────────────────── */

function computeCycle(growStage: 'veg' | 'flower', minutesIntoDay: number): { cycle: 'day' | 'night'; progress: number } {
  const dayLength = growStage === 'veg' ? 18 * 60 : 12 * 60 // minutes
  const cycleLen = 24 * 60
  const phase = minutesIntoDay % cycleLen
  const isDay = phase < dayLength
  const progress = isDay ? phase / dayLength : (phase - dayLength) / (cycleLen - dayLength)
  return { cycle: isDay ? 'day' : 'night', progress }
}

/* ─── Sensor baselines by cycle ───────────────────────────────────────────── */

function sensorBaseline(
  growStage: 'veg' | 'flower',
  cycle: 'day' | 'night',
  cycleProgress: number,
  seed: number,
  t: number
): LiveSensors {
  const isVeg = growStage === 'veg'
  const isDay = cycle === 'day'

  // Day/night offsets
  const tempOffset = isDay ? 2.5 : -1.2
  const rhOffset = isDay ? -5 : 8
  const co2Offset = isDay ? 400 : -200
  const vpdOffset = isDay ? 0.25 : -0.15
  const ppfdOffset = isDay ? (isVeg ? 580 : 720) : 0

  // Transition smoothing at cycle boundaries
  const transition = Math.sin(cycleProgress * Math.PI) // peak in middle of phase

  return {
    temperature_f: boundedNoise(seed + 1, t, 70, 76) + tempOffset + transition * 0.5,
    humidity_rh: boundedNoise(seed + 2, t, 55, 65) + rhOffset - transition * 2,
    co2_ppm: boundedNoise(seed + 3, t, 1200, 1500) + co2Offset + transition * 50,
    vpd_kpa: boundedNoise(seed + 4, t, 0.9, 1.3) + vpdOffset + transition * 0.05,
    ph: boundedNoise(seed + 5, t, 5.8, 6.3),
    ec_ms: boundedNoise(seed + 6, t, 1.4, 1.9),
    ppfd_umol: ppfdOffset + (isDay ? boundedNoise(seed + 7, t, -30, 30) : 0),
    irrigation_ml: isDay && cycleProgress > 0.3 && cycleProgress < 0.7
      ? boundedNoise(seed + 8, t, 240, 290)
      : 0,
  }
}

/* ─── Scenario drift injection ────────────────────────────────────────────── */

function applyScenarioDrift(
  base: LiveSensors,
  scenarioIdx: number,
  scenarioProgress: number,
  seed: number,
  t: number
): LiveSensors {
  const narrativeProgress = (scenarioIdx + scenarioProgress) / DEMO_SCENARIOS.length
  const driftIntensity = Math.max(0, (narrativeProgress - 0.15) * 1.4)

  const tempDrift = driftIntensity * (12 + smoothNoise(seed + 100, t * 0.1) * 8)
  const rhDrift = -driftIntensity * (15 + smoothNoise(seed + 101, t * 0.1) * 10)
  const co2Drift = -driftIntensity * (400 + smoothNoise(seed + 102, t * 0.1) * 300)
  const vpdDrift = driftIntensity * (1.0 + smoothNoise(seed + 103, t * 0.1) * 0.6)
  const phDrift = driftIntensity * (0.4 + smoothNoise(seed + 104, t * 0.1) * 0.3)
  const ecDrift = driftIntensity * (0.5 + smoothNoise(seed + 105, t * 0.1) * 0.4)

  return {
    temperature_f: base.temperature_f + tempDrift,
    humidity_rh: Math.max(20, Math.min(90, base.humidity_rh + rhDrift)),
    co2_ppm: Math.max(300, base.co2_ppm + co2Drift),
    vpd_kpa: Math.max(0.2, base.vpd_kpa + vpdDrift),
    ph: Math.max(5.0, Math.min(7.5, base.ph + phDrift)),
    ec_ms: Math.max(0.5, base.ec_ms + ecDrift),
    ppfd_umol: base.ppfd_umol,
    irrigation_ml: base.irrigation_ml,
  }
}

/* ─── Room configuration ──────────────────────────────────────────────────── */

const ROOM_CONFIGS = [
  { id: 'veg-a',  name: 'Vegetative Zone A', shortName: 'VEG-A',  floor: 1, stage: 'Vegetative (Week 3-4)', plantCount: 950,  capacity: 1000, growStage: 'veg' as const,    seed: 1000 },
  { id: 'veg-b',  name: 'Vegetative Zone B', shortName: 'VEG-B',  floor: 1, stage: 'Vegetative (Week 2-3)', plantCount: 920,  capacity: 1000, growStage: 'veg' as const,    seed: 2000 },
  { id: 'flow-a', name: 'Flowering Zone A',  shortName: 'FLOW-A', floor: 2, stage: 'Flowering (Week 4-5)', plantCount: 1150, capacity: 1200, growStage: 'flower' as const, seed: 3000 },
  { id: 'flow-b', name: 'Flowering Zone B',  shortName: 'FLOW-B', floor: 2, stage: 'Flowering (Week 5-6)', plantCount: 1100, capacity: 1200, growStage: 'flower' as const, seed: 4000 },
  { id: 'flow-c', name: 'Flowering Zone C',  shortName: 'FLOW-C', floor: 2, stage: 'Flowering (Week 3-4)', plantCount: 1080, capacity: 1200, growStage: 'flower' as const, seed: 5000 },
]

/* ─── Behavioral state derivation ─────────────────────────────────────────── */

function behavioralState(sensors: LiveSensors, driftContribution: number): string {
  if (driftContribution > 0.7) {
    if (sensors.temperature_f > 85) return 'Thermal runaway'
    if (sensors.humidity_rh < 40) return 'Desiccation risk'
    if (sensors.vpd_kpa > 2.2) return 'VPD critical'
    if (sensors.co2_ppm < 800) return 'CO₂ depletion'
    return 'Breakdown forming'
  }
  if (driftContribution > 0.35) {
    if (sensors.temperature_f > 80) return 'Heat accumulation'
    if (sensors.humidity_rh < 48) return 'Drying trend'
    if (sensors.vpd_kpa > 1.6) return 'VPD elevated'
    return 'Recovery lag'
  }
  if (sensors.irrigation_ml > 0) return 'Post-irrigation'
  if (sensors.ppfd_umol > 0) return 'Active photosynthesis'
  return 'Steady state'
}

/* ─── Simulation tick ─────────────────────────────────────────────────────── */

function simulateTick(
  scenarioIdx: number,
  scenarioProgress: number,
  simTime: number, // total simulation minutes elapsed
  prevRooms: LiveRoom[] | null
): LiveSystem {
  const narrativeProgress = (scenarioIdx + scenarioProgress) / DEMO_SCENARIOS.length
  const baseState = DEMO_SCENARIOS[scenarioIdx].state

  const minutesIntoDay = (simTime % (24 * 60))

  // System-level metrics
  const structuralDrift = Math.min(1, baseState.driftChart.dataPoints[baseState.driftChart.dataPoints.length - 1]?.driftScore ?? 0)
  const coherence = Math.max(0, 1 - structuralDrift * 1.3)
  const stability = Math.max(0, 1 - structuralDrift * 0.9)
  const onsetMinutes = narrativeProgress > 0.15 ? Math.round(narrativeProgress * 180) : null

  // Histories (maintain 30 points)
  const historyLength = 30

  const rooms: LiveRoom[] = ROOM_CONFIGS.map((cfg, i) => {
    const { cycle, progress } = computeCycle(cfg.growStage, minutesIntoDay + cfg.seed)
    const t = simTime * 0.05 + cfg.seed

    const base = sensorBaseline(cfg.growStage, cycle, progress, cfg.seed, t)
    const sensors = applyScenarioDrift(base, scenarioIdx, scenarioProgress, cfg.seed, t)

    // Per-room drift contribution varies by scenario
    let driftContribution = 0.05 + smoothNoise(cfg.seed + 500, t * 0.2) * 0.1
    if (cfg.id === 'flow-b') {
      driftContribution += narrativeProgress * 0.85
    } else if (cfg.id === 'flow-a') {
      driftContribution += Math.max(0, narrativeProgress - 0.3) * 0.6
    } else if (cfg.id === 'flow-c') {
      driftContribution += Math.max(0, narrativeProgress - 0.5) * 0.4
    } else if (cfg.id === 'veg-b') {
      driftContribution += Math.max(0, narrativeProgress - 0.6) * 0.3
    }
    driftContribution = Math.min(1, driftContribution)

    const status: LiveRoom['status'] =
      driftContribution > 0.7 ? 'critical'
      : driftContribution > 0.35 ? 'warning'
      : 'optimal'

    // Build or extend histories
    const histories: LiveRoom['histories'] = {
      temperature_f: [], humidity_rh: [], co2_ppm: [], vpd_kpa: [],
      ph: [], ec_ms: [], ppfd_umol: [], irrigation_ml: [],
    }

    if (prevRooms && prevRooms[i]) {
      Object.keys(histories).forEach((k) => {
        const key = k as keyof LiveSensors
        const prev = prevRooms[i].histories[key]
        histories[key] = [...prev.slice(-(historyLength - 1)), sensors[key]]
      })
    } else {
      // Seed history backwards
      Object.keys(histories).forEach((k) => {
        const key = k as keyof LiveSensors
        const arr: number[] = []
        for (let h = historyLength - 1; h >= 0; h--) {
          const ht = (simTime - h) * 0.05 + cfg.seed
          const hBase = sensorBaseline(cfg.growStage, cycle, progress, cfg.seed, ht)
          const hSensors = applyScenarioDrift(hBase, scenarioIdx, Math.max(0, scenarioProgress - h * 0.001), cfg.seed, ht)
          arr.push(hSensors[key])
        }
        histories[key] = arr
      })
    }

    // Trend from history
    const tempHist = histories.temperature_f
    const tempTrend = tempHist[tempHist.length - 1] - tempHist[Math.max(0, tempHist.length - 6)]
    const trend: LiveRoom['trend'] = Math.abs(tempTrend) < 0.3 ? 'stable' : tempTrend > 0 ? 'rising' : 'falling'
    const trendMagnitude = Math.abs(tempTrend)

    // Irrigation / recovery
    const lastIrrigationMinutes = sensors.irrigation_ml > 0 ? 0 : Math.round(smoothNoise(cfg.seed + 600, t) * 120)
    const recoveryProgress = Math.min(1, lastIrrigationMinutes / 45)

    return {
      id: cfg.id,
      name: cfg.name,
      shortName: cfg.shortName,
      floor: cfg.floor,
      stage: cfg.stage,
      plantCount: cfg.plantCount,
      capacity: cfg.capacity,
      growStage: cfg.growStage,
      cycle,
      cycleProgress: progress,
      status,
      sensors,
      histories,
      behavioralState: behavioralState(sensors, driftContribution),
      driftContribution,
      confidence: Math.max(0, 1 - driftContribution),
      trend,
      trendMagnitude,
      lastIrrigationMinutes,
      recoveryProgress,
    }
  })

  // Primary driver = room with highest drift contribution
  const driverRoom = rooms.reduce((a, b) => a.driftContribution > b.driftContribution ? a : b)
  const primaryDriver = driverRoom.name

  // Intelligence text
  let explanation: string
  let operatorFocus: string
  let pathOutlook: string

  if (narrativeProgress < 0.15) {
    explanation = 'All subsystems operating within nominal parameters. Structural coherence is strong and stable across the facility.'
    operatorFocus = 'Continue standard monitoring protocols. Next irrigation cycle in 12 minutes.'
    pathOutlook = 'Stable continuation expected. No intervention required.'
  } else if (narrativeProgress < 0.35) {
    explanation = `Early structural shift detected in ${primaryDriver}. Humidity decay remains slower than expected following irrigation, increasing separation between climate recovery and plant-layer response.`
    operatorFocus = `Inspect ${primaryDriver} climate setpoints. Verify VPD target alignment with current growth stage.`
    pathOutlook = 'Drift is contained to a single zone. Correction likely within one cycle.'
  } else if (narrativeProgress < 0.55) {
    explanation = `${primaryDriver} is now persistently off-baseline. The relationship between climate response and irrigation recovery is degrading. Coherence has weakened to moderate levels.`
    operatorFocus = `Adjust ${primaryDriver} dehumidification rate. Check for blocked airflow or failing circulation fan.`
    pathOutlook = 'Instability is forming. Without intervention, expect escalation within 45 minutes.'
  } else if (narrativeProgress < 0.75) {
    explanation = `System-wide coherence has dropped significantly. ${primaryDriver} is driving structural separation. Multiple subsystems are beginning to decouple from baseline behavior.`
    operatorFocus = `Immediate attention required in ${primaryDriver}. Reduce irrigation frequency and increase air turnover. Escalate to facility lead.`
    pathOutlook = 'Critical transition probable. Intervention window narrowing.'
  } else {
    explanation = `Critical regime break in ${primaryDriver}. Thermal runaway combined with VPD decoupling indicates equipment failure or severe environmental shift. Coherence has collapsed.`
    operatorFocus = `EMERGENCY PROTOCOL: Evacuate ${primaryDriver} if possible. Cut irrigation. Increase exhaust to maximum. Contact maintenance immediately.`
    pathOutlook = 'Failure imminent without immediate corrective action.'
  }

  // System histories
  const prevDrift = prevRooms?.[0]?.histories.temperature_f ?? []
  const driftHistory = prevDrift.length > 0
    ? [...prevDrift.slice(-(historyLength - 1)), structuralDrift]
    : Array.from({ length: historyLength }, (_, i) => Math.max(0, structuralDrift - (historyLength - i) * 0.01))

  const coherenceHistory = driftHistory.map(d => Math.max(0, 1 - d * 1.3))
  const instabilityHistory = driftHistory.map(d => d * 0.8 + smoothNoise(999, d * 10) * 0.1)

  const now = new Date(SIMULATION_EPOCH_MS + Math.round(simTime * 60_000) + scenarioIdx * 60_000)

  return {
    timestamp: now,
    scenarioIdx,
    scenarioProgress,
    structuralDrift,
    coherence,
    stability,
    onsetMinutes,
    severity: baseState.statusHeader.severity,
    stage: baseState.statusHeader.degradationStage,
    trajectory: baseState.statusHeader.trajectory,
    confidenceLabel: baseState.statusHeader.confidence,
    driftHistory,
    coherenceHistory,
    instabilityHistory,
    primaryDriver,
    explanation,
    operatorFocus,
    pathOutlook,
    rooms,
  }
}

/* ─── Hook ────────────────────────────────────────────────────────────────── */

export function useLiveSimulation() {
  const [live, setLive] = useState<LiveSystem>(() =>
    simulateTick(0, 0, 0, null)
  )
  const [isPlaying, setIsPlaying] = useState(true)
  const [selectedRoomId, setSelectedRoomId] = useState<string | null>(null)

  const simRef = useRef({
    scenarioIdx: 0,
    scenarioProgress: 0,
    simTime: 0,
    lastTick: 0,
    prevRooms: null as LiveRoom[] | null,
  })

  useEffect(() => {
    let raf: number
    const loop = (now: number) => {
      const s = simRef.current
      if (isPlaying) {
        const dt = s.lastTick ? (now - s.lastTick) / 1000 : 0.1
        s.simTime += dt * 8 // 8x speed: 1 real second = 8 sim minutes
        s.scenarioProgress += dt * 0.08

        if (s.scenarioProgress >= 1) {
          s.scenarioProgress = 0
          s.scenarioIdx = (s.scenarioIdx + 1) % DEMO_SCENARIOS.length
        }
      }
      s.lastTick = now

      const next = simulateTick(s.scenarioIdx, s.scenarioProgress, s.simTime, s.prevRooms)
      s.prevRooms = next.rooms
      setLive(next)

      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [isPlaying])

  const stepScenario = useCallback((delta: number) => {
    const s = simRef.current
    s.scenarioIdx = (s.scenarioIdx + delta + DEMO_SCENARIOS.length) % DEMO_SCENARIOS.length
    s.scenarioProgress = 0
    const next = simulateTick(s.scenarioIdx, 0, s.simTime, s.prevRooms)
    s.prevRooms = next.rooms
    setLive(next)
  }, [])

  return {
    live,
    isPlaying,
    setIsPlaying,
    selectedRoomId,
    setSelectedRoomId,
    stepScenario,
  }
}
