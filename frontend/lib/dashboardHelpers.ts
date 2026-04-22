/**
 * Deterministic helpers for the Louddpax × Neraium dashboard.
 * All functions are pure — no randomness, no side effects.
 * Safe for SSR and hydration.
 */

import { DecisionUIState, Severity, DegradationStage, Trajectory } from '@/lib/decisionToUI'

/* ─── Color Palette ───────────────────────────────────────────────────────── */

export const PALETTE = {
  bg: '#050607',
  surface: '#0B0D0F',
  surfaceHighlight: '#111419',
  border: 'rgba(255,255,255,0.06)',
  borderActive: 'rgba(255,255,255,0.12)',
  textPrimary: '#E8EBF0',
  textSecondary: '#8B92A0',
  textMuted: '#4B5563',
  neon: '#A3E635',
  neonGlow: 'rgba(163,230,53,0.35)',
  blue: '#60A5FA',
  blueGlow: 'rgba(96,165,250,0.35)',
  orange: '#FF8A5C',
  orangeGlow: 'rgba(255,138,92,0.35)',
  red: '#EF4444',
  redGlow: 'rgba(239,68,68,0.40)',
}

/* ─── Severity / Stage Mapping ────────────────────────────────────────────── */

export function severityColor(sev: Severity): string {
  switch (sev) {
    case Severity.HIGH: return PALETTE.red
    case Severity.ELEVATED: return PALETTE.orange
    case Severity.MODERATE: return PALETTE.blue
    default: return PALETTE.neon
  }
}

export function severityGlow(sev: Severity): string {
  switch (sev) {
    case Severity.HIGH: return PALETTE.redGlow
    case Severity.ELEVATED: return PALETTE.orangeGlow
    case Severity.MODERATE: return PALETTE.blueGlow
    default: return PALETTE.neonGlow
  }
}

export function stageLabel(stage: DegradationStage): string {
  switch (stage) {
    case DegradationStage.BASELINE: return 'STABLE'
    case DegradationStage.EARLY_SHIFT: return 'DRIFTING'
    case DegradationStage.EMERGING: return 'UNSTABLE'
    case DegradationStage.PERSISTENT: return 'UNSTABLE'
    case DegradationStage.ACCELERATED: return 'CRITICAL'
    case DegradationStage.FAILURE_APPROACH: return 'CRITICAL'
  }
}

export function stageIndex(stage: DegradationStage): number {
  const order = [
    DegradationStage.BASELINE,
    DegradationStage.EARLY_SHIFT,
    DegradationStage.EMERGING,
    DegradationStage.PERSISTENT,
    DegradationStage.ACCELERATED,
    DegradationStage.FAILURE_APPROACH,
  ]
  return order.indexOf(stage)
}

export function trajectoryLabel(t: Trajectory): string {
  if (t === Trajectory.DEGRADING) return 'DETERIORATING'
  if (t === Trajectory.UNCERTAIN) return 'UNCERTAIN'
  return 'STABLE'
}

/* ─── Subsystem Derivation ────────────────────────────────────────────────── */

export interface Subsystem {
  id: string
  name: string
  icon: string
  state: string
  contribution: 'low' | 'medium' | 'high'
  explanation: string
  health: 'stable' | 'warning' | 'critical'
}

export function deriveSubsystems(state: DecisionUIState): Subsystem[] {
  const { statusHeader, decisionTrace, tetrahedron } = state
  const sev = statusHeader.severity
  const stage = statusHeader.degradationStage
  const idx = stageIndex(stage)

  const isHigh = sev === Severity.HIGH || sev === Severity.ELEVATED
  const isDrift = idx >= 1
  const isUnstable = idx >= 3

  const primary = decisionTrace.primaryFactor.toLowerCase()
  const contains = (s: string) => primary.includes(s)

  return [
    {
      id: 'climate',
      name: 'Climate',
      icon: '🌡',
      state: isUnstable ? 'Recovery lag' : isDrift ? 'Slight lag' : 'Balanced',
      contribution: isHigh && (contains('temp') || contains('humid') || contains('vpd')) ? 'high'
        : isDrift ? 'medium' : 'low',
      explanation: isUnstable
        ? 'Humidity not decaying normally after irrigation'
        : isDrift
        ? 'Temperature cycling slightly out of phase'
        : 'VPD holding within target range',
      health: isHigh && (contains('temp') || contains('humid')) ? 'critical' : isDrift ? 'warning' : 'stable',
    },
    {
      id: 'irrigation',
      name: 'Irrigation',
      icon: '💧',
      state: isUnstable ? 'Decoupling' : isDrift ? 'Flow irregular' : 'Steady',
      contribution: isHigh && (contains('water') || contains('irriga') || contains('drain')) ? 'high'
        : isDrift ? 'medium' : 'low',
      explanation: isUnstable
        ? 'Root zone moisture not responding to feed cycles'
        : isDrift
        ? 'Drain EC lagging behind feed EC'
        : 'Feed and drain in balance',
      health: isHigh && (contains('water') || contains('irriga')) ? 'critical' : isDrift ? 'warning' : 'stable',
    },
    {
      id: 'lighting',
      name: 'Lighting',
      icon: '☀',
      state: isUnstable ? 'Spectrum shift' : isDrift ? 'PPFD drift' : 'On target',
      contribution: isHigh && (contains('light') || contains('ppfd') || contains('spec')) ? 'high'
        : 'low',
      explanation: isUnstable
        ? 'Diode thermal runoff shifting spectrum'
        : isDrift
        ? 'PPFD uniformity degrading at edges'
        : 'Uniform canopy coverage',
      health: isHigh && contains('light') ? 'critical' : 'stable',
    },
    {
      id: 'airflow',
      name: 'Airflow',
      icon: '〰',
      state: isUnstable ? 'Stagnation' : isDrift ? 'CO2 stratifying' : 'Circulating',
      contribution: isHigh && (contains('co2') || contains('air') || contains('vent')) ? 'high'
        : isDrift ? 'medium' : 'low',
      explanation: isUnstable
        ? 'CO2 settling at canopy level, not mixing'
        : isDrift
        ? 'Ventilation rate below setpoint by 8%'
        : 'Air turnover maintaining target',
      health: isHigh && (contains('co2') || contains('air')) ? 'critical' : isDrift ? 'warning' : 'stable',
    },
    {
      id: 'rootzone',
      name: 'Root Zone',
      icon: '🌱',
      state: isUnstable ? 'Salt buildup' : isDrift ? 'pH wandering' : 'Healthy',
      contribution: isHigh && (contains('ph') || contains('ec') || contains('root') || contains('salt')) ? 'high'
        : isDrift ? 'medium' : 'low',
      explanation: isUnstable
        ? 'EC climbing faster than feed can compensate'
        : isDrift
        ? 'pH drifting toward alkaline overnight'
        : 'Nutrient uptake in optimal range',
      health: isHigh && (contains('ph') || contains('ec')) ? 'critical' : isDrift ? 'warning' : 'stable',
    },
  ]
}

/* ─── Intelligence Derivation ─────────────────────────────────────────────── */

export interface Intelligence {
  currentState: string
  onset: string
  primaryDriver: string
  explanation: string
  operatorFocus: string
  severity: Severity
}

export function deriveIntelligence(state: DecisionUIState): Intelligence {
  const { statusHeader, decisionTrace, actionPanel, tetrahedron } = state
  const stage = statusHeader.degradationStage
  const idx = stageIndex(stage)

  const currentState =
    idx <= 1 ? 'Early Drift'
    : idx <= 3 ? 'Instability'
    : 'Critical'

  const onset =
    idx === 0 ? 'No deviation detected'
    : idx === 1 ? '~18 minutes ago'
    : idx === 2 ? '~42 minutes ago'
    : idx === 3 ? '~1.2 hours ago'
    : idx === 4 ? '~2.1 hours ago'
    : '~3+ hours ago'

  const subsystems = deriveSubsystems(state)
  const primaryDriver = subsystems.find(s => s.contribution === 'high')?.name
    ?? subsystems.find(s => s.contribution === 'medium')?.name
    ?? 'Climate'

  const explanation =
    idx === 0
      ? 'All subsystems operating within normal parameters. No structural drift detected.'
      : idx === 1
      ? `Early structural shift detected. ${primaryDriver} showing first signs of decoupling from baseline behavior. Action window is open.`
      : idx <= 3
      ? `${primaryDriver} is now persistently off-baseline. The relationship between subsystems is degrading. Intervention recommended within the next cycle.`
      : `System-wide breakdown imminent. ${primaryDriver} has crossed critical threshold. Immediate operator action required.`

  const operatorFocus =
    idx === 0
      ? 'Continue monitoring. No action required.'
      : idx === 1
      ? `Inspect ${primaryDriver.toLowerCase()} settings. Verify sensor calibration.`
      : idx <= 3
      ? `Prioritize ${primaryDriver.toLowerCase()} adjustment. Check for equipment malfunction or environmental shift.`
      : `EMERGENCY: Adjust ${primaryDriver.toLowerCase()} immediately. Escalate to senior operator.`

  return {
    currentState,
    onset,
    primaryDriver,
    explanation,
    operatorFocus,
    severity: statusHeader.severity,
  }
}

/* ─── Deterministic "noise" for animation ─────────────────────────────────── */

export function pseudoNoise(seed: number, t: number): number {
  const x = Math.sin(seed * 12.9898 + t * 0.5) * 43758.5453
  return x - Math.floor(x)
}

export function breatheOffset(seed: number, t: number, amplitude: number = 1): number {
  return Math.sin(t * 0.8 + seed * 2.3) * amplitude
}

export function jitterOffset(seed: number, t: number, intensity: number): number {
  if (intensity <= 0) return 0
  const n1 = pseudoNoise(seed, t * 3)
  const n2 = pseudoNoise(seed + 100, t * 2.7)
  return (n1 - 0.5) * intensity * 4 + (n2 - 0.5) * intensity * 2
}

/* ─── Room Data ───────────────────────────────────────────────────────────── */

export interface RoomData {
  id: string
  name: string
  shortName: string
  floor: number
  status: 'optimal' | 'warning' | 'critical'
  stage: string
  sensors: {
    temperature_f: number
    humidity_rh: number
    co2_ppm: number
    vpd_kpa: number
    ph: number
    ec_ms: number
    ppfd_umol: number
    irrigation_ml: number
  }
  plantCount: number
  capacity: number
}

export const ROOMS: RoomData[] = [
  {
    id: 'veg-a',
    name: 'Vegetative Zone A',
    shortName: 'VEG-A',
    floor: 1,
    status: 'optimal',
    stage: 'Vegetative (Week 3-4)',
    sensors: { temperature_f: 75.5, humidity_rh: 62.3, co2_ppm: 1350, vpd_kpa: 1.1, ph: 6.1, ec_ms: 1.6, ppfd_umol: 600, irrigation_ml: 250 },
    plantCount: 950, capacity: 1000,
  },
  {
    id: 'veg-b',
    name: 'Vegetative Zone B',
    shortName: 'VEG-B',
    floor: 1,
    status: 'optimal',
    stage: 'Vegetative (Week 2-3)',
    sensors: { temperature_f: 74.8, humidity_rh: 63.1, co2_ppm: 1400, vpd_kpa: 1.05, ph: 6.05, ec_ms: 1.55, ppfd_umol: 620, irrigation_ml: 260 },
    plantCount: 920, capacity: 1000,
  },
  {
    id: 'flow-a',
    name: 'Flowering Zone A',
    shortName: 'FLOW-A',
    floor: 2,
    status: 'optimal',
    stage: 'Flowering (Week 4-5)',
    sensors: { temperature_f: 72.2, humidity_rh: 55.4, co2_ppm: 1500, vpd_kpa: 1.3, ph: 6.2, ec_ms: 1.8, ppfd_umol: 750, irrigation_ml: 280 },
    plantCount: 1150, capacity: 1200,
  },
  {
    id: 'flow-b',
    name: 'Flowering Zone B',
    shortName: 'FLOW-B',
    floor: 2,
    status: 'optimal',
    stage: 'Flowering (Week 5-6)',
    sensors: { temperature_f: 72.5, humidity_rh: 54.8, co2_ppm: 1480, vpd_kpa: 1.28, ph: 6.15, ec_ms: 1.75, ppfd_umol: 740, irrigation_ml: 275 },
    plantCount: 1100, capacity: 1200,
  },
  {
    id: 'flow-c',
    name: 'Flowering Zone C',
    shortName: 'FLOW-C',
    floor: 2,
    status: 'optimal',
    stage: 'Flowering (Week 3-4)',
    sensors: { temperature_f: 73.1, humidity_rh: 56.2, co2_ppm: 1520, vpd_kpa: 1.25, ph: 6.18, ec_ms: 1.78, ppfd_umol: 735, irrigation_ml: 282 },
    plantCount: 1080, capacity: 1200,
  },
]

export function deriveRoomStates(state: DecisionUIState): RoomData[] {
  const idx = stageIndex(state.statusHeader.degradationStage)
  const sev = state.statusHeader.severity

  return ROOMS.map((room, i) => {
    /* perturb sensor values based on scenario for realism */
    const drift = (seed: number) => pseudoNoise(seed + i * 37, idx * 0.5) * 0.3

    let status: RoomData['status'] = 'optimal'
    if (idx >= 5) status = 'critical'
    else if (idx >= 3 && (room.id === 'flow-b' || room.id === 'flow-a')) status = 'critical'
    else if (idx >= 2 && room.id === 'flow-b') status = 'warning'
    else if (idx >= 1 && room.id === 'flow-b') status = 'warning'
    else if (idx >= 4 && room.id === 'veg-b') status = 'warning'
    else if (idx >= 3 && room.id === 'flow-c') status = 'warning'

    const tempDelta = status === 'critical' ? 12 + drift(1) * 6 : status === 'warning' ? 5 + drift(1) * 3 : drift(1) * 2
    const rhDelta = status === 'critical' ? -18 + drift(2) * 8 : status === 'warning' ? -8 + drift(2) * 4 : drift(2) * 2
    const vpdDelta = status === 'critical' ? 1.0 + drift(3) * 0.4 : status === 'warning' ? 0.3 + drift(3) * 0.2 : drift(3) * 0.1
    const co2Delta = status === 'critical' ? -500 + drift(4) * 200 : status === 'warning' ? -150 + drift(4) * 80 : drift(4) * 50

    return {
      ...room,
      status,
      sensors: {
        temperature_f: Math.round((room.sensors.temperature_f + tempDelta) * 10) / 10,
        humidity_rh: Math.round((room.sensors.humidity_rh + rhDelta) * 10) / 10,
        co2_ppm: Math.round(room.sensors.co2_ppm + co2Delta),
        vpd_kpa: Math.round((room.sensors.vpd_kpa + vpdDelta) * 100) / 100,
        ph: Math.round((room.sensors.ph + (status === 'critical' ? 0.5 : 0) + drift(5) * 0.2) * 100) / 100,
        ec_ms: Math.round((room.sensors.ec_ms + (status === 'critical' ? 0.6 : 0) + drift(6) * 0.2) * 100) / 100,
        ppfd_umol: Math.round(room.sensors.ppfd_umol + (status === 'critical' ? -80 : 0) + drift(7) * 30),
        irrigation_ml: Math.round(room.sensors.irrigation_ml + drift(8) * 20),
      },
    }
  })
}

/* ─── Sparkline helpers ───────────────────────────────────────────────────── */

export function generateSparkline(values: number[], width: number, height: number): string {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const n = values.length
  return values.map((v, i) => {
    const x = (i / Math.max(n - 1, 1)) * width
    const y = height - ((v - min) / range) * height
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

export function generateSensorHistory(baseValue: number, seed: number, count: number = 20): number[] {
  const values: number[] = []
  let current = baseValue
  for (let i = 0; i < count; i++) {
    const noise = (pseudoNoise(seed, i * 0.3) - 0.5) * baseValue * 0.08
    current += noise
    values.push(current)
  }
  return values
}
