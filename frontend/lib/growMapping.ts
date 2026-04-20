/**
 * Grow-operation mapping layer.
 *
 * Links subsystem issues to operator actions using rule-based mapping.
 * All language is grow-specific and actionable.
 */

export type GrowSubsystem =
  | 'climate'
  | 'humidity'
  | 'airflow'
  | 'lighting'
  | 'plant_stress'

export type GrowStage =
  | 'optimal'
  | 'stress_forming'
  | 'high_risk'

export type UrgencyLevel = 'Monitor' | 'Prepare' | 'Act Now'
export type ConfidenceLevel = 'LOW' | 'MEDIUM' | 'HIGH'

export interface GrowAction {
  subsystem: GrowSubsystem
  action: string
  detail: string
}

export const SUBSYSTEM_LABELS: Record<GrowSubsystem, string> = {
  climate: 'Climate Control',
  humidity: 'Humidity Regulation',
  airflow: 'Airflow',
  lighting: 'Lighting Load',
  plant_stress: 'Plant Stress Indicators',
}

export const SUBSYSTEM_ACTIONS: Record<GrowSubsystem, string[]> = {
  climate: [
    'Check HVAC setpoints and thermal load balance',
    'Inspect cooling loop and airflow consistency',
    'Verify temperature sensor calibration',
  ],
  humidity: [
    'Check dehumidifier capacity and drain lines',
    'Inspect vapor pressure deficit (VPD) alignment',
    'Verify humidity sensor placement',
  ],
  airflow: [
    'Inspect circulation fans and filter condition',
    'Check CO₂ distribution uniformity',
    'Verify intake / exhaust balance',
  ],
  lighting: [
    'Check diode thermal runoff and spectrum drift',
    'Inspect PPFD uniformity across canopy',
    'Verify timer and dimmer settings',
  ],
  plant_stress: [
    'Inspect canopy for wilting or tip burn',
    'Check root zone moisture and EC trend',
    'Review irrigation schedule against transpiration rate',
  ],
}

export function recommendedActionsForSubsystem(
  subsystem: GrowSubsystem
): string[] {
  return SUBSYSTEM_ACTIONS[subsystem] ?? ['Continue monitoring']
}

export function primarySubsystemFromDriver(driver: string): GrowSubsystem {
  const d = driver.toLowerCase()
  if (d.includes('temp') || d.includes('thermal') || d.includes('heat') || d.includes('cool')) return 'climate'
  if (d.includes('humid') || d.includes('vpd') || d.includes('moist')) return 'humidity'
  if (d.includes('air') || d.includes('co2') || d.includes('vent') || d.includes('flow')) return 'airflow'
  if (d.includes('light') || d.includes('ppfd') || d.includes('spectrum') || d.includes('diode')) return 'lighting'
  return 'plant_stress'
}

export function growStageFromSystem(
  structuralDrift: number,
  severity: string
): GrowStage {
  if (severity === 'HIGH' || structuralDrift > 0.7) return 'high_risk'
  if (severity === 'ELEVATED' || structuralDrift > 0.4) return 'stress_forming'
  return 'optimal'
}

export function growStageLabel(stage: GrowStage): string {
  switch (stage) {
    case 'optimal': return 'Optimal'
    case 'stress_forming': return 'Stress Forming'
    case 'high_risk': return 'High Risk'
  }
}

export function growStageColor(stage: GrowStage): string {
  switch (stage) {
    case 'optimal': return '#7cb342'
    case 'stress_forming': return '#ff9e6d'
    case 'high_risk': return '#e05c5c'
  }
}

export function urgencyFromStage(stage: GrowStage, trajectory: string): UrgencyLevel {
  if (stage === 'high_risk') return 'Act Now'
  if (stage === 'stress_forming' && trajectory === 'Degrading') return 'Act Now'
  if (stage === 'stress_forming') return 'Prepare'
  return 'Monitor'
}

export function confidenceFromLabel(label: 'LOW' | 'MEDIUM' | 'HIGH'): ConfidenceLevel {
  return label
}

export function timeToImpactText(
  structuralDrift: number,
  stage: GrowStage,
  onsetMinutes: number | null
): string {
  if (stage === 'optimal') return 'No immediate environmental risk'
  if (stage === 'stress_forming') {
    const hours = onsetMinutes ? Math.round(onsetMinutes / 60) : 2
    return `~${hours}–${hours + 2} hours before VPD imbalance impacts plant health`
  }
  const hours = onsetMinutes ? Math.max(1, Math.round(onsetMinutes / 120)) : 1
  return `~${hours} hour${hours > 1 ? 's' : ''} before mold risk or heat stress increases significantly`
}

export function situationText(
  primaryDriver: string,
  stage: GrowStage,
  subsystem: GrowSubsystem
): string {
  const sub = SUBSYSTEM_LABELS[subsystem]
  if (stage === 'optimal') return `Environmental conditions stable across all zones`
  if (stage === 'stress_forming') return `Environmental instability forming within ${sub.toLowerCase()}`
  return `Critical environmental shift detected in ${sub.toLowerCase()}`
}

export function cleanLanguagePass(
  oldText: string,
  driverRoom?: string
): string {
  // Replace generic structural language with grow-specific language
  let text = oldText
    .replace(/structural drift/gi, 'environmental instability')
    .replace(/structural separation/gi, 'environmental decoupling')
    .replace(/subsystem instability/gi, 'environmental imbalance')
    .replace(/transition/gi, 'stress threshold')
    .replace(/regime break/gi, 'environmental regime shift')
    .replace(/regime change/gi, 'environmental shift')
    .replace(/breakdown/gi, 'crop stress event')
    .replace(/failure approach/gi, 'critical stress phase')
    .replace(/coherence/gi, 'environmental coherence')
    .replace(/drift/gi, 'environmental drift')

  if (driverRoom) {
    text = text.replace(/the facility/gi, driverRoom)
    text = text.replace(/this zone/gi, driverRoom)
  }

  return text
}

export interface ZoneInfo {
  id: string
  name: string
  floor: number
  growStage: 'veg' | 'flower'
}

export const ZONES: ZoneInfo[] = [
  { id: 'veg-a', name: 'Zone A', floor: 1, growStage: 'veg' },
  { id: 'veg-b', name: 'Zone B', floor: 1, growStage: 'veg' },
  { id: 'flow-a', name: 'Zone C', floor: 2, growStage: 'flower' },
  { id: 'flow-b', name: 'Zone D', floor: 2, growStage: 'flower' },
  { id: 'flow-c', name: 'Zone E', floor: 2, growStage: 'flower' },
]

export function zoneNameFromRoomId(roomId: string): string {
  return ZONES.find(z => z.id === roomId)?.name ?? roomId.toUpperCase()
}
