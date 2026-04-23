import type { RunnerCycleRecord } from '@/lib/runnerOutputLoader'

export type GrowZoneId = 'VEG-A' | 'VEG-B' | 'FLOW-A' | 'FLOW-B' | 'FLOW-C'
export type ZoneSeverity = 'green' | 'yellow' | 'red'
export type SystemSeverity = ZoneSeverity
export type Progression = 'Increasing' | 'Spreading' | 'Stable'

export interface GrowScenarioFrame {
  step: number
  cycle: number
  unitId: string
  systemSeverity: SystemSeverity
  primaryZone: GrowZoneId | null
  location: GrowZoneId | null
  issue: string
  progression: Progression
  zones: Record<GrowZoneId, ZoneSeverity>
  details: {
    timestamp?: string
    drift?: number
    instability?: number
    riskLevel?: string
    trend?: string
    phase?: string
    divergence?: boolean
    driver?: string
    operatorMessage?: string
    raw: RunnerCycleRecord
  }
}

export interface GrowScenario {
  source: { kind: 'runner' | 'fallback'; unitId: string; recordCount: number }
  zones: GrowZoneId[]
  frames: GrowScenarioFrame[]
}

const GROW_ZONES: GrowZoneId[] = ['VEG-A', 'VEG-B', 'FLOW-A', 'FLOW-B', 'FLOW-C']

function asGrowZoneId(value: unknown): GrowZoneId | null {
  const text = String(value ?? '').trim().toUpperCase()
  if (text === 'VEG-A') return 'VEG-A'
  if (text === 'VEG-B') return 'VEG-B'
  if (text === 'FLOW-A') return 'FLOW-A'
  if (text === 'FLOW-B') return 'FLOW-B'
  if (text === 'FLOW-C') return 'FLOW-C'
  return null
}

function operatorStateToSeverity(state: unknown): ZoneSeverity | null {
  const s = String(state ?? '').trim().toUpperCase()
  if (s === 'INTERVENE') return 'red'
  if (s === 'WATCH') return 'yellow'
  if (s === 'NORMAL') return 'green'
  return null
}

function rawDriftAlertFromRunner(rec: RunnerCycleRecord): boolean {
  const operatorSeverity = operatorStateToSeverity(rec.state)
  if (operatorSeverity === 'red' || operatorSeverity === 'yellow') return true
  if (typeof rec.drift_alert === 'boolean') return rec.drift_alert

  const risk = String(rec.risk_level ?? rec.severity ?? '').toUpperCase()
  const phase = String(rec.phase ?? '').toUpperCase()

  if (risk.includes('HIGH') || risk.includes('MEDIUM')) return true
  if (phase.includes('ALERT') || phase.includes('WATCH') || phase.includes('DRIFT')) return true

  const drift = rec.structural_drift_score ?? 0
  const instab = rec.composite_instability ?? 0
  return drift >= 0.42 || instab >= 0.48 || Boolean(rec.divergence_alert)
}

function failureCycleEstimateFromRunner(rec: RunnerCycleRecord): number | null {
  if (typeof rec.failure_cycle === 'number' && Number.isFinite(rec.failure_cycle)) return rec.failure_cycle
  if (typeof rec.estimated_rul === 'number' && Number.isFinite(rec.estimated_rul)) return rec.cycle + rec.estimated_rul
  return null
}

function median(values: number[]): number | null {
  const kept = values.filter(v => Number.isFinite(v)).slice().sort((a, b) => a - b)
  if (kept.length === 0) return null
  const mid = Math.floor(kept.length / 2)
  if (kept.length % 2 === 1) return kept[mid]!
  return (kept[mid - 1]! + kept[mid]!) / 2
}

function severityRank(s: ZoneSeverity): number {
  if (s === 'red') return 2
  if (s === 'yellow') return 1
  return 0
}

function stableSeverityFromRunnerWithActionWindow(
  rec: RunnerCycleRecord,
  opts: { failureCycleEstimate: number | null; previousSeverity: ZoneSeverity }
): ZoneSeverity {
  const { failureCycleEstimate, previousSeverity } = opts

  if (failureCycleEstimate == null) {
    // Fallback: keep previous logic when we cannot estimate a failure horizon.
    return previousSeverity
  }

  const actionWindowCycles = Math.max(1, Number(rec.action_window_cycles ?? 100))
  const watchStart = failureCycleEstimate - 1.5 * actionWindowCycles
  const interveneStart = failureCycleEstimate - actionWindowCycles

  const rawAlert = rawDriftAlertFromRunner(rec)
  let candidate: ZoneSeverity = 'green'
  if (rawAlert && rec.cycle >= interveneStart) candidate = 'red'
  else if (rawAlert && rec.cycle >= watchStart) candidate = 'yellow'

  // Enforce monotonic escalation (no early noise / no flicker).
  const prevRank = severityRank(previousSeverity)
  const candRank = severityRank(candidate)
  if (candRank <= prevRank) return previousSeverity

  // No jumping directly to INTERVENE from NORMAL unless very late-stage.
  if (previousSeverity === 'green' && candidate === 'red') {
    const lateOverrideCycles = Math.max(1, Math.min(10, Math.floor(actionWindowCycles * 0.1)))
    if (rec.cycle >= failureCycleEstimate - lateOverrideCycles) return 'red'
    return 'yellow'
  }

  return candidate
}

function pickActiveUnit(records: RunnerCycleRecord[]): string {
  const unitIds = Array.from(new Set(records.map(r => r.unit_id))).sort()
  if (unitIds.length === 0) return 'unit_001'

  const scoreByUnit: Record<string, number> = {}
  for (const unitId of unitIds) scoreByUnit[unitId] = 0

  for (const rec of records) {
    const sev = rawDriftAlertFromRunner(rec) ? 'yellow' : 'green'
    const base = sev === 'yellow' ? 2 : 1
    const drift = rec.structural_drift_score ?? 0
    const instab = rec.composite_instability ?? 0
    scoreByUnit[rec.unit_id] = Math.max(scoreByUnit[rec.unit_id] ?? 0, base + drift * 0.5 + instab * 0.5)
  }

  return unitIds
    .slice()
    .sort((a, b) => {
      const da = scoreByUnit[a] ?? 0
      const db = scoreByUnit[b] ?? 0
      if (db !== da) return db - da
      return a < b ? -1 : a > b ? 1 : 0
    })[0]!
}

function clampIssueWords(phrase: string): string {
  const cleaned = phrase
    .replace(/[\r\n]+/g, ' ')
    .replace(/[.,;:!?()[\]{}"'\u2026]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  const words = cleaned.split(' ').filter(Boolean)
  const sliced = words.slice(0, 5)
  if (sliced.length >= 2) return sliced.join(' ')
  if (sliced.length === 1) return `${sliced[0]} anomaly`
  return 'System anomaly'
}

function mapDriverToIssue(rec: RunnerCycleRecord): string {
  const driver = String(rec.dominant_driver ?? rec.recommended_action ?? rec.trend ?? rec.phase ?? '').toLowerCase()
  const msg = String(rec.operator_message ?? '').toLowerCase()
  const text = `${driver} ${msg}`.trim()

  if (text.includes('air') || text.includes('flow')) return 'Airflow imbalance'
  if (text.includes('climate') || text.includes('temp') || text.includes('humid') || text.includes('vpd')) return 'Climate instability'
  if (text.includes('irrig') || text.includes('water') || text.includes('ec') || text.includes('ph')) return 'Irrigation variance'
  if (text.includes('canopy') || text.includes('stress')) return 'Canopy stress signal'
  if (text.includes('pressure')) return 'Pressure drift'
  if (text.includes('diverg') || rec.divergence_alert) return 'Flow instability'

  return clampIssueWords(String(rec.dominant_driver ?? rec.phase ?? rec.trend ?? 'System anomaly'))
}

function computeProgression(cycleRecords: RunnerCycleRecord[], idx: number, zones: Record<GrowZoneId, ZoneSeverity>): Progression {
  const redCount = Object.values(zones).filter(z => z === 'red').length
  const yellowCount = Object.values(zones).filter(z => z === 'yellow').length
  if (redCount >= 1 && yellowCount >= 1) return 'Spreading'

  const window = cycleRecords.slice(Math.max(0, idx - 3), idx + 1)
  const drifts = window.map(r => r.structural_drift_score ?? 0)
  if (drifts.length >= 2) {
    const delta = drifts[drifts.length - 1]! - drifts[0]!
    if (delta >= 0.045) return 'Increasing'
  }

  return 'Stable'
}

function zonesAllGreen(): Record<GrowZoneId, ZoneSeverity> {
  return {
    'VEG-A': 'green',
    'VEG-B': 'green',
    'FLOW-A': 'green',
    'FLOW-B': 'green',
    'FLOW-C': 'green',
  }
}

function applyZoneEscalation(
  baseZones: Record<GrowZoneId, ZoneSeverity>,
  primaryZone: GrowZoneId,
  systemSeverity: ZoneSeverity,
  spread: boolean
): Record<GrowZoneId, ZoneSeverity> {
  const zones = { ...baseZones }
  zones[primaryZone] = systemSeverity

  if (spread && systemSeverity === 'red') {
    const neighbor: GrowZoneId = primaryZone === 'FLOW-B' ? 'FLOW-A' : 'FLOW-B'
    if (zones[neighbor] === 'green') zones[neighbor] = 'yellow'
  }
  return zones
}

export function adaptRunnerCyclesToGrowScenario(records: RunnerCycleRecord[]): GrowScenario {
  const unitId = pickActiveUnit(records)
  const cycles = records
    .filter(r => r.unit_id === unitId)
    .slice()
    .sort((a, b) => (a.cycle ?? 0) - (b.cycle ?? 0))

  // Deterministic primary zone defaults to FLOW-B (used when runner lacks a zone).
  const defaultPrimaryZone: GrowZoneId = 'FLOW-B'

  const frames: GrowScenarioFrame[] = []
  let zones = zonesAllGreen()
  let severity: ZoneSeverity = 'green'
  const failureCycleHistory: number[] = []
  const failureCycleHistoryWindow = 7

  for (let i = 0; i < cycles.length; i++) {
    const rec = cycles[i]!
    const operatorSeverity = operatorStateToSeverity(rec.state)
    const operatorZone = asGrowZoneId(rec.zone) ?? defaultPrimaryZone

    const failureEstimate = failureCycleEstimateFromRunner(rec)
    if (failureEstimate != null) {
      failureCycleHistory.push(failureEstimate)
      if (failureCycleHistory.length > failureCycleHistoryWindow) failureCycleHistory.shift()
    }
    const stabilizedFailureCycle = median(failureCycleHistory)

    if (operatorSeverity != null) {
      severity = operatorSeverity
    } else {
      severity = stableSeverityFromRunnerWithActionWindow(rec, {
        failureCycleEstimate: stabilizedFailureCycle,
        previousSeverity: severity,
      })
    }

    // EARLY SIGNAL exists only in details: mild drift without operator escalation.
    const earlySignal = (rec.structural_drift_score ?? 0) >= 0.18 && severity === 'green'
    const systemSeverity: ZoneSeverity = earlySignal ? 'green' : severity

    const worsening =
      i >= 2 &&
      (cycles[i - 1]!.structural_drift_score ?? 0) <= (rec.structural_drift_score ?? 0) &&
      (cycles[i - 2]!.structural_drift_score ?? 0) <= (cycles[i - 1]!.structural_drift_score ?? 0)

    const progressionFromRunner = String(rec.progression ?? '').trim()
    const runnerProgression: Progression | null =
      progressionFromRunner === 'Increasing' || progressionFromRunner === 'Spreading' || progressionFromRunner === 'Stable'
        ? (progressionFromRunner as Progression)
        : null

    const spread =
      runnerProgression === 'Spreading' ||
      Boolean(rec.divergence_alert) ||
      (systemSeverity === 'red' && worsening)

    zones = applyZoneEscalation(zonesAllGreen(), operatorZone, systemSeverity, spread)

    frames.push({
      step: i,
      cycle: rec.cycle,
      unitId,
      systemSeverity,
      primaryZone: systemSeverity === 'green' ? null : operatorZone,
      location: systemSeverity === 'green' ? null : operatorZone,
      issue: mapDriverToIssue(rec),
      progression: runnerProgression ?? computeProgression(cycles, i, zones),
      zones,
      details: {
        timestamp: rec.timestamp,
        drift: rec.structural_drift_score,
        instability: rec.composite_instability,
        riskLevel: String(rec.risk_level ?? ''),
        trend: rec.trend,
        phase: rec.phase,
        divergence: rec.divergence_alert,
        driver: rec.dominant_driver,
        operatorMessage: rec.operator_message,
        raw: rec,
      },
    })
  }

  return {
    source: { kind: 'runner', unitId, recordCount: frames.length },
    zones: GROW_ZONES,
    frames,
  }
}
