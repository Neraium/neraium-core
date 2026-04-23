import fs from 'node:fs/promises'
import path from 'node:path'

export type RunnerSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'UNKNOWN'

export interface RunnerCycleRecord {
  unit_id: string
  cycle: number
  timestamp?: string
  state?: string
  zone?: string
  progression?: string
  confidence?: number
  persistence?: number
  structural_drift_score?: number
  composite_instability?: number
  drift_alert?: boolean
  is_alert?: boolean
  trend?: string
  phase?: string
  risk_level?: RunnerSeverity | string
  estimated_rul?: number
  failure_cycle?: number
  action_window_cycles?: number
  operator_message?: string
  early_signal_detected?: boolean
  early_signal_confirmed?: boolean
  early_signal_confidence?: number
  divergence_alert?: boolean
  dominant_driver?: string
  recommended_urgency?: string
  severity?: string
  recommended_action?: string
}

function toNumber(value: string | undefined): number | undefined {
  if (!value) return undefined
  const n = Number(value)
  return Number.isFinite(n) ? n : undefined
}

function toBool(value: string | undefined): boolean | undefined {
  if (value === undefined) return undefined
  const v = value.trim().toLowerCase()
  if (v === 'true' || v === '1' || v === 'yes' || v === 'y') return true
  if (v === 'false' || v === '0' || v === 'no' || v === 'n') return false
  return undefined
}

function splitCsvLine(line: string): string[] {
  const out: string[] = []
  let current = ''
  let inQuotes = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!
    if (ch === '"') {
      const next = line[i + 1]
      if (inQuotes && next === '"') {
        current += '"'
        i++
        continue
      }
      inQuotes = !inQuotes
      continue
    }
    if (ch === ',' && !inQuotes) {
      out.push(current)
      current = ''
      continue
    }
    current += ch
  }
  out.push(current)
  return out
}

function parseCsv(text: string): Array<Record<string, string>> {
  const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0)
  if (lines.length < 2) return []
  const header = splitCsvLine(lines[0]!).map(h => h.trim())
  const rows: Array<Record<string, string>> = []
  for (let i = 1; i < lines.length; i++) {
    const parts = splitCsvLine(lines[i]!)
    const row: Record<string, string> = {}
    for (let j = 0; j < header.length; j++) {
      row[header[j]!] = (parts[j] ?? '').trim()
    }
    rows.push(row)
  }
  return rows
}

function rowToCycleRecord(row: Record<string, string>): RunnerCycleRecord | null {
  const unit = row.unit_id || row.asset_id || row.unit || row.unitId
  const cycleStr = row.cycle || row.step || row.frame
  const cycle = toNumber(cycleStr)
  if (!unit || cycle === undefined) return null

  const estimatedRul = toNumber(row.estimated_rul || row.rul_estimate || row.rul || row.remaining_life)
  const failureCycle = toNumber(row.failure_cycle) ?? (estimatedRul !== undefined ? cycle + estimatedRul : undefined)

  return {
    unit_id: unit,
    cycle,
    timestamp: row.timestamp || row.time || row.datetime,
    state: row.state || row.operator_state || undefined,
    zone: row.zone || row.dominant_zone || row.dominant_component || undefined,
    progression: row.progression || undefined,
    confidence: toNumber(row.confidence),
    persistence: toNumber(row.persistence),
    structural_drift_score: toNumber(row.structural_drift_score || row.drift || row.drift_score),
    composite_instability: toNumber(row.composite_instability || row.instability || row.instability_score),
    drift_alert: toBool(row.drift_alert) ?? undefined,
    is_alert: toBool(row.is_alert) ?? undefined,
    trend: row.trend,
    phase: row.phase || row.phase_name || row.regime_name,
    risk_level: (row.risk_level || row.severity || 'UNKNOWN') as RunnerSeverity | string,
    estimated_rul: estimatedRul,
    failure_cycle: failureCycle,
    action_window_cycles: toNumber(row.action_window_cycles || row.actionWindowCycles),
    operator_message: row.operator_message || row.message,
    early_signal_detected: toBool(row.early_signal_detected),
    early_signal_confirmed: toBool(row.early_signal_confirmed),
    early_signal_confidence: toNumber(row.early_signal_confidence),
    divergence_alert: toBool(row.divergence_alert),
    dominant_driver: row.dominant_driver,
    recommended_urgency: row.recommended_urgency,
    severity: row.severity,
    recommended_action: row.recommended_action,
  }
}

async function readJsonl(filePath: string): Promise<RunnerCycleRecord[]> {
  const text = await fs.readFile(filePath, 'utf8')
  const out: RunnerCycleRecord[] = []
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    try {
      const obj = JSON.parse(trimmed) as Record<string, unknown>
      const unit = String(obj.unit_id ?? obj.asset_id ?? obj.unit ?? '')
      const cycle = Number(obj.cycle ?? obj.step ?? obj.frame)
      if (!unit || !Number.isFinite(cycle)) continue
      const estimatedRul =
        typeof obj.estimated_rul === 'number'
          ? obj.estimated_rul
          : typeof obj.rul === 'number'
            ? obj.rul
            : undefined
      const failureCycle =
        typeof obj.failure_cycle === 'number' ? obj.failure_cycle : typeof estimatedRul === 'number' ? cycle + estimatedRul : undefined
      out.push({
        unit_id: unit,
        cycle,
        timestamp: typeof obj.timestamp === 'string' ? obj.timestamp : undefined,
        state: typeof (obj.state ?? obj.operator_state) === 'string' ? String(obj.state ?? obj.operator_state) : undefined,
        zone:
          typeof (obj.zone ?? obj.dominant_zone ?? obj.dominant_component) === 'string'
            ? String(obj.zone ?? obj.dominant_zone ?? obj.dominant_component)
            : undefined,
        progression: typeof obj.progression === 'string' ? obj.progression : undefined,
        confidence: typeof obj.confidence === 'number' ? obj.confidence : undefined,
        persistence: typeof obj.persistence === 'number' ? obj.persistence : undefined,
        structural_drift_score: typeof obj.structural_drift_score === 'number' ? obj.structural_drift_score : undefined,
        composite_instability: typeof obj.composite_instability === 'number' ? obj.composite_instability : undefined,
        drift_alert: typeof obj.drift_alert === 'boolean' ? obj.drift_alert : undefined,
        is_alert: typeof obj.is_alert === 'boolean' ? obj.is_alert : undefined,
        trend: typeof obj.trend === 'string' ? obj.trend : undefined,
        phase: typeof obj.phase === 'string' ? obj.phase : undefined,
        risk_level: typeof (obj.risk_level ?? obj.severity) === 'string' ? String(obj.risk_level ?? obj.severity) : 'UNKNOWN',
        estimated_rul: estimatedRul,
        failure_cycle: failureCycle,
        action_window_cycles: typeof obj.action_window_cycles === 'number' ? obj.action_window_cycles : undefined,
        operator_message: typeof (obj.operator_message ?? obj.message) === 'string' ? String(obj.operator_message ?? obj.message) : undefined,
        early_signal_detected: typeof obj.early_signal_detected === 'boolean' ? obj.early_signal_detected : undefined,
        early_signal_confirmed: typeof obj.early_signal_confirmed === 'boolean' ? obj.early_signal_confirmed : undefined,
        early_signal_confidence: typeof obj.early_signal_confidence === 'number' ? obj.early_signal_confidence : undefined,
        divergence_alert: typeof obj.divergence_alert === 'boolean' ? obj.divergence_alert : undefined,
        dominant_driver: typeof obj.dominant_driver === 'string' ? obj.dominant_driver : undefined,
        recommended_urgency: typeof obj.recommended_urgency === 'string' ? obj.recommended_urgency : undefined,
        severity: typeof obj.severity === 'string' ? obj.severity : undefined,
        recommended_action: typeof obj.recommended_action === 'string' ? obj.recommended_action : undefined,
      })
    } catch {
      // ignore invalid line
    }
  }
  return out
}

async function tryReadFile(filePath: string): Promise<string | null> {
  try {
    return await fs.readFile(filePath, 'utf8')
  } catch {
    return null
  }
}

function getCandidatePaths(frontendCwd: string): string[] {
  const repoRoot = path.resolve(frontendCwd, '..')
  return [
    // Preferred: cycle-level “timeseries” style output already present in this repo.
    path.join(repoRoot, 'outputs', 'fd004_validation_cycles_operator.jsonl'),
    path.join(repoRoot, 'outputs', 'fd004_validation_cycles_operator.csv'),
    path.join(repoRoot, 'outputs', 'fd004_validation_cycles.jsonl'),
    path.join(repoRoot, 'outputs', 'fd004_validation_cycles.csv'),
    path.join(repoRoot, 'fd004_outputs_subset', 'fd004_real_timeseries_operator.csv'),
    path.join(repoRoot, 'fd004_outputs_subset', 'fd004_real_timeseries.csv'),
    path.join(repoRoot, 'fd004_outputs_subset', 'hero_unit_timeseries.csv'),
  ]
}

export async function loadRunnerCyclesFromDisk(): Promise<RunnerCycleRecord[]> {
  const frontendCwd = process.cwd()
  const candidates = getCandidatePaths(frontendCwd)

  for (const filePath of candidates) {
    if (filePath.endsWith('.jsonl')) {
      try {
        const recs = await readJsonl(filePath)
        if (recs.length > 0) return recs
      } catch {
        // keep trying
      }
      continue
    }

    const text = await tryReadFile(filePath)
    if (!text) continue
    const rows = parseCsv(text)
    const records: RunnerCycleRecord[] = []
    for (const row of rows) {
      const rec = rowToCycleRecord(row)
      if (rec) records.push(rec)
    }
    if (records.length > 0) return records
  }

  throw new Error('No runner cycle outputs found')
}
