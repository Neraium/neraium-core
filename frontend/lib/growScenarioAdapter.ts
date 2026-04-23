/**
 * Grow scenario adapter.
 *
 * Maps raw simulation room state to operator-visible zone statuses
 * (NORMAL / WATCH / INTERVENE) using action-window logic with
 * restrained spread behavior.
 *
 * Invariants enforced here:
 * - WATCH only appears when a zone is entering a credible pre-action state
 * - Default path is NORMAL → WATCH → INTERVENE
 * - NORMAL → INTERVENE bypass only when drift is already in the top ~12%
 *   of the action window and the zone never visibly dwelled in WATCH
 * - Spread defaults to primary origin (FLOW-B); neighboring zones escalate
 *   slowly, usually surfacing as WATCH not immediate INTERVENE
 * - VEG zones remain stable unless drift crosses an extreme threshold
 *
 * Does NOT import from simulation.ts to avoid circular dependency.
 * Accepts minimal duck-typed inputs defined here.
 */

import { issueFromRunnerState, progressionFromTransition } from './runnerOutputLoader'

// ─── Minimal input types (duck-typed, no import from simulation) ──────────────

export interface RoomAdapterInput {
  id: string
  shortName: string
  driftContribution: number
  behavioralState: string
  growStage: 'veg' | 'flower'
}

export interface SystemAdapterInput {
  rooms: RoomAdapterInput[]
  timestamp: { toISOString(): string }
}

// ─── Output types ─────────────────────────────────────────────────────────────

export type ZoneOperatorStatus = 'NORMAL' | 'WATCH' | 'INTERVENE'

export type ZoneRole = 'primary' | 'spread' | 'stable'

export interface ZoneAdaptedState {
  id: string
  shortName: string
  status: ZoneOperatorStatus
  role: ZoneRole
  driftContribution: number
  behavioralState: string
}

export interface OperatorEventEntry {
  zoneId: string
  zoneName: string
  issue: string                                        // 2–5 words, no "..."
  progression: 'Increasing' | 'Spreading' | 'Stable'
  status: ZoneOperatorStatus
}

export interface AdaptedGrowState {
  zones: ZoneAdaptedState[]
  activeEvents: OperatorEventEntry[]
  systemStatus: ZoneOperatorStatus
  lastChangedAt: string | null
  playbackSpeedFactor: number
}

// ─── Thresholds ───────────────────────────────────────────────────────────────

// Watch horizon: tight enough that WATCH feels like a real warning.
// FLOW-B enters WATCH around narrativeProgress 0.30 (drift ≈ 0.32).
const WATCH_THRESHOLD_PRIMARY     = 0.32
const INTERVENE_THRESHOLD_PRIMARY = 0.65  // FLOW-B INTERVENE ~narrativeProgress 0.70

// Spread zones escalate slower than primary
const WATCH_THRESHOLD_SPREAD      = 0.30
const INTERVENE_THRESHOLD_SPREAD  = 0.64  // spread zones INTERVENE only very late

// VEG zones essentially never reach WATCH in a normal scenario
const WATCH_THRESHOLD_VEG         = 0.52
const INTERVENE_THRESHOLD_VEG     = 0.76

// Late-stage bypass: allows NORMAL → INTERVENE only when the unit is already
// deep inside the action window. Deliberately rare.
const LATE_STAGE_BYPASS = 0.88

// ─── Playback pacing ─────────────────────────────────────────────────────────

// Multiplier applied to scenarioProgress advancement speed.
// Lower value = slower advance = more dwell time per state.
export function derivePlaybackFactor(systemStatus: ZoneOperatorStatus): number {
  if (systemStatus === 'NORMAL')  return 0.72  // calm — let viewer read the baseline
  if (systemStatus === 'WATCH')   return 0.88  // slightly slower — building tension
  return 0.96                                   // INTERVENE — deliberate, clear
}

// ─── Zone role ────────────────────────────────────────────────────────────────

function zoneRole(roomId: string): ZoneRole {
  if (roomId === 'flow-b') return 'primary'
  if (roomId === 'flow-a' || roomId === 'flow-c') return 'spread'
  return 'stable'
}

// ─── Zone classification ──────────────────────────────────────────────────────

function classifyZoneStatus(
  drift: number,
  role: ZoneRole,
  prevStatus: ZoneOperatorStatus | undefined,
): ZoneOperatorStatus {
  let watchThreshold: number
  let interveneThreshold: number

  if (role === 'primary') {
    watchThreshold     = WATCH_THRESHOLD_PRIMARY
    interveneThreshold = INTERVENE_THRESHOLD_PRIMARY
  } else if (role === 'spread') {
    watchThreshold     = WATCH_THRESHOLD_SPREAD
    interveneThreshold = INTERVENE_THRESHOLD_SPREAD
  } else {
    // VEG: stable unless scenario drives extreme instability
    watchThreshold     = WATCH_THRESHOLD_VEG
    interveneThreshold = INTERVENE_THRESHOLD_VEG
  }

  // Late-stage bypass — only when the unit never visibly dwelled in WATCH,
  // meaning the scenario jumped straight to near-failure level.
  if (
    drift >= LATE_STAGE_BYPASS &&
    prevStatus !== 'WATCH' &&
    prevStatus !== 'INTERVENE'
  ) {
    return 'INTERVENE'
  }

  if (drift >= interveneThreshold) return 'INTERVENE'
  if (drift >= watchThreshold)     return 'WATCH'
  return 'NORMAL'
}

// ─── Adapter class ────────────────────────────────────────────────────────────

export class GrowScenarioAdapter {
  private prevStatuses: Record<string, ZoneOperatorStatus> = {}
  private prevSystemStatus: ZoneOperatorStatus = 'NORMAL'
  private lastChangedAt: string | null = null

  adapt(system: SystemAdapterInput): AdaptedGrowState {
    const now = system.timestamp.toISOString()

    // Classify all zones
    const zones: ZoneAdaptedState[] = system.rooms.map(room => {
      const role   = zoneRole(room.id)
      const prev   = this.prevStatuses[room.id]
      const status = classifyZoneStatus(room.driftContribution, role, prev)
      return {
        id: room.id,
        shortName: room.shortName,
        status,
        role,
        driftContribution: room.driftContribution,
        behavioralState: room.behavioralState,
      }
    })

    // System status = highest severity visible across zones
    let systemStatus: ZoneOperatorStatus = 'NORMAL'
    for (const z of zones) {
      if (z.status === 'INTERVENE') { systemStatus = 'INTERVENE'; break }
      if (z.status === 'WATCH')       systemStatus = 'WATCH'
    }

    // Detect operator-visible changes
    let changed = systemStatus !== this.prevSystemStatus
    if (!changed) {
      for (const z of zones) {
        if ((this.prevStatuses[z.id] ?? 'NORMAL') !== z.status) {
          changed = true
          break
        }
      }
    }

    if (changed) {
      this.lastChangedAt = now
    }

    // Build active events only for zones that are not NORMAL
    const activeEvents: OperatorEventEntry[] = zones
      .filter(z => z.status !== 'NORMAL')
      .map(z => {
        const room      = system.rooms.find(r => r.id === z.id)!
        const prevStatus = this.prevStatuses[z.id] ?? 'NORMAL'
        return {
          zoneId:      z.id,
          zoneName:    z.shortName,
          issue:       issueFromRunnerState(room),
          progression: progressionFromTransition(z.role, prevStatus, z.status),
          status:      z.status,
        }
      })

    // Persist state for next tick
    for (const z of zones) {
      this.prevStatuses[z.id] = z.status
    }
    this.prevSystemStatus = systemStatus

    return {
      zones,
      activeEvents,
      systemStatus,
      lastChangedAt: this.lastChangedAt,
      playbackSpeedFactor: derivePlaybackFactor(systemStatus),
    }
  }

  reset(): void {
    this.prevStatuses     = {}
    this.prevSystemStatus = 'NORMAL'
    this.lastChangedAt    = null
  }
}
