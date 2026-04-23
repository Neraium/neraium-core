/**
 * Runner output loader.
 *
 * Maps runtime runner state (behavioral state, dominant driver)
 * to operator-facing event panel values.
 *
 * Event panel rules:
 * - Issue: 2–5 words, no truncation, no "..."
 * - Progression: Increasing | Spreading | Stable
 * - Issue derived from dominant behavioral signal — not hardcoded per severity
 *
 * Does NOT import from simulation.ts to avoid circular dependency.
 * Accepts the same duck-typed RoomAdapterInput from growScenarioAdapter.
 */

import { ZoneOperatorStatus, ZoneRole, RoomAdapterInput } from './growScenarioAdapter'
import { primarySubsystemFromDriver } from './growMapping'

// ─── Issue label mapping ──────────────────────────────────────────────────────

// Behavioral state → 2–5 word operator issue label
const BEHAVIORAL_ISSUE_MAP: Record<string, string> = {
  'Thermal runaway':       'Thermal runaway',
  'Desiccation risk':      'Humidity deficit',
  'VPD critical':          'VPD critically elevated',
  'CO₂ depletion':         'CO₂ supply depleted',
  'Heat accumulation':     'Heat accumulation',
  'Drying trend':          'Humidity trending low',
  'VPD elevated':          'VPD out of range',
  'Recovery lag':          'Climate recovery lag',
  'Breakdown forming':     'Environmental breakdown',
  'Post-irrigation':       'Airflow imbalance',
  'Active photosynthesis': 'Climate instability',
  'Steady state':          'Environmental drift',
}

// Subsystem fallback when behavioral state is not in the map
const SUBSYSTEM_ISSUE_MAP: Record<string, string> = {
  climate:      'Climate instability',
  humidity:     'Humidity imbalance',
  airflow:      'Airflow imbalance',
  lighting:     'Lighting load drift',
  plant_stress: 'Plant stress forming',
}

export function issueFromRunnerState(room: RoomAdapterInput): string {
  const direct = BEHAVIORAL_ISSUE_MAP[room.behavioralState]
  if (direct) return direct

  const subsystem = primarySubsystemFromDriver(room.behavioralState)
  return SUBSYSTEM_ISSUE_MAP[subsystem] ?? 'Environmental drift'
}

// ─── Progression mapping ──────────────────────────────────────────────────────

export function progressionFromTransition(
  role: ZoneRole,
  prevStatus: ZoneOperatorStatus,
  currentStatus: ZoneOperatorStatus,
): 'Increasing' | 'Spreading' | 'Stable' {
  // Spread zone becoming visible for the first time → "Spreading"
  if (role === 'spread' && currentStatus !== 'NORMAL' && prevStatus === 'NORMAL') {
    return 'Spreading'
  }

  // Status worsened → "Increasing"
  if (prevStatus === 'NORMAL' && currentStatus === 'WATCH')     return 'Increasing'
  if (prevStatus === 'WATCH'  && currentStatus === 'INTERVENE') return 'Increasing'

  // Stayed at same elevated level: not resolved, not worsening
  return 'Stable'
}
