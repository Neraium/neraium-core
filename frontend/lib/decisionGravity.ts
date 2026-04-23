/**
 * Decision Gravity System
 *
 * Introduces threshold moments, time-to-impact, and consequence feeling.
 * Transforms interface from passive observation to active decision-making.
 */

export interface ConsequenceState {
  timeToImpact: number | null // milliseconds remaining before critical
  timeToImpactLabel: string // "~12 cycles", "Imminent", "Window closing"
  hasThresholdCrossed: boolean // Just transitioned into instability+
  thresholdIntensity: number // 0-1, for snap effects
  operatorFocus: string | null // Suggested action
  isApproachingFailure: boolean // System nearing collapse
  escalationLevel: number // 0=stable, 1=drift, 2=instability, 3=critical, 4=failure
}

/**
 * Calculate time-to-impact based on current drift and stability trends.
 * Returns approximate time until system reaches critical state.
 */
export function calculateTimeToImpact(
  drift: number,
  stability: number,
  phase: string,
  phaseProgress: number
): number | null {
  // Only show time-to-impact during instability or worse
  if (phase === 'Stable' || phase === 'Drift forming') {
    return null
  }

  // Estimate time remaining based on current trajectory
  // Drift accelerating, stability declining
  const driftVelocity = drift * 0.5 // How fast drift is building
  const stabilityVelocity = (1 - stability) * 0.3 // How fast stability is declining

  // Combined "speed toward critical"
  const convergenceSpeed = (driftVelocity + stabilityVelocity) / 2
  if (convergenceSpeed < 0.01) return null

  // Time until reaching critical threshold (drift > 0.8, stability < 0.2)
  const driftTimeRemaining = Math.max(0, (0.8 - drift) / (driftVelocity + 0.01))
  const stabilityTimeRemaining = Math.max(0, (stability - 0.2) / (stabilityVelocity + 0.01))

  // Take minimum (whichever hits threshold first)
  const timeMs = Math.min(driftTimeRemaining, stabilityTimeRemaining) * 8000 // Scale to 8s base unit

  return Math.max(1000, timeMs) // Min 1 second, max reasonable value
}

/**
 * Format time-to-impact into human-readable labels.
 */
export function formatTimeToImpact(timeMs: number | null): string {
  if (timeMs === null) return ''
  if (timeMs < 3000) return 'Imminent'
  if (timeMs < 8000) return 'Window closing'
  if (timeMs < 15000) return '~12 cycles'
  if (timeMs < 25000) return '~20 cycles'
  return '~30 cycles'
}

/**
 * Get escalation language for narrative based on phase.
 * Escalates from descriptive to directive as system degrades.
 */
export function getEscalationNarrative(
  phase: string,
  phaseProgress: number
): string {
  const narratives: Record<string, string[]> = {
    Stable: ['System coherent', 'Drift nominal', 'Coupled systems aligned'],
    'Drift forming': ['Drift emerging', 'Coupling stressed', 'Recovery window open'],
    'Instability forming': [
      'Intervention advisable',
      'Structural stress detected',
      'Recovery pathway closing',
    ],
    Critical: ['Intervention required', 'Failure trajectory active', 'Immediate action needed'],
  }

  const options = narratives[phase] || narratives.Stable
  // Use phase progress to cycle through options for variety
  const index = Math.floor(phaseProgress * options.length) % options.length
  return options[index]
}

/**
 * Get operator focus (suggested action) based on system state.
 * Only shown during instability+.
 * NOTE: This is now aligned with the primary action from evaluateActionDecision
 * to prevent contradictory guidance.
 */
export function getOperatorFocus(
  phase: string,
  drift: number,
  stability: number,
  coherence: number,
  primaryActionLabel?: string | null
): string | null {
  if (phase === 'Stable' || phase === 'Drift forming') {
    return null
  }

  // If a primary action is provided from the decision engine, use it directly
  // to ensure coherence across the UI
  if (primaryActionLabel) {
    return primaryActionLabel
  }

  // Fallback: derive from state
  if (drift > 0.6 && coherence < 0.5) {
    return 'Inspect coupling nodes'
  }
  if (drift > 0.7) {
    return 'Check airflow distribution'
  }
  if (stability < 0.4) {
    return 'Reduce system load'
  }
  if (coherence < 0.3) {
    return 'Isolate failing subsystems'
  }

  return 'Monitor system closely'
}

/**
 * Detect if system just crossed a critical threshold.
 */
export function detectThresholdCross(
  previousPhase: string,
  currentPhase: string
): boolean {
  // Crossed into instability or worse
  const criticalPhases = ['Instability forming', 'Critical']
  const wasBefore = previousPhase === 'Stable' || previousPhase === 'Drift forming'
  const isNow = criticalPhases.includes(currentPhase)
  return wasBefore && isNow
}

/**
 * Get escalation level (0-4) for intensity-based effects.
 */
export function getEscalationLevel(phase: string): number {
  const levels: Record<string, number> = {
    Stable: 0,
    'Drift forming': 1,
    'Instability forming': 2,
    Critical: 3,
  }
  return levels[phase] ?? 0
}

/**
 * Compute consequence state based on all system inputs.
 */
export function computeConsequenceState(
  phase: string,
  phaseProgress: number,
  drift: number,
  stability: number,
  coherence: number,
  previousPhase: string,
  primaryActionLabel?: string | null
): ConsequenceState {
  const timeToImpact = calculateTimeToImpact(drift, stability, phase, phaseProgress)
  const thresholdCrossed = detectThresholdCross(previousPhase, phase)
  const escalationLevel = getEscalationLevel(phase)
  const isApproachingFailure = escalationLevel >= 3 && timeToImpact !== null && timeToImpact < 5000

  return {
    timeToImpact,
    timeToImpactLabel: formatTimeToImpact(timeToImpact),
    hasThresholdCrossed: thresholdCrossed,
    thresholdIntensity: thresholdCrossed ? Math.min(1, phaseProgress * 4) : 0,
    operatorFocus: getOperatorFocus(phase, drift, stability, coherence, primaryActionLabel),
    isApproachingFailure,
    escalationLevel,
  }
}
