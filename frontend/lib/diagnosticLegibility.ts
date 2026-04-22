/**
 * Diagnostic Legibility System
 *
 * Provides operator-oriented diagnosis:
 * - What started the problem (origin)
 * - How it spread (propagation path)
 * - What matters most now (current risk)
 * - Why action is required now (change in system)
 * - How certain we are (confidence)
 */

export interface DiagnosticLegibility {
  origin: string | null // "Airflow imbalance", "Humidity spike", etc.
  originSubsystem: string | null // which subsystem started it
  propagationPath: string | null // "airflow -> coupling -> coherence"
  currentRiskZone: string | null // "Climate response", "Structural coupling", etc.
  whyNow: string | null // "Coupling decay accelerating", "Recovery lag widening"
  confidence: 'high' | 'moderate' | 'low'
  dominantDriver: string | null // Primary failure mode
}

/**
 * Diagnose the origin of system degradation.
 * Which subsystem started the problem?
 */
export function diagnoseOrigin(
  drift: number,
  stability: number,
  coherence: number,
  driftHistory: number[],
  stabilityHistory: number[]
): { subsystem: string; trigger: string } | null {
  // No problem if system is stable
  if (drift < 0.3 && stability > 0.7) {
    return null
  }

  // Determine what initiated the problem
  // Check if drift led (structural imbalance)
  const driftTrend = driftHistory.length > 1 ? driftHistory[driftHistory.length - 1] - driftHistory[0] : 0
  // Check if stability led (coherence loss)
  const stabilityTrend = stabilityHistory.length > 1 ? stabilityHistory[0] - stabilityHistory[stabilityHistory.length - 1] : 0

  if (driftTrend > stabilityTrend * 0.5) {
    // Drift was primary driver
    if (drift > 0.6) {
      return { subsystem: 'airflow', trigger: 'Airflow imbalance initiating' }
    }
    return { subsystem: 'structural', trigger: 'Structural drift detected' }
  } else {
    // Stability/coherence was primary
    if (coherence < 0.5) {
      return { subsystem: 'coherence', trigger: 'Coherence loss detected' }
    }
    return { subsystem: 'stability', trigger: 'Stability degradation detected' }
  }
}

/**
 * Derive propagation path: how the problem spread.
 */
export function derivePropagationPath(
  origin: string | null,
  drift: number,
  stability: number,
  coherence: number
): string | null {
  if (!origin) return null

  const segments: string[] = []

  // Start with origin
  segments.push(origin)

  // Add intermediate propagation based on current state
  if (drift > 0.5 && stability < 0.6) {
    if (!segments.includes('structural coupling')) {
      segments.push('structural coupling')
    }
  }

  if (coherence < 0.6 && !segments.includes('coherence')) {
    segments.push('coherence')
  }

  if (stability < 0.4 && !segments.includes('system stability')) {
    segments.push('system stability')
  }

  // Format as path
  if (segments.length === 1) return null // Just origin, no propagation yet
  return segments.join(' → ')
}

/**
 * Identify current dominant risk zone.
 */
export function identifyCurrentRiskZone(
  drift: number,
  stability: number,
  coherence: number
): string {
  // Determine what's currently the biggest threat
  if (drift > 0.75 && coherence < 0.4) {
    return 'Coupling cascade'
  }
  if (stability < 0.3) {
    return 'Structural instability'
  }
  if (coherence < 0.35) {
    return 'System coherence'
  }
  if (drift > 0.6) {
    return 'Drift amplification'
  }
  if (stability < 0.5) {
    return 'Recovery degradation'
  }
  return 'Multi-subsystem stress'
}

/**
 * Generate "why now" explanation.
 * What changed that makes action urgent now?
 */
export function generateWhyNow(
  phase: string,
  drift: number,
  stability: number,
  coherence: number,
  driftHistory: number[],
  stabilityHistory: number[]
): string | null {
  // Only explain "why now" when entering critical decisions
  if (phase === 'Stable' || phase === 'Drift forming') {
    return null
  }

  // Calculate recent trends
  const recentDriftAccel = driftHistory.length > 3
    ? (driftHistory[driftHistory.length - 1] - driftHistory[driftHistory.length - 3]) / 2
    : 0

  const recentStabilityDecay = stabilityHistory.length > 3
    ? (stabilityHistory[stabilityHistory.length - 3] - stabilityHistory[stabilityHistory.length - 1]) / 2
    : 0

  // Generate reason based on what's accelerating
  if (recentDriftAccel > 0.1) {
    return 'Drift acceleration widening'
  }
  if (recentStabilityDecay > 0.1) {
    return 'Recovery lag extending'
  }
  if (drift > 0.7 && coherence < 0.4) {
    return 'Coupling cascade imminent'
  }
  if (stability < 0.3) {
    return 'Coherence threshold breached'
  }

  return null
}

/**
 * Calculate confidence in diagnosis.
 */
export function calculateDiagnosticConfidence(
  drift: number,
  stability: number,
  coherence: number,
  driftHistory: number[],
  stabilityHistory: number[]
): 'high' | 'moderate' | 'low' {
  // Confidence increases with:
  // - Strong trend consistency
  // - Clear subsystem correlation
  // - High margin from neutral

  const driftClarity = Math.abs(drift - 0.5) // Distance from neutral
  const stabilityClarity = Math.abs(stability - 0.5)
  const coherenceClarity = Math.abs(coherence - 0.5)

  const avgClarity = (driftClarity + stabilityClarity + coherenceClarity) / 3

  // Trend consistency
  let trendConsistency = 0
  if (driftHistory.length > 2) {
    const driftDir = Math.sign(driftHistory[driftHistory.length - 1] - driftHistory[driftHistory.length - 2])
    const prevDriftDir = Math.sign(driftHistory[driftHistory.length - 2] - driftHistory[driftHistory.length - 3])
    if (driftDir === prevDriftDir) trendConsistency += 0.5
  }
  if (stabilityHistory.length > 2) {
    const stabDir = Math.sign(stabilityHistory[stabilityHistory.length - 1] - stabilityHistory[stabilityHistory.length - 2])
    const prevStabDir = Math.sign(stabilityHistory[stabilityHistory.length - 2] - stabilityHistory[stabilityHistory.length - 3])
    if (stabDir === prevStabDir) trendConsistency += 0.5
  }

  const confidence = avgClarity * 0.6 + trendConsistency * 0.4

  if (confidence > 0.65) return 'high'
  if (confidence > 0.45) return 'moderate'
  return 'low'
}

/**
 * Determine primary failure mode for specific actions.
 */
export function determinePrimaryFailureMode(
  drift: number,
  stability: number,
  coherence: number
): string {
  if (drift > 0.75) return 'drift-driven'
  if (stability < 0.3) return 'stability-critical'
  if (coherence < 0.35) return 'coherence-loss'
  if (drift > stability) return 'drift-dominant'
  return 'multi-mode'
}

/**
 * Get operationally-grounded action based on failure mode.
 * Not generic guidance—specific to what's actually failing.
 */
export function getFailureModeAction(failureMode: string): string {
  const actions: Record<string, string> = {
    'drift-driven': 'Stabilize airflow distribution',
    'stability-critical': 'Reduce coupling load immediately',
    'coherence-loss': 'Isolate stressed subsystems',
    'drift-dominant': 'Check structural damping',
    'multi-mode': 'Reduce system load across all modes',
  }
  return actions[failureMode] || 'Reduce system stress'
}

/**
 * Compute full diagnostic legibility state.
 */
export function computeDiagnosticLegibility(
  phase: string,
  drift: number,
  stability: number,
  coherence: number,
  driftHistory: number[],
  stabilityHistory: number[]
): DiagnosticLegibility {
  const origin = diagnoseOrigin(drift, stability, coherence, driftHistory, stabilityHistory)
  const propagationPath = derivePropagationPath(origin?.subsystem ?? null, drift, stability, coherence)
  const currentRiskZone = identifyCurrentRiskZone(drift, stability, coherence)
  const whyNow = generateWhyNow(phase, drift, stability, coherence, driftHistory, stabilityHistory)
  const confidence = calculateDiagnosticConfidence(drift, stability, coherence, driftHistory, stabilityHistory)
  const failureMode = determinePrimaryFailureMode(drift, stability, coherence)
  const action = getFailureModeAction(failureMode)

  return {
    origin: origin?.trigger ?? null,
    originSubsystem: origin?.subsystem ?? null,
    propagationPath,
    currentRiskZone,
    whyNow,
    confidence,
    dominantDriver: failureMode,
  }
}
