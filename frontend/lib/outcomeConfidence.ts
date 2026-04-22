/**
 * Outcome Confidence System
 *
 * Projects likely system response to:
 * 1. Following the recommended action
 * 2. Not acting (no-action consequence)
 *
 * Language scaled by confidence level.
 * Everything probabilistic, never guaranteed.
 */

export interface OutcomeConfidence {
  actionOutcome: string // What happens if they act
  noActionConsequence: string | null // What happens if they don't
  temporalHint: string | null // "within ~10 cycles", "short-term effect", etc.
}

/**
 * Project expected system response to recommended action.
 * Scaled by confidence and current system state.
 */
export function projectActionOutcome(
  failureMode: string,
  drift: number,
  stability: number,
  coherence: number,
  confidence: 'high' | 'moderate' | 'low'
): string {
  // Base outcomes by failure mode
  const outcomeMap: Record<string, Record<'high' | 'moderate' | 'low', string>> = {
    'drift-driven': {
      high: 'Stabilization likely',
      moderate: 'Stabilization possible',
      low: 'Outcome uncertain',
    },
    'stability-critical': {
      high: 'Coherence recovery expected',
      moderate: 'Partial recovery possible',
      low: 'Limited improvement uncertain',
    },
    'coherence-loss': {
      high: 'Coupling stabilization likely',
      moderate: 'Partial coupling recovery possible',
      low: 'Recovery outcome uncertain',
    },
    'drift-dominant': {
      high: 'Drift reduction expected',
      moderate: 'Drift reduction possible',
      low: 'Drift response uncertain',
    },
    'multi-mode': {
      high: 'Multi-point stabilization likely',
      moderate: 'Partial stabilization possible',
      low: 'Outcome uncertain',
    },
  }

  return outcomeMap[failureMode]?.[confidence] || outcomeMap['multi-mode'][confidence]
}

/**
 * Project consequence of taking no action.
 * Only appears during instability+.
 */
export function projectNoActionConsequence(
  phase: string,
  drift: number,
  stability: number,
  coherence: number,
  confidence: 'high' | 'moderate' | 'low'
): string | null {
  // Only show consequence during instability+
  if (phase === 'Stable' || phase === 'Drift forming') {
    return null
  }

  // Infer consequence from current state
  if (drift > 0.75 && coherence < 0.4) {
    return confidence === 'high' ? 'Cascade progression inevitable' : 'Cascade progression likely'
  }
  if (stability < 0.3) {
    return confidence === 'high' ? 'System degradation certain' : 'System degradation likely'
  }
  if (coherence < 0.3) {
    return confidence === 'high' ? 'Coherence collapse imminent' : 'Coherence collapse likely'
  }
  if (stability < 0.5 && drift > 0.6) {
    return confidence === 'high' ? 'Recovery window closing' : 'Recovery difficulty increasing'
  }

  return null
}

/**
 * Optional temporal hint for outcome.
 * Only if confidence is high and situation is clear.
 */
export function getTemporalHint(
  confidence: 'high' | 'moderate' | 'low',
  phase: string,
  drift: number
): string | null {
  if (confidence !== 'high') {
    return null
  }

  // High clarity + critical = faster response needed
  if (phase === 'Critical' && drift > 0.7) {
    return 'within ~8 cycles'
  }

  // High clarity + instability = moderate timeframe
  if (phase === 'Instability forming') {
    return 'within ~12 cycles'
  }

  return null
}

/**
 * Compute complete outcome confidence state.
 */
export function computeOutcomeConfidence(
  failureMode: string,
  phase: string,
  drift: number,
  stability: number,
  coherence: number,
  confidence: 'high' | 'moderate' | 'low'
): OutcomeConfidence {
  const actionOutcome = projectActionOutcome(failureMode, drift, stability, coherence, confidence)
  const noActionConsequence = projectNoActionConsequence(phase, drift, stability, coherence, confidence)
  const temporalHint = getTemporalHint(confidence, phase, drift)

  return {
    actionOutcome,
    noActionConsequence,
    temporalHint,
  }
}
