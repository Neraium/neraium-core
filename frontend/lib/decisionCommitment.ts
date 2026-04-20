/**
 * Decision Commitment System
 *
 * Makes recommendation confidence perceptible through visual strengthening
 * as the system converges on a decision path.
 *
 * Momentum: emerging → establishing → strengthening
 * Commitment: 0-1 scalar driving visual weight
 */

export type DecisionMomentum = 'emerging' | 'establishing' | 'strengthening'

export interface DecisionCommitment {
  momentum: DecisionMomentum
  commitmentScore: number // 0-1
  hasChanged: boolean // Did recommendation change this tick?
  stepsAtCurrent: number // How many ticks at current recommendation
}

/**
 * Calculate commitment score based on metric confidence and consistency.
 * Higher when metrics strongly support the recommendation.
 */
function calculateCommitmentScore(
  drift: number,
  stability: number,
  coherence: number,
  confidence: 'high' | 'moderate' | 'low'
): number {
  // Base on diagnostic confidence
  const confidenceScore = confidence === 'high' ? 0.8 : confidence === 'moderate' ? 0.5 : 0.2

  // Boost if metrics are unambiguous (far from neutral 0.5)
  const driftClarity = Math.abs(drift - 0.5)
  const stabilityClarity = Math.abs(stability - 0.5)
  const coherenceClarity = Math.abs(coherence - 0.5)
  const metricClarity = (driftClarity + stabilityClarity + coherenceClarity) / 3

  // Commitment = confidence weighted by metric clarity
  return Math.min(1, confidenceScore + metricClarity * 0.3)
}

/**
 * Map commitment score to momentum stage.
 * Uses hysteresis (20% threshold) to prevent flickering.
 */
function calculateMomentum(
  currentMomentum: DecisionMomentum,
  commitmentScore: number
): DecisionMomentum {
  // Thresholds with hysteresis
  const EMERGING_THRESHOLD = 0.35
  const ESTABLISHING_THRESHOLD = 0.65
  const HYSTERESIS = 0.08

  // Determine new momentum based on commitment score
  if (currentMomentum === 'strengthening') {
    // Require drop below threshold - hysteresis to prevent flicker
    if (commitmentScore < ESTABLISHING_THRESHOLD - HYSTERESIS) {
      return 'establishing'
    }
    return 'strengthening'
  }

  if (currentMomentum === 'establishing') {
    if (commitmentScore >= ESTABLISHING_THRESHOLD + HYSTERESIS) {
      return 'strengthening'
    }
    if (commitmentScore < EMERGING_THRESHOLD - HYSTERESIS) {
      return 'emerging'
    }
    return 'establishing'
  }

  // currentMomentum === 'emerging'
  if (commitmentScore >= EMERGING_THRESHOLD + HYSTERESIS) {
    return 'establishing'
  }
  return 'emerging'
}

/**
 * Track decision commitment over time.
 * Requires previous state to detect changes and maintain momentum.
 */
export function computeDecisionCommitment(
  drift: number,
  stability: number,
  coherence: number,
  confidence: 'high' | 'moderate' | 'low',
  previousCommitment: DecisionCommitment | null
): DecisionCommitment {
  const commitmentScore = calculateCommitmentScore(drift, stability, coherence, confidence)

  // Determine if recommendation changed
  const hasChanged = previousCommitment === null
  const stepsAtCurrent = hasChanged ? 1 : (previousCommitment?.stepsAtCurrent ?? 0) + 1

  // Calculate momentum, preserving previous if just starting
  const previousMomentum = previousCommitment?.momentum ?? 'emerging'
  const momentum = calculateMomentum(previousMomentum, commitmentScore)

  return {
    momentum,
    commitmentScore,
    hasChanged,
    stepsAtCurrent,
  }
}

/**
 * Map momentum to visual opacity multiplier.
 * emerging: 0.6, establishing: 0.85, strengthening: 1.0
 */
export function getMomentumOpacity(momentum: DecisionMomentum): number {
  const opacities: Record<DecisionMomentum, number> = {
    emerging: 0.6,
    establishing: 0.85,
    strengthening: 1.0,
  }
  return opacities[momentum]
}

/**
 * Map commitment score (0-1) to glow intensity multiplier.
 * Lower at emerging, higher at strengthening.
 */
export function getCommitmentGlowIntensity(commitmentScore: number): number {
  // Map 0-1 to glow range: 0.4 (subtle) to 1.2 (intense)
  return 0.4 + commitmentScore * 0.8
}

/**
 * Determine if recommendation should feel "locked in".
 * True when in strengthening phase with sufficient score.
 */
export function isRecommendationLocked(
  momentum: DecisionMomentum,
  commitmentScore: number
): boolean {
  return momentum === 'strengthening' && commitmentScore > 0.7
}
