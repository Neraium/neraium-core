/**
 * Continuous demo data generator
 * Produces smooth, physically meaningful system evolution over 45-60 seconds
 * All curves are continuous functions of time, no discrete phases
 */

// Easing functions for smooth curves
const easeInOutSigmoid = (t: number): number => {
  // Smooth sigmoid from 0 to 1
  return t / (t + (1 - t)) * 0.5 + 0.5
}

const easeInCubic = (t: number): number => t * t * t
const easeOutCubic = (t: number): number => 1 - Math.pow(1 - t, 3)
const easeInOutCubic = (t: number): number =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2

// Smooth step functions
const smoothstep = (edge0: number, edge1: number, x: number): number => {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)))
  return t * t * (3.0 - 2.0 * t)
}

interface ContinuousFrameData {
  time: number
  timestamp: string
  coherence: number
  drift: number
  subsystems: {
    climate: { drift: number; confidence: number }
    airflow: { drift: number; confidence: number }
    irrigation: { drift: number; confidence: number }
    plant: { drift: number; confidence: number }
  }
  fragility: number
  confidence: number
}

/**
 * Generate continuous demo data over 45-60 seconds
 * Early (0-15s): stable with micro-asymmetry
 * Mid (15-35s): drift begins in airflow, spreads through coupling
 * Late (35-60s): strong deformation, acceleration
 */
export function generateContinuousDemoFrame(
  frameIndex: number,
  totalFrames: number
): ContinuousFrameData {
  // Total duration: 45 seconds at 30fps = 1350 frames, cap at 60s
  const maxDurationSeconds = 60
  const frameRate = 30
  const maxFrames = maxDurationSeconds * frameRate

  // Normalize time to 0-1 over the demo duration
  const normalizedTime = Math.min(frameIndex / maxFrames, 1.0)
  const timeSeconds = frameIndex / frameRate

  // ============================================
  // PHASE 1: Micro-asymmetry (0-15s, t: 0-0.25)
  // ============================================
  // Almost imperceptible changes, system appears stable
  const phase1End = 0.25
  const microAsymmetryMagnitude = easeInCubic(normalizedTime / phase1End) * 0.05

  // ============================================
  // PHASE 2: Airflow drift initiation (15-35s, t: 0.25-0.583)
  // ============================================
  // Drift begins in one subsystem and spreads through coupling
  const phase2Start = 0.25
  const phase2End = 0.583
  const phase2Progress =
    normalizedTime < phase2Start ? 0 : normalizedTime > phase2End ? 1 : (normalizedTime - phase2Start) / (phase2End - phase2Start)

  // Airflow is primary instigator
  const airflowDriftPrimary = phase2Progress * phase2Progress * 0.45 // Accelerating drift

  // Coupling: airflow drift spreads to other subsystems with delay
  const couplingDelay = 0.1 // 10% of phase2 duration
  const irrigationCoupling = Math.max(
    0,
    phase2Progress - couplingDelay
  ) ** 1.2 * 0.3
  const climateCoupling = Math.max(0, phase2Progress - couplingDelay * 0.5) ** 1.2 * 0.25
  const plantCoupling = Math.max(0, phase2Progress - couplingDelay * 1.5) ** 1.2 * 0.2

  // ============================================
  // PHASE 3: Strong deformation (35-60s, t: 0.583-1.0)
  // ============================================
  const phase3Start = 0.583
  const phase3Progress =
    normalizedTime < phase3Start ? 0 : (normalizedTime - phase3Start) / (1.0 - phase3Start)

  // All subsystems accelerate their drift
  const airflowDriftLate = airflowDriftPrimary + phase3Progress * phase3Progress * 0.3
  const irrigationDriftLate = irrigationCoupling + phase3Progress * 0.25
  const climateDriftLate = climateCoupling + phase3Progress * 0.2
  const plantDriftLate = plantCoupling + phase3Progress * 0.15

  // Add micro-asymmetry throughout
  const climateAsymmetry = microAsymmetryMagnitude * Math.sin(timeSeconds * 0.5)
  const airflowAsymmetry = microAsymmetryMagnitude * Math.cos(timeSeconds * 0.3)
  const irrigationAsymmetry = microAsymmetryMagnitude * Math.sin(timeSeconds * 0.7)
  const plantAsymmetry = microAsymmetryMagnitude * Math.cos(timeSeconds * 0.4)

  // ============================================
  // Combine subsystem drifts
  // ============================================
  const climateDrift = Math.max(0, climateDriftLate + climateAsymmetry)
  const airflowDrift = Math.max(0, airflowDriftLate + airflowAsymmetry)
  const irrigationDrift = Math.max(0, irrigationDriftLate + irrigationAsymmetry)
  const plantDrift = Math.max(0, plantDriftLate + plantAsymmetry)

  // Total drift is max of all subsystems (most stressed one dominates)
  const totalDrift = Math.max(climateDrift, airflowDrift, irrigationDrift, plantDrift)

  // ============================================
  // COHERENCE: inversely related to drift
  // ============================================
  // Coherence drops smoothly as drift increases
  // At drift=0, coherence≈0.85
  // At drift=0.5, coherence≈0.6
  // At drift=0.8, coherence≈0.3
  const coherence = Math.max(0.1, 0.85 - totalDrift * 0.9)

  // ============================================
  // CONFIDENCE: stability of measurements
  // ============================================
  // Confidence generally decreases under instability but recovers slightly if motion is smooth
  // This is different from coherence (fragility)
  const turbulence = Math.abs(airflowDrift - (frameIndex > 0 ? 0.1 : 0))
  const confidence = Math.max(0.3, 0.75 - turbulence * 0.5)

  // ============================================
  // FRAGILITY: brittleness regardless of confidence
  // ============================================
  // Fragility = how easily system breaks = (drift contribution) × (1 - coherence)
  // Higher fragility means system is close to failure
  const fragility = totalDrift * (1 - coherence)

  // ============================================
  // Build subsystem data
  // ============================================
  const subsystems = {
    climate: {
      drift: climateDrift,
      confidence: Math.max(0.2, confidence - climateDrift * 0.3),
    },
    airflow: {
      drift: airflowDrift,
      confidence: Math.max(0.2, confidence - airflowDrift * 0.5),
    },
    irrigation: {
      drift: irrigationDrift,
      confidence: Math.max(0.2, confidence - irrigationDrift * 0.3),
    },
    plant: {
      drift: plantDrift,
      confidence: Math.max(0.2, confidence - plantDrift * 0.2),
    },
  }

  // ============================================
  // Timestamp (arbitrary, for display)
  // ============================================
  const baseTime = new Date('2026-04-19T12:00:00Z')
  const frameDate = new Date(baseTime.getTime() + timeSeconds * 1000)
  const timestamp = frameDate.toISOString()

  return {
    time: timeSeconds,
    timestamp,
    coherence: Math.max(0.1, Math.min(1.0, coherence)),
    drift: Math.max(0, Math.min(1.0, totalDrift)),
    subsystems,
    fragility: Math.max(0, Math.min(1.0, fragility)),
    confidence: Math.max(0, Math.min(1.0, confidence)),
  }
}

/**
 * Generate a full sequence of continuous demo frames
 */
export function generateContinuousDemoSequence(
  durationSeconds: number = 60,
  frameRate: number = 30
): ContinuousFrameData[] {
  const totalFrames = durationSeconds * frameRate
  const frames: ContinuousFrameData[] = []

  for (let i = 0; i < totalFrames; i++) {
    frames.push(generateContinuousDemoFrame(i, totalFrames))
  }

  return frames
}
