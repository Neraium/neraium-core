import { useState, useEffect, useRef } from 'react'

type SystemPhase = 'Stable' | 'Drift forming' | 'Instability forming' | 'Critical'

interface PhaseConfig {
  phase: SystemPhase
  minDuration: number // milliseconds
  maxDuration: number // milliseconds
  weight: number // how long it lingers (0-1, higher = lingers longer)
}

const PHASE_SEQUENCE: PhaseConfig[] = [
  { phase: 'Stable', minDuration: 15000, maxDuration: 20000, weight: 0.3 },
  { phase: 'Drift forming', minDuration: 18000, maxDuration: 25000, weight: 0.5 },
  { phase: 'Instability forming', minDuration: 20000, maxDuration: 28000, weight: 0.7 },
  { phase: 'Critical', minDuration: 25000, maxDuration: 40000, weight: 0.9 }, // Lingers longer
]

/**
 * Hook to manage discrete system phases with directional progression bias.
 *
 * Instead of cycling randomly, phases progress toward criticality with:
 * - Faster drift buildup (acceleration toward instability)
 * - Slower recovery (resistance toward stability)
 * - Critical lingers longer (phase weight)
 * - Causal feel, not timed
 */
export function usePhaseController(isPlaying: boolean, speed: number = 1) {
  const [phase, setPhase] = useState<SystemPhase>('Stable')
  const [phaseProgress, setPhaseProgress] = useState(0) // 0-1
  const currentPhaseIndexRef = useRef(0)
  const phaseStartTimeRef = useRef(Date.now())
  const phaseDurationRef = useRef(0)
  const animationFrameRef = useRef<number | null>(null)
  const hasLingeredRef = useRef(false) // Track if critical phase lingered

  useEffect(() => {
    if (!isPlaying) {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
      return
    }

    // Initialize duration on first run
    if (phaseDurationRef.current === 0) {
      const config = PHASE_SEQUENCE[0]
      const baseDuration = config.minDuration + Math.random() * (config.maxDuration - config.minDuration)
      // Apply weight to duration (critical phases linger longer)
      phaseDurationRef.current = baseDuration + config.weight * 8000
    }

    const updatePhase = () => {
      const now = Date.now()
      const elapsed = (now - phaseStartTimeRef.current) * speed
      const currentDuration = phaseDurationRef.current

      // Calculate progress within current phase (0 to 1)
      const progress = Math.min(elapsed / currentDuration, 1)
      setPhaseProgress(progress)

      // If phase duration exceeded, move to next phase
      if (elapsed >= currentDuration) {
        // Directional progression with temporal gravity
        const currentIndex = currentPhaseIndexRef.current
        const isRecoveryPhase = currentIndex > 1 // Coming back from Critical

        if (isRecoveryPhase && hasLingeredRef.current) {
          // Recovery is slower: move back one phase instead of forward
          currentPhaseIndexRef.current = Math.max(0, currentIndex - 1)
          hasLingeredRef.current = false
        } else if (currentIndex < PHASE_SEQUENCE.length - 1) {
          // Progression: move toward instability (faster acceleration)
          currentPhaseIndexRef.current += 1
        } else {
          // At critical: linger before recovery
          hasLingeredRef.current = true
        }

        const nextConfig = PHASE_SEQUENCE[currentPhaseIndexRef.current]
        const baseDuration = nextConfig.minDuration +
          Math.random() * (nextConfig.maxDuration - nextConfig.minDuration)

        // Temporal gravity: asymmetric time
        // Drift builds faster (acceleration toward instability)
        // Recovery is slower (resistance toward stability)
        const progressionBias = currentIndex < 3 ? 0.7 : 1.2 // Forward is faster, backward is slower
        const finalDuration = baseDuration * progressionBias + nextConfig.weight * 8000

        setPhase(nextConfig.phase)
        phaseStartTimeRef.current = now
        phaseDurationRef.current = finalDuration

        // Reset progress for next phase
        setPhaseProgress(0)
      }

      animationFrameRef.current = requestAnimationFrame(updatePhase)
    }

    animationFrameRef.current = requestAnimationFrame(updatePhase)

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [isPlaying, speed])

  return { phase, phaseProgress }
}

/**
 * Get color based on system phase for drift visualization
 */
export function getPhaseColor(phase: SystemPhase): string {
  const colors: Record<SystemPhase, string> = {
    'Stable': '#38BDF8', // cyan-blue
    'Drift forming': '#FBBF24', // amber
    'Instability forming': '#FB923C', // orange
    'Critical': '#EF4444', // red
  }
  return colors[phase]
}

/**
 * Get intensity value (0-1) based on phase for visual effects
 */
export function getPhaseIntensity(phase: SystemPhase, progress: number): number {
  const baseIntensity: Record<SystemPhase, number> = {
    'Stable': 0.15,
    'Drift forming': 0.4,
    'Instability forming': 0.7,
    'Critical': 0.95,
  }

  // Add slight breathing effect with sine wave
  const breathing = Math.sin(progress * Math.PI * 4) * 0.05
  return Math.max(0, Math.min(1, baseIntensity[phase] + breathing))
}
