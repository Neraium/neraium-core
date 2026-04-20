import { useState, useEffect, useRef } from 'react'

type SystemPhase = 'Stable' | 'Drift forming' | 'Instability forming' | 'Critical'

interface PhaseConfig {
  phase: SystemPhase
  minDuration: number // milliseconds
}

const PHASE_SEQUENCE: PhaseConfig[] = [
  { phase: 'Stable', minDuration: 8000 }, // 8-15 seconds
  { phase: 'Drift forming', minDuration: 8000 },
  { phase: 'Instability forming', minDuration: 8000 },
  { phase: 'Critical', minDuration: 8000 },
]

const PHASE_DURATION_VARIANCE = 7000 // Add 0-7 seconds random variance

/**
 * Hook to manage discrete system phases with smooth transitions.
 * Each phase lasts 8-15 seconds, then cycles through the sequence.
 */
export function usePhaseController(isPlaying: boolean, speed: number = 1) {
  const [phase, setPhase] = useState<SystemPhase>('Stable')
  const [phaseProgress, setPhaseProgress] = useState(0) // 0-1
  const currentPhaseIndexRef = useRef(0)
  const phaseStartTimeRef = useRef(Date.now())
  const phaseDurationRef = useRef(
    PHASE_SEQUENCE[0].minDuration + Math.random() * PHASE_DURATION_VARIANCE
  )
  const animationFrameRef = useRef<number | null>(null)

  useEffect(() => {
    if (!isPlaying) {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
      return
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
        currentPhaseIndexRef.current = (currentPhaseIndexRef.current + 1) % PHASE_SEQUENCE.length
        const nextPhase = PHASE_SEQUENCE[currentPhaseIndexRef.current]

        setPhase(nextPhase.phase)
        phaseStartTimeRef.current = now
        phaseDurationRef.current = nextPhase.minDuration + Math.random() * PHASE_DURATION_VARIANCE

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
