import { useEffect, useRef, useState } from 'react'

interface SystemData {
  drift: number
  stability: number
  coherence: number
  confidence: number
  timestamp: string
  systemState: string
}

interface InterpolatedData extends SystemData {
  interpolatedDrift: number
  interpolatedStability: number
  interpolatedCoherence: number
  interpolatedConfidence: number
}

/**
 * Spring physics interpolation for smooth, natural motion
 */
function springInterpolate(current: number, target: number, stiffness: number = 0.1) {
  // Simple spring easing: gradually move toward target
  return current + (target - current) * stiffness
}

/**
 * Easing function: ease-in-out cubic
 */
function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

/**
 * Hook to interpolate system data for smooth value updates
 * Prevents jitter and flickering by gradually moving toward target values
 */
export function useSystemInterpolation(
  targetData: SystemData | undefined,
  phase: string
): InterpolatedData {
  const [interpolated, setInterpolated] = useState<InterpolatedData>(() => ({
    drift: targetData?.drift ?? 0.2,
    stability: targetData?.stability ?? 0.8,
    coherence: targetData?.coherence ?? 0.7,
    confidence: targetData?.confidence ?? 0.6,
    timestamp: targetData?.timestamp ?? new Date().toISOString(),
    systemState: targetData?.systemState ?? 'Stable',
    interpolatedDrift: targetData?.drift ?? 0.2,
    interpolatedStability: targetData?.stability ?? 0.8,
    interpolatedCoherence: targetData?.coherence ?? 0.7,
    interpolatedConfidence: targetData?.confidence ?? 0.6,
  }))

  const animationFrameRef = useRef<number | null>(null)
  const previousRef = useRef(interpolated)

  useEffect(() => {
    if (!targetData) return

    const update = () => {
      const stiffness = phase === 'Stable' ? 0.08 : 0.15 // Faster updates during instability

      const newInterpolated = {
        ...targetData,
        interpolatedDrift: springInterpolate(
          previousRef.current.interpolatedDrift,
          targetData.drift,
          stiffness
        ),
        interpolatedStability: springInterpolate(
          previousRef.current.interpolatedStability,
          targetData.stability,
          stiffness
        ),
        interpolatedCoherence: springInterpolate(
          previousRef.current.interpolatedCoherence,
          targetData.coherence,
          stiffness
        ),
        interpolatedConfidence: springInterpolate(
          previousRef.current.interpolatedConfidence,
          targetData.confidence,
          stiffness
        ),
      }

      previousRef.current = newInterpolated
      setInterpolated(newInterpolated)

      // Continue animation if values haven't converged
      const hasConverged =
        Math.abs(newInterpolated.interpolatedDrift - targetData.drift) < 0.001 &&
        Math.abs(newInterpolated.interpolatedStability - targetData.stability) < 0.001 &&
        Math.abs(newInterpolated.interpolatedCoherence - targetData.coherence) < 0.001 &&
        Math.abs(newInterpolated.interpolatedConfidence - targetData.confidence) < 0.001

      if (!hasConverged) {
        animationFrameRef.current = requestAnimationFrame(update)
      }
    }

    animationFrameRef.current = requestAnimationFrame(update)

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [targetData, phase])

  return interpolated
}
