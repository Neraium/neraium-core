import { useEffect, useState } from 'react'

interface FrameData {
  structural_drift_score: number
  [key: string]: any
}

interface DriftAlertProps {
  frames: FrameData[]
  currentIndex: number
}

export default function DriftAlert({ frames, currentIndex }: DriftAlertProps) {
  const [alertTriggered, setAlertTriggered] = useState(false)
  const [dismissedAt, setDismissedAt] = useState<number | null>(null)

  const CRITICAL_THRESHOLD = 0.78
  const WATCH_THRESHOLD = 0.55

  useEffect(() => {
    if (frames.length === 0) return

    const safeIndex = Math.max(0, Math.min(currentIndex, frames.length - 1))
    const currentFrame = frames[safeIndex]
    const driftScore = currentFrame.structural_drift_score || 0

    if (driftScore >= CRITICAL_THRESHOLD) {
      setAlertTriggered(true)
      setDismissedAt(null)
    } else {
      setAlertTriggered(false)
    }
  }, [frames, currentIndex])

  const handleDismiss = () => {
    setDismissedAt(Date.now())
    setAlertTriggered(false)
  }

  if (!alertTriggered || dismissedAt) {
    return null
  }

  const safeIndex = Math.max(0, Math.min(currentIndex, Math.max(frames.length - 1, 0)))
  const driftPercent = (frames[safeIndex]?.structural_drift_score || 0) * 100

  return (
    <div className="drift-alert critical-alert" role="alert" aria-live="assertive">
      <div className="alert-icon">⚠</div>
      <div className="alert-content">
        <div className="alert-title">System Drift Detected</div>
        <div className="alert-message">
          Structural drift has reached critical levels ({driftPercent.toFixed(1)}%).
          System may require immediate attention or remediation.
        </div>
      </div>
      <button
        className="alert-dismiss"
        onClick={handleDismiss}
        aria-label="Dismiss alert"
        type="button"
      >
        ✕
      </button>
    </div>
  )
}
