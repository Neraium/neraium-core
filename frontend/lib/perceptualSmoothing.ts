/**
 * Perceptual state smoothing and value formatting
 * Implements temporal smoothing, value hold times, and exponential decay
 */

interface SmoothedValue {
  target: number
  current: number
  lastUpdateTime: number
  holdUntil: number
}

export class PerceptualStateManager {
  private smoothedValues: Map<string, SmoothedValue> = new Map()
  private lastUIUpdateTime: number = 0
  private minUIUpdateIntervalMs: number = 150 // 6-7fps

  constructor() {
    this.initializeValues()
  }

  private initializeValues() {
    const keys = [
      'coherence',
      'drift',
      'fragility',
      'confidence',
      'climate_drift',
      'airflow_drift',
      'irrigation_drift',
      'plant_drift',
    ]

    keys.forEach((key) => {
      this.smoothedValues.set(key, {
        target: 0,
        current: 0,
        lastUpdateTime: Date.now(),
        holdUntil: Date.now(),
      })
    })
  }

  /**
   * Update a target value (internal state)
   * This happens frequently, even if UI doesn't update
   */
  updateTargetValue(key: string, newValue: number): void {
    if (!this.smoothedValues.has(key)) {
      this.smoothedValues.set(key, {
        target: newValue,
        current: newValue,
        lastUpdateTime: Date.now(),
        holdUntil: Date.now(),
      })
      return
    }

    const smoothed = this.smoothedValues.get(key)!
    smoothed.target = Math.max(0, Math.min(1, newValue))
    smoothed.lastUpdateTime = Date.now()
  }

  /**
   * Apply exponential smoothing to move current toward target
   * Call this frequently (every frame of internal simulation)
   */
  updateSmoothedValues(deltaTimeMs: number): void {
    const now = Date.now()

    this.smoothedValues.forEach((smoothed) => {
      // Don't update during hold time
      if (now < smoothed.holdUntil) return

      // Exponential smoothing: move toward target at rate 0.08 per frame
      // This is independent of frame rate
      const alpha = 0.08
      smoothed.current += (smoothed.target - smoothed.current) * alpha

      // Set hold time: lock value for 300ms to prevent flicker
      if (Math.abs(smoothed.target - smoothed.current) < 0.01) {
        smoothed.holdUntil = now + 300 // Hold for 300ms when close to target
      }
    })
  }

  /**
   * Check if UI should update (rate limited)
   */
  shouldUpdateUI(): boolean {
    const now = Date.now()
    if (now - this.lastUIUpdateTime >= this.minUIUpdateIntervalMs) {
      this.lastUIUpdateTime = now
      return true
    }
    return false
  }

  /**
   * Get current smoothed value (for UI display)
   */
  getDisplayValue(key: string): number {
    const smoothed = this.smoothedValues.get(key)
    return smoothed?.current ?? 0
  }

  /**
   * Format percentage for display (rounded to whole numbers)
   */
  formatPercent(key: string): string {
    const value = this.getDisplayValue(key)
    return `${Math.round(value * 100)}%`
  }

  /**
   * Format confidence/fragility (rounded to nearest 5%)
   */
  formatConfidence(key: string): string {
    const value = this.getDisplayValue(key)
    return `${Math.round(value * 20) * 5}%`
  }

  /**
   * Get all smoothed values as object (for snapshot)
   */
  getSmoothedState() {
    const state: Record<string, number> = {}
    this.smoothedValues.forEach((smoothed, key) => {
      state[key] = smoothed.current
    })
    return state
  }
}

/**
 * Text update manager with threshold-based updates and fade transitions
 */
export class TextUpdateManager {
  private lastTexts: Map<string, string> = new Map()
  private transitionEndTime: Map<string, number> = new Map()
  private textChangeThresholds: Map<string, number> = new Map([
    ['state', 0.1], // Only update state text if threshold difference > 0.1
    ['focus', 0.2], // Only update focus if significant change
    ['recoverability', 0.15],
  ])

  /**
   * Check if text should update based on threshold
   */
  shouldUpdateText(key: string, newValue: string): boolean {
    const lastText = this.lastTexts.get(key)
    if (!lastText) return true // First time

    // For text, we just check if it changed
    return lastText !== newValue
  }

  /**
   * Update text with fade transition timing
   */
  updateText(key: string, newValue: string): { text: string; opacity: number } {
    const shouldUpdate = this.shouldUpdateText(key, newValue)

    if (shouldUpdate) {
      this.lastTexts.set(key, newValue)
      this.transitionEndTime.set(key, Date.now() + 300) // 300ms fade
    }

    // Calculate fade opacity (full opacity until fade time)
    const fadeEndTime = this.transitionEndTime.get(key) ?? Date.now()
    const timeUntilFade = fadeEndTime - Date.now()
    const opacity = Math.max(0.4, Math.min(1, timeUntilFade / 300))

    return {
      text: this.lastTexts.get(key) ?? newValue,
      opacity,
    }
  }

  /**
   * Get current text with fade effect
   */
  getText(key: string): { text: string; opacity: number } {
    const text = this.lastTexts.get(key) ?? ''
    const fadeEndTime = this.transitionEndTime.get(key) ?? Date.now()
    const timeUntilFade = fadeEndTime - Date.now()
    const opacity = Math.max(0.4, Math.min(1, timeUntilFade / 300))

    return { text, opacity }
  }
}
