/**
 * Adaptive replay pacing and frame timing for Emergent demo.
 *
 * Ported from ui/replay_timing.py
 *
 * Principles:
 * - Stable phases linger longer to establish baseline
 * - Transition phases feel noticeably different in duration
 * - Verdict changes feel earned by replay progression
 * - Frame advancement is smooth and predictable
 */

import { DemoFrame } from "./types";

export function getPhaseType(row: Record<string, unknown>): string {
  if (!row || typeof row !== "object") return "STABLE";

  const transitionType = String(row.transition_type || "").toUpperCase();
  if (["TRANSITION", "REORGANIZATION"].includes(transitionType)) {
    return transitionType;
  }

  const drift = safeFloat(row.structural_drift_score || row.drift, 0.0);
  const stability = safeFloat(row.relational_stability_score || row.stability, 1.0);

  if (drift >= 0.55 || stability <= 0.45) return "TRANSITION";
  if (drift >= 0.35 || stability <= 0.65) return "TRANSITION";

  return "STABLE";
}

function safeFloat(value: unknown, defaultValue: number = 0.0): number {
  try {
    const num = Number(value);
    return isFinite(num) ? num : defaultValue;
  } catch {
    return defaultValue;
  }
}

export class ReplayPaceController {
  speedMultiplier: number;
  stableLingerRatio: number;
  transitionEmphasisRatio: number;
  private previousPhase: string = "STABLE";

  constructor(
    speedMultiplier: number = 1.0,
    stableLingerRatio: number = 1.4,
    transitionEmphasisRatio: number = 1.1
  ) {
    this.speedMultiplier = Math.max(0.1, Math.min(1.5, speedMultiplier));
    this.stableLingerRatio = Math.max(1.0, stableLingerRatio);
    this.transitionEmphasisRatio = Math.max(1.0, transitionEmphasisRatio);
  }

  /**
   * Calculate delay for current frame based on context.
   *
   * Returns delay in seconds that creates good pacing:
   * - Baseline delays around 0.65-0.8s per frame at speed 1.0
   * - Stable phases linger longer to establish baseline
   * - Transitions feel more urgent/active
   * - Speed multiplier scales inversely (higher speed = faster frames)
   */
  getStepDelay(frameIndex: number, rows: Record<string, unknown>[]): number {
    // Base delay calculation
    // Lower speed_multiplier = longer delay
    // Formula gives ~0.7s at speed=1.0
    const baseDelay = Math.max(0.45, 0.85 / Math.max(this.speedMultiplier, 0.1));

    // Get current phase
    let currentPhase = "STABLE";
    if (frameIndex >= 0 && frameIndex < rows.length) {
      const currentRow = rows[frameIndex];
      currentPhase = getPhaseType(currentRow);
    }

    // Apply phase-specific adjustment
    let phaseFactor = 1.0;

    if (currentPhase === "STABLE") {
      phaseFactor = this.stableLingerRatio;
    } else if (currentPhase === "TRANSITION" || currentPhase === "REORGANIZATION") {
      phaseFactor = this.transitionEmphasisRatio;
    }

    this.previousPhase = currentPhase;

    return Math.round(baseDelay * phaseFactor * 1000) / 1000;
  }

  reset(): void {
    this.previousPhase = "STABLE";
  }
}

export class VerdictStabilizer {
  private hysteresisThreshold: number;
  private displayedVerdict: string = "SUPPRESS";
  private realVerdict: string = "SUPPRESS";
  private signalStrength: number = 0.0;

  constructor(hysteresisThreshold: number = 0.08) {
    this.hysteresisThreshold = Math.max(0.01, Math.min(0.2, hysteresisThreshold));
  }

  /**
   * Apply hysteresis to a gate decision.
   *
   * The displayed verdict only changes when:
   * 1. Real verdict changes, AND
   * 2. Signal strength has moved enough to justify it
   */
  applyStability(
    gateDecision: Record<string, unknown>,
    signalStrength?: number
  ): Record<string, unknown> {
    const decision = { ...gateDecision };

    const realVerdict = String(decision.decision || "SUPPRESS").toUpperCase();

    let computedSignalStrength = signalStrength;
    if (computedSignalStrength === undefined) {
      const transition = decision.transition as Record<string, unknown> | undefined;
      if (transition && typeof transition === "object") {
        computedSignalStrength = safeFloat((transition as Record<string, unknown>).delta_drift, 0.0);
      } else {
        computedSignalStrength = 0.0;
      }
    }

    computedSignalStrength = safeFloat(computedSignalStrength, 0.0);
    this.realVerdict = realVerdict;
    this.signalStrength = computedSignalStrength;

    // Check if we should change displayed verdict
    if (this.displayedVerdict !== realVerdict) {
      if (Math.abs(computedSignalStrength) >= this.hysteresisThreshold) {
        this.displayedVerdict = realVerdict;
      }
    }

    return {
      ...decision,
      decision: this.displayedVerdict,
      _real_decision: realVerdict,
      _signal_strength: Math.round(computedSignalStrength * 1000000) / 1000000,
      _verdict_changed: this.displayedVerdict !== realVerdict,
    };
  }

  reset(): void {
    this.displayedVerdict = "SUPPRESS";
    this.realVerdict = "SUPPRESS";
    this.signalStrength = 0.0;
  }
}

export class ReasoningStateTracker {
  private changeThreshold: number;
  private previousState: Record<string, unknown> = {};

  constructor(changeThreshold: number = 0.06) {
    this.changeThreshold = Math.max(0.01, Math.min(0.2, changeThreshold));
  }

  /**
   * Check if reasoning should be re-rendered.
   * Returns true if metrics have changed enough to warrant update.
   */
  shouldUpdateReasoning(
    currentRow: Record<string, unknown>,
    previousRow?: Record<string, unknown>
  ): boolean {
    if (!currentRow || typeof currentRow !== "object") {
      return false;
    }

    const prev = previousRow || this.previousState;

    const keyFields = [
      "structural_drift_score",
      "relational_stability_score",
      "event_admitted",
      "transition_type",
      "system_health",
    ];

    for (const field of keyFields) {
      const currVal = currentRow[field];
      const prevVal = prev[field];

      const currNum = safeFloat(currVal);
      const prevNum = safeFloat(prevVal);

      if (isFinite(currNum) && isFinite(prevNum)) {
        if (Math.abs(currNum - prevNum) > this.changeThreshold) {
          this.previousState = { ...currentRow };
          return true;
        }
      } else if (String(currVal) !== String(prevVal)) {
        this.previousState = { ...currentRow };
        return true;
      }
    }

    return false;
  }

  reset(): void {
    this.previousState = {};
  }
}

export class InterpolationHelper {
  static lerp(start: number, end: number, t: number): number {
    return start + (end - start) * Math.max(0.0, Math.min(1.0, t));
  }

  static smoothStep(t: number): number {
    t = Math.max(0.0, Math.min(1.0, t));
    return t * t * (3.0 - 2.0 * t);
  }
}

export interface PhaseProgression {
  stable_count: number;
  transition_count: number;
  reorganization_count: number;
  phase_sequence: string[];
  average_stable_run_length: number;
  transition_entry_indices: number[];
}

export function calculatePhaseProgression(rows: Record<string, unknown>[]): PhaseProgression {
  if (!rows || rows.length === 0) {
    return {
      stable_count: 0,
      transition_count: 0,
      reorganization_count: 0,
      phase_sequence: [],
      average_stable_run_length: 0,
      transition_entry_indices: [],
    };
  }

  const phaseSequence = rows.map(getPhaseType);

  return {
    stable_count: phaseSequence.filter((p) => p === "STABLE").length,
    transition_count: phaseSequence.filter((p) => p === "TRANSITION").length,
    reorganization_count: phaseSequence.filter((p) => p === "REORGANIZATION").length,
    phase_sequence: phaseSequence,
    average_stable_run_length: calculateAverageRunLength(phaseSequence, "STABLE"),
    transition_entry_indices: phaseSequence
      .map((phase, i) => ({
        phase,
        i,
        isPrevStable: i === 0 || phaseSequence[i - 1] === "STABLE",
      }))
      .filter((item) => ["TRANSITION", "REORGANIZATION"].includes(item.phase) && item.isPrevStable)
      .map((item) => item.i),
  };
}

function calculateAverageRunLength(sequence: string[], value: string): number {
  if (!sequence || sequence.length === 0 || !sequence.includes(value)) {
    return 0.0;
  }

  const runs: number[] = [];
  let currentRun = 0;

  for (const item of sequence) {
    if (item === value) {
      currentRun++;
    } else {
      if (currentRun > 0) {
        runs.push(currentRun);
      }
      currentRun = 0;
    }
  }

  if (currentRun > 0) {
    runs.push(currentRun);
  }

  return runs.length > 0 ? runs.reduce((a, b) => a + b, 0) / runs.length : 0.0;
}
