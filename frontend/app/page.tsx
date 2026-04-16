"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { fetchDemoInit } from "@/lib/api";
import { DemoFrame } from "@/lib/types";
import {
  ReplayPaceController,
  VerdictStabilizer,
  ReasoningStateTracker,
} from "@/lib/replay-timing";
import { HeaderBar } from "@/components/HeaderBar";
import { ReplayChart } from "@/components/ReplayChart";
import { TetrahedronPanel } from "@/components/TetrahedronPanel";
import { PlaybackControls } from "@/components/PlaybackControls";
import { InsightPanels } from "@/components/InsightPanels";

function extractPosition(frame: DemoFrame): [number, number, number] {
  const pos = frame.tetrahedral_state?.position;
  if (Array.isArray(pos) && pos.length >= 3) {
    return [Number(pos[0]), Number(pos[1]), Number(pos[2])];
  }
  if (pos && typeof pos === "object") {
    const v = pos as { x?: number; y?: number; z?: number };
    return [Number(v.x ?? 0), Number(v.y ?? 0), Number(v.z ?? 0)];
  }
  return [0, 0, 0];
}

export default function HomePage() {
  // Core state
  const [frames, setFrames] = useState<DemoFrame[]>([]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);

  // Refs for adaptive timing
  const paceControllerRef = useRef<ReplayPaceController | null>(null);
  const verdictStabilizerRef = useRef<VerdictStabilizer | null>(null);
  const reasoningTrackerRef = useRef<ReasoningStateTracker | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const shouldUpdateReasoningRef = useRef<boolean>(true);

  // Initialize demo data and timing controllers
  useEffect(() => {
    fetchDemoInit().then((payload) => {
      setFrames(payload.frames);
      // Initialize timing controllers once frames are loaded
      if (payload.frames.length > 0) {
        paceControllerRef.current = new ReplayPaceController(speed);
        verdictStabilizerRef.current = new VerdictStabilizer();
        reasoningTrackerRef.current = new ReasoningStateTracker();
      }
    });
  }, [speed]);

  // Adaptive frame playback loop
  useEffect(() => {
    if (!playing || frames.length === 0 || !paceControllerRef.current) return;

    const playFrame = () => {
      const now = performance.now();
      const currentFrame = frames[index];

      if (!currentFrame) return;

      // Get adaptive delay based on current frame's phase
      const delaySeconds = paceControllerRef.current!.getStepDelay(index, frames);
      const delayMs = delaySeconds * 1000;

      // Check if enough time has passed to advance
      if (now - lastFrameTimeRef.current >= delayMs) {
        lastFrameTimeRef.current = now;

        // Check if reasoning should update
        const previousFrame = index > 0 ? frames[index - 1] : undefined;
        shouldUpdateReasoningRef.current = reasoningTrackerRef.current!.shouldUpdateReasoning(
          currentFrame as Record<string, unknown>,
          previousFrame as Record<string, unknown> | undefined
        );

        // Apply verdict stabilization
        if (currentFrame.gate_decision && verdictStabilizerRef.current) {
          const stabilizedDecision = verdictStabilizerRef.current.applyStability(
            currentFrame.gate_decision
          );
          // The verdict is now stabilized internally; no need to store it separately
          // because we use it for display via the VerdictStabilizer
        }

        // Advance to next frame
        if (index < frames.length - 1) {
          setIndex((current) => current + 1);
        }
        // else: reached the end, stay on last frame
      }
    };

    const animationFrameId = requestAnimationFrame(playFrame);
    return () => cancelAnimationFrame(animationFrameId);
  }, [playing, frames, index, speed]);

  // Current frame with adaptive state
  const currentFrame = useMemo(() => {
    if (frames.length === 0 || index >= frames.length) return null;

    const frame = frames[index];
    const previousFrame = index > 0 ? frames[index - 1] : null;

    // Apply verdict stabilization to the displayed frame
    if (frame.gate_decision && verdictStabilizerRef.current) {
      const stabilizedDecision = verdictStabilizerRef.current.applyStability(
        frame.gate_decision
      );
      return {
        ...frame,
        gate_decision: stabilizedDecision,
        _should_update_reasoning: shouldUpdateReasoningRef.current,
        _previous_frame: previousFrame,
      };
    }

    return {
      ...frame,
      _should_update_reasoning: shouldUpdateReasoningRef.current,
      _previous_frame: previousFrame,
    };
  }, [frames, index]);

  if (!currentFrame) {
    return <main className="loading">Loading demo...</main>;
  }

  return (
    <main className="system">
      <HeaderBar
        phase={currentFrame.current_phase ?? "UNKNOWN"}
        confidence={currentFrame.metrics?.confidence ?? 0}
        playing={playing}
      />

      <div className="summary-band">
        <div className="summary-item">
          <span className="summary-label">Phase</span>
          <span className="summary-value">{currentFrame.current_phase ?? "UNKNOWN"}</span>
        </div>
        <div className="summary-divider" />
        <div className="summary-item">
          <span className="summary-label">Confidence</span>
          <span className="summary-value">{Math.round((currentFrame.metrics?.confidence ?? 0) * 100)}%</span>
        </div>
        <div className="summary-divider" />
        <div className="summary-item">
          <span className="summary-label">Stability</span>
          <span className="summary-value">{(currentFrame.metrics?.relational_stability ?? 0).toFixed(2)}</span>
        </div>
      </div>

      <section className="view-main">
        <ReplayChart frames={frames} currentIndex={index} />
        <TetrahedronPanel
          position={extractPosition(currentFrame)}
          confidence={currentFrame.metrics?.confidence ?? 0.5}
        />
      </section>

      <InsightPanels frame={currentFrame} />

      <PlaybackControls
        playing={playing}
        speed={speed}
        onPlayPause={() => setPlaying((v) => !v)}
        onRestart={() => {
          setIndex(0);
          setPlaying(true);
          lastFrameTimeRef.current = 0;
          paceControllerRef.current?.reset();
          verdictStabilizerRef.current?.reset();
          reasoningTrackerRef.current?.reset();
        }}
        onSpeed={(newSpeed) => {
          setSpeed(newSpeed);
          if (paceControllerRef.current) {
            paceControllerRef.current.speedMultiplier = newSpeed;
          }
        }}
      />
    </main>
  );
}
