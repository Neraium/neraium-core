"use client";

import { useMemo } from "react";
import { DemoFrame } from "@/lib/types";

type Props = {
  frames: DemoFrame[];
  currentIndex: number;
};

type PhaseRegion = {
  startIndex: number;
  endIndex: number;
  phase: string;
  color: string;
};

type InstabilityRegion = {
  startIndex: number;
  endIndex: number;
  severity: number; // 0-1
};

function getPhaseColor(phase: string): string {
  const p = String(phase || "").toUpperCase();
  if (p.includes("STABLE")) return "#065f46";
  if (p.includes("DRIFT") || p.includes("DEGRA")) return "#b45309";
  if (p.includes("CRITICAL") || p.includes("ALERT")) return "#991b1b";
  if (p.includes("RECOVERY")) return "#0b5e1f";
  return "#1e293b";
}

function identifyPhaseRegions(frames: DemoFrame[]): PhaseRegion[] {
  if (frames.length === 0) return [];

  const regions: PhaseRegion[] = [];
  let currentPhase = frames[0].current_phase;
  let startIndex = 0;

  for (let i = 1; i < frames.length; i++) {
    if (frames[i].current_phase !== currentPhase) {
      regions.push({
        startIndex,
        endIndex: i - 1,
        phase: currentPhase,
        color: getPhaseColor(currentPhase),
      });
      currentPhase = frames[i].current_phase;
      startIndex = i;
    }
  }

  // Add final region
  regions.push({
    startIndex,
    endIndex: frames.length - 1,
    phase: currentPhase,
    color: getPhaseColor(currentPhase),
  });

  return regions;
}

function identifyInstabilityRegions(frames: DemoFrame[]): InstabilityRegion[] {
  const regions: InstabilityRegion[] = [];
  const threshold = 0.4; // Below this stability = unstable

  let inUnstable = false;
  let unstableStart = 0;
  let maxUnstability = 0;

  for (let i = 0; i < frames.length; i++) {
    const stability = frames[i].metrics?.relational_stability ?? 1;
    const isUnstable = stability < threshold;

    if (isUnstable && !inUnstable) {
      unstableStart = i;
      inUnstable = true;
      maxUnstability = 1 - stability;
    } else if (!isUnstable && inUnstable) {
      regions.push({
        startIndex: unstableStart,
        endIndex: i - 1,
        severity: maxUnstability,
      });
      inUnstable = false;
      maxUnstability = 0;
    } else if (inUnstable) {
      maxUnstability = Math.max(maxUnstability, 1 - stability);
    }
  }

  if (inUnstable) {
    regions.push({
      startIndex: unstableStart,
      endIndex: frames.length - 1,
      severity: maxUnstability,
    });
  }

  return regions;
}

export function ReplayChart({ frames, currentIndex }: Props) {
  const width = 900;
  const height = 280;
  const pad = 24;

  const phaseRegions = useMemo(() => identifyPhaseRegions(frames), [frames]);
  const instabilityRegions = useMemo(() => identifyInstabilityRegions(frames), [frames]);

  const toX = (index: number) => pad + (index / Math.max(frames.length - 1, 1)) * (width - pad * 2);
  const toY = (value: number) => height - pad - value * (height - pad * 2);

  const driftPath = frames
    .map((f, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(f.metrics?.structural_drift ?? 0)}`)
    .join(" ");
  const healthPath = frames
    .map((f, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(f.metrics?.relational_stability ?? 0)}`)
    .join(" ");

  const currentX = toX(currentIndex);
  const progressPercent = (currentIndex / Math.max(frames.length - 1, 1)) * 100;

  return (
    <section className="panel">
      <h3>System Evolution</h3>
      <p className="subtle">Structural drift (orange) and relational stability (green) across phases. Red highlights instability periods.</p>

      <div className="chart-container">
        <svg viewBox={`0 0 ${width} ${height}`} className="chart" aria-label="system evolution chart">
          <defs>
            <linearGradient id="instabilityGrad" x1="0%" x2="0%" y1="0%" y2="100%">
              <stop offset="0%" stopColor="#991b1b" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#991b1b" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Phase regions as background */}
          {phaseRegions.map((region, idx) => (
            <rect
              key={`phase-${idx}`}
              x={toX(region.startIndex)}
              y={pad}
              width={toX(region.endIndex + 1) - toX(region.startIndex)}
              height={height - pad * 2}
              fill={region.color}
              opacity="0.08"
            />
          ))}

          {/* Instability overlay */}
          {instabilityRegions.map((region, idx) => (
            <rect
              key={`unstable-${idx}`}
              x={toX(region.startIndex)}
              y={pad}
              width={toX(region.endIndex + 1) - toX(region.startIndex)}
              height={height - pad * 2}
              fill="url(#instabilityGrad)"
            />
          ))}

          {/* Phase transition boundaries */}
          {phaseRegions.map((region, idx) => {
            if (idx === phaseRegions.length - 1) return null; // Skip last, no boundary after
            return (
              <line
                key={`boundary-${idx}`}
                x1={toX(region.endIndex + 0.5)}
                x2={toX(region.endIndex + 0.5)}
                y1={pad}
                y2={height - pad}
                stroke="#475569"
                strokeWidth="1"
                strokeDasharray="3,2"
                opacity="0.4"
              />
            );
          })}

          {/* Data traces */}
          <path d={driftPath} stroke="#f97316" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <path d={healthPath} stroke="#22c55e" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" />

          {/* Current position indicator with animation */}
          <g className="current-marker">
            <line
              x1={currentX}
              x2={currentX}
              y1={pad - 4}
              y2={height - pad + 4}
              stroke="#64b5f6"
              strokeWidth="2"
              opacity="0.7"
            />
            <circle cx={currentX} cy={pad - 6} r="3" fill="#64b5f6" opacity="0.8" />
          </g>

          {/* Progress overlay glow */}
          <rect
            x={pad}
            y={pad}
            width={currentX - pad}
            height={height - pad * 2}
            fill="#64b5f6"
            opacity="0.02"
          />
        </svg>

        {/* Progress bar underneath */}
        <div className="chart-progress">
          <div className="progress-bar" style={{ width: `${progressPercent}%` }} />
        </div>
      </div>

      {/* Phase legend */}
      <div className="chart-legend">
        {phaseRegions.map((region, idx) => (
          <span key={`legend-${idx}`} className="legend-item">
            <span className="legend-color" style={{ backgroundColor: region.color }} />
            <span className="legend-label">{region.phase}</span>
          </span>
        ))}
      </div>
    </section>
  );
}
