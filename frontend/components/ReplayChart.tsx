"use client";

import { DemoFrame } from "@/lib/types";

type Props = {
  frames: DemoFrame[];
  currentIndex: number;
};

export function ReplayChart({ frames, currentIndex }: Props) {
  const width = 900;
  const height = 280;
  const pad = 24;

  const toX = (index: number) => pad + (index / Math.max(frames.length - 1, 1)) * (width - pad * 2);
  const toY = (value: number) => height - pad - value * (height - pad * 2);

  const driftPath = frames
    .map((f, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(f.metrics?.structural_drift ?? 0)}`)
    .join(" ");
  const healthPath = frames
    .map((f, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(f.metrics?.relational_stability ?? 0)}`)
    .join(" ");

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Drift & Stability Timeline</h3>
      </div>
      <p className="subtitle">Track structural drift (orange) and relational stability (green) across all frames. Shaded regions indicate lifecycle phases.</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart" aria-label="drift and stability chart">
        <rect x={toX(0)} y={pad} width={toX(20) - toX(0)} height={height - pad * 2} fill="#32d583" opacity="0.08" />
        <rect x={toX(20)} y={pad} width={toX(35) - toX(20)} height={height - pad * 2} fill="#5b9bd5" opacity="0.08" />
        <rect x={toX(35)} y={pad} width={toX(55) - toX(35)} height={height - pad * 2} fill="#f0b14a" opacity="0.08" />
        <rect x={toX(55)} y={pad} width={toX(80) - toX(55)} height={height - pad * 2} fill="#ea6f6f" opacity="0.08" />
        <rect x={toX(80)} y={pad} width={toX(frames.length - 1) - toX(80)} height={height - pad * 2} fill="#32d583" opacity="0.08" />

        <path d={driftPath} stroke="#f97316" strokeWidth="3" fill="none" />
        <path d={healthPath} stroke="#32d583" strokeWidth="3" fill="none" />
        <line x1={toX(currentIndex)} x2={toX(currentIndex)} y1={pad} y2={height - pad} stroke="#d7e5f5" strokeDasharray="6,4" strokeWidth="2" />
      </svg>
    </section>
  );
}
