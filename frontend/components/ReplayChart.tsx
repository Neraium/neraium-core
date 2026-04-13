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
    .map((f, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(f.metrics.structural_drift)}`)
    .join(" ");
  const healthPath = frames
    .map((f, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(f.metrics.relational_stability)}`)
    .join(" ");

  return (
    <section className="panel">
      <h3>Structural Drift & System Health</h3>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart" aria-label="drift chart">
        <rect x={toX(0)} y={pad} width={toX(20) - toX(0)} height={height - pad * 2} fill="#1e293b" opacity="0.25" />
        <rect x={toX(20)} y={pad} width={toX(35) - toX(20)} height={height - pad * 2} fill="#1d4ed8" opacity="0.17" />
        <rect x={toX(35)} y={pad} width={toX(55) - toX(35)} height={height - pad * 2} fill="#b45309" opacity="0.17" />
        <rect x={toX(55)} y={pad} width={toX(80) - toX(55)} height={height - pad * 2} fill="#991b1b" opacity="0.17" />
        <rect x={toX(80)} y={pad} width={toX(frames.length - 1) - toX(80)} height={height - pad * 2} fill="#065f46" opacity="0.17" />

        <path d={driftPath} stroke="#f97316" strokeWidth="3" fill="none" />
        <path d={healthPath} stroke="#22c55e" strokeWidth="3" fill="none" />
        <line x1={toX(currentIndex)} x2={toX(currentIndex)} y1={pad} y2={height - pad} stroke="#f8fafc" strokeDasharray="6,4" />
      </svg>
      <p className="subtle">Orange: structural drift, Green: relational stability, shaded regions: lifecycle phases.</p>
    </section>
  );
}
