"use client";

type Props = {
  phase: string;
  confidence: number;
  index: number;
  total: number;
};

export function HeaderBar({ phase, confidence, index, total }: Props) {
  return (
    <header className="header panel">
      <div>
        <h1>Neraium</h1>
        <p className="subtle">Synthetic Structural Demo</p>
      </div>
      <div className="stats">
        <span>Phase: {phase}</span>
        <span>Confidence: {(confidence * 100).toFixed(1)}%</span>
        <span>Frame: {index + 1}/{total}</span>
      </div>
    </header>
  );
}
