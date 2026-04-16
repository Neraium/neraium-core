"use client";

type Props = {
  phase: string;
  confidence: number;
  index: number;
  total: number;
};

export function Topbar({ phase, confidence, index, total }: Props) {
  return (
    <header className="topbar">
      <div className="topbar-main">
        <div className="topbar-brand-row">
          <a href="/" className="topbar-wordmark">Neraium</a>
        </div>
        <div className="title-block telemetry-title">
          <h2>Structural Synthesis Demo</h2>
          <p className="subtitle">Real-time tetrahedral state evolution across 450 frames</p>
        </div>
      </div>
      <div className="telemetry-status-strip" aria-label="System status">
        <span className="chip chip-live">
          <span className="live-dot" aria-hidden="true"></span>
          <span>LIVE</span>
        </span>
        <span className="chip">Phase <strong>{phase}</strong></span>
        <span className="chip">Confidence <strong>{(confidence * 100).toFixed(0)}%</strong></span>
        <span className="chip">Frame <strong>{index + 1}/{total}</strong></span>
      </div>
    </header>
  );
}
