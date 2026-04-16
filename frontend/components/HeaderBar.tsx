"use client";

type Props = {
  phase: string;
  confidence: number;
  playing: boolean;
};

export function HeaderBar({ phase, confidence, playing }: Props) {
  return (
    <header className="header">
      {/* Left: System name */}
      <div className="header-left">
        <h1>Neraium</h1>
      </div>

      {/* Center: Phase and confidence (prominent) */}
      <div className="header-center">
        <div className="phase-block">
          <span className="phase-label">Phase</span>
          <span className="phase-value">{phase}</span>
        </div>
        <div className="confidence-block">
          <span className="confidence-label">Confidence</span>
          <span className="confidence-value">{Math.round(confidence * 100)}%</span>
        </div>
      </div>

      {/* Right: Live status */}
      <div className="header-right">
        <div className={`status-indicator ${playing ? "live" : "paused"}`}>
          <span className="status-dot" />
          <span className="status-text">{playing ? "LIVE" : "PAUSED"}</span>
        </div>
      </div>
    </header>
  );
}
