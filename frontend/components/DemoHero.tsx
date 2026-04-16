"use client";

import { DemoFrame } from "@/lib/types";

type Props = {
  frame: DemoFrame;
};

export function DemoHero({ frame }: Props) {
  const health = Math.round((frame.metrics?.coherence ?? 0) * 100);
  const drift = (frame.metrics?.structural_drift ?? 0).toFixed(2);
  const stability = (frame.metrics?.relational_stability ?? 0).toFixed(2);
  const confidence = Math.round((frame.metrics?.confidence ?? 0) * 100);

  return (
    <section className="panel dashboard-hero">
      <div className="dashboard-hero-grid">
        <div className="dashboard-hero-primary">
          <p className="dashboard-hero-kicker">Current synthesis state</p>
          <div className="dashboard-hero-score-row">
            <div className="health-ring-wrap dashboard-hero-health-col">
              <svg className="health-ring" viewBox="0 0 120 120" aria-hidden="true">
                <circle className="health-ring-bg" cx="60" cy="60" r="48"></circle>
                <circle
                  className="health-ring-arc"
                  cx="60"
                  cy="60"
                  r="48"
                  style={{
                    strokeDasharray: `${(health / 100) * 300} 300`,
                  }}
                ></circle>
              </svg>
              <div className="health-ring-label">
                <span className="health-ring-value">{health}</span>
                <span className="health-ring-caption">Coherence</span>
              </div>
            </div>
            <div className="dashboard-hero-state-block">
              <div className="dashboard-risk-row">
                <p className="signal-label">Structural Drift</p>
                <p className="dashboard-risk-value">{drift}</p>
              </div>
              <p className="dashboard-trend-line">Relational Stability: {stability}</p>
              <p className="dashboard-narrative-strip">
                System position in tetrahedral state space evolving in real-time. Confidence score reflects alignment certainty.
              </p>
              <p className="dashboard-operator-line">Analysis confidence: {confidence}%</p>
            </div>
          </div>
        </div>
        <aside className="dashboard-hero-aside">
          <article className="insight-tile">
            <p className="insight-tile-label">Current Phase</p>
            <p className="insight-tile-sub">{frame.current_phase ?? "Unknown"}</p>
            <p className="insight-tile-label">Verdict</p>
            <p className="insight-tile-sub">{frame.verdict || "Pending"}</p>
          </article>
          <article className="insight-tile">
            <p className="insight-tile-label">Timestamp</p>
            <p className="insight-tile-sub">{frame.timestamp || "N/A"}</p>
            <p className="insight-tile-label">Status</p>
            <p className="insight-tile-sub">{frame.status || "Active"}</p>
          </article>
        </aside>
      </div>
    </section>
  );
}
