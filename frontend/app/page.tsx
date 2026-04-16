"use client";

import { useEffect, useMemo, useState } from "react";

import { fetchDemoInit } from "@/lib/api";
import { DemoFrame } from "@/lib/types";
import { AppShell } from "@/components/AppShell";
import { Topbar } from "@/components/Topbar";
import { DemoHero } from "@/components/DemoHero";
import { ReplayChart } from "@/components/ReplayChart";
import { TetrahedronPanel } from "@/components/TetrahedronPanel";
import { PlaybackControls } from "@/components/PlaybackControls";
import { InsightPanels } from "@/components/InsightPanels";

const BASE_FPS = 14;

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
  const [frames, setFrames] = useState<DemoFrame[]>([]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    fetchDemoInit().then((payload) => setFrames(payload.frames));
  }, []);

  useEffect(() => {
    if (!playing || frames.length === 0) return;
    const intervalMs = 1000 / (BASE_FPS * speed);
    const timer = window.setInterval(() => {
      setIndex((current) => (current >= frames.length - 1 ? current : current + 1));
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [playing, frames.length, speed]);

  const currentFrame = useMemo(() => frames[index], [frames, index]);

  if (!currentFrame) {
    return (
      <AppShell>
        <Topbar phase="LOADING" confidence={0} index={0} total={1} />
        <div className="content">
          <div className="page">
            <div style={{ padding: "40px", textAlign: "center", color: "#96acc4" }}>
              Loading Neraium structural synthesis demo...
            </div>
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <Topbar
        phase={currentFrame.current_phase ?? "UNKNOWN"}
        confidence={currentFrame.metrics?.confidence ?? 0}
        index={index}
        total={frames.length}
      />
      <div className="content">
        <div className="page demo-section">
          <DemoHero frame={currentFrame} />

          <ReplayChart frames={frames} currentIndex={index} />

          <div className="tetra-chart-grid">
            <TetrahedronPanel position={extractPosition(currentFrame)} />
            <div>
              <div className="panel">
                <div className="panel-header">
                  <h3>Playback Controls</h3>
                </div>
                <PlaybackControls
                  playing={playing}
                  speed={speed}
                  onPlayPause={() => setPlaying((v) => !v)}
                  onRestart={() => {
                    setIndex(0);
                    setPlaying(true);
                  }}
                  onSpeed={setSpeed}
                />
              </div>
              <InsightPanels frame={currentFrame} />
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
