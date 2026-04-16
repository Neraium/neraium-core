"use client";

import { useEffect, useMemo, useState } from "react";

import { fetchDemoInit } from "@/lib/api";
import { DemoFrame } from "@/lib/types";
import { HeaderBar } from "@/components/HeaderBar";
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
    return <main className="loading">Loading...</main>;
  }

  return (
    <main className="system">
      <HeaderBar
        phase={currentFrame.current_phase ?? "UNKNOWN"}
        confidence={currentFrame.metrics?.confidence ?? 0}
        playing={playing}
      />

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
        }}
        onSpeed={setSpeed}
      />
    </main>
  );
}
