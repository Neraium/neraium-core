"use client";

type Props = {
  playing: boolean;
  speed: number;
  onPlayPause: () => void;
  onRestart: () => void;
  onSpeed: (speed: number) => void;
};

export function PlaybackControls({ playing, speed, onPlayPause, onRestart, onSpeed }: Props) {
  return (
    <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
      <button onClick={onPlayPause}>{playing ? "Pause" : "Play"}</button>
      <button onClick={onRestart}>Restart</button>
      <select
        value={speed}
        onChange={(e) => onSpeed(Number(e.target.value))}
        style={{
          border: "1px solid #355274",
          background: "linear-gradient(180deg, #17314b, #13263b)",
          color: "#deecfb",
          padding: "8px 12px",
          borderRadius: "10px",
          fontSize: "12px",
          fontWeight: "600",
          cursor: "pointer",
          boxShadow: "0 8px 20px rgba(1, 8, 16, 0.3)",
        }}
      >
        {[0.5, 1, 1.5, 2, 3].map((v) => (
          <option value={v} key={v}>{v}x</option>
        ))}
      </select>
    </div>
  );
}
