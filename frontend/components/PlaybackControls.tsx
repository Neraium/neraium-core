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
    <div className="controls">
      <button onClick={onPlayPause}>{playing ? "Pause" : "Play"}</button>
      <button onClick={onRestart}>Restart</button>
      <select value={speed} onChange={(e) => onSpeed(Number(e.target.value))}>
        {[0.5, 1, 1.5, 2, 3].map((v) => (
          <option value={v} key={v}>{v}x</option>
        ))}
      </select>
    </div>
  );
}
