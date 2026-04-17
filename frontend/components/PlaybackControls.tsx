const SPEED_MIN = 0.5
const SPEED_MAX = 2
const SPEED_STEP = 0.1

interface PlaybackControlsProps {
  currentIndex: number
  totalFrames: number
  isPlaying: boolean
  playbackSpeed: number
  phaseMarkers: Array<{ label: string; index: number; key: string }>
  onIndexChange: (index: number) => void
  onPlayPause: () => void
  onSpeedChange: (speed: number) => void
  onRestart: () => void
  onStep: (direction: 1 | -1) => void
}

export default function PlaybackControls({
  currentIndex,
  totalFrames,
  isPlaying,
  playbackSpeed,
  phaseMarkers,
  onIndexChange,
  onPlayPause,
  onSpeedChange,
  onRestart,
  onStep,
}: PlaybackControlsProps) {
  return (
    <div className="demo-controls-row" role="region" aria-label="Playback controls">
      <div className="controls-group">
        <div className="control-slider">
          <label>Frame</label>
          <input
            type="range"
            min="0"
            max={Math.max(0, totalFrames - 1)}
            value={currentIndex}
            onChange={(e) => {
              onIndexChange(parseInt(e.target.value))
            }}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: '12px', color: '#94a3b8', minWidth: '70px', fontWeight: '500', fontVariantNumeric: 'tabular-nums' }}>
            {currentIndex + 1} / {totalFrames}
          </span>
        </div>

        <div className="control-slider">
          <label>Speed</label>
          <input
            type="range"
            min={SPEED_MIN}
            max={SPEED_MAX}
            step={SPEED_STEP}
            value={playbackSpeed}
            onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
            style={{ width: '120px' }}
          />
          <span style={{ fontSize: '12px', color: '#94a3b8', minWidth: '50px', fontWeight: '500', fontVariantNumeric: 'tabular-nums' }}>
            {playbackSpeed.toFixed(1)}x
          </span>
        </div>

        <div className="control-buttons">
          <button className="btn btn-secondary" onClick={() => onStep(-1)}>
            ← Prev
          </button>
          <button className={`btn ${isPlaying ? 'active' : ''}`} onClick={onPlayPause}>
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button className="btn btn-secondary" onClick={() => onStep(1)}>
            Next →
          </button>
          <button className="btn" onClick={onRestart}>
            ↻ Restart
          </button>
        </div>
      </div>
      <div className="phase-chip-row" aria-label="Phase jump shortcuts">
        {phaseMarkers.map((phase) => (
          <button
            key={`${phase.key}-${phase.index}`}
            type="button"
            className={`phase-chip ${currentIndex >= phase.index ? 'reached' : ''}`}
            onClick={() => onIndexChange(phase.index)}
          >
            {phase.label}
          </button>
        ))}
      </div>
    </div>
  )
}
