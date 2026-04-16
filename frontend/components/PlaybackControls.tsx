interface PlaybackControlsProps {
  currentIndex: number
  totalFrames: number
  isPlaying: boolean
  playbackSpeed: number
  onIndexChange: (index: number) => void
  onPlayPause: () => void
  onSpeedChange: (speed: number) => void
  onRestart: () => void
}

export default function PlaybackControls({
  currentIndex,
  totalFrames,
  isPlaying,
  playbackSpeed,
  onIndexChange,
  onPlayPause,
  onSpeedChange,
  onRestart,
}: PlaybackControlsProps) {
  return (
    <div className="demo-controls-row">
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
          <span style={{ fontSize: '12px', color: '#9CA3AF', minWidth: '60px' }}>
            {currentIndex + 1} / {totalFrames}
          </span>
        </div>

        <div className="control-slider">
          <label>Speed</label>
          <input
            type="range"
            min="0.5"
            max="2"
            step="0.1"
            value={playbackSpeed}
            onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
            style={{ width: '120px' }}
          />
          <span style={{ fontSize: '12px', color: '#9CA3AF', minWidth: '40px' }}>
            {playbackSpeed.toFixed(1)}x
          </span>
        </div>

        <div className="control-buttons">
          <button className={`btn ${isPlaying ? 'active' : ''}`} onClick={onPlayPause}>
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button className="btn" onClick={onRestart}>
            ↻ Restart
          </button>
        </div>
      </div>
    </div>
  )
}
