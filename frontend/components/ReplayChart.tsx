interface FrameData {
  structural_drift_score: number
  relational_stability_score: number
  [key: string]: any
}

interface ReplayChartProps {
  frames: FrameData[]
  currentIndex: number
}

export default function ReplayChart({ frames, currentIndex }: ReplayChartProps) {
  if (frames.length === 0) return null

  const W = 1200
  const H = 320
  const PX = 60
  const PY = 40
  const IW = W - 2 * PX
  const IH = H - 2 * PY

  const getDriftPercent = (frame: FrameData) => Math.max(0, Math.min(100, (frame.structural_drift_score || 0) * 100))
  const getInstabilityPercent = (frame: FrameData) => {
    const stability = Math.max(0, Math.min(100, (frame.relational_stability_score || 0) * 100))
    return 100 - stability
  }

  // Render full timeline and use the cursor to show replay position
  const driftPoints = frames.map((frame, idx) => ({
    x: (idx / Math.max(frames.length - 1, 1)) * IW + PX,
    y: PY + IH - (getDriftPercent(frame) / 100) * IH,
    value: getDriftPercent(frame),
  }))

  const instabilityPoints = frames.map((frame, idx) => ({
    x: (idx / Math.max(frames.length - 1, 1)) * IW + PX,
    y: PY + IH - (getInstabilityPercent(frame) / 100) * IH,
    value: getInstabilityPercent(frame),
  }))

  // Current frame position
  const currentX = (currentIndex / Math.max(frames.length - 1, 1)) * IW + PX
  const currentDriftPoint = driftPoints[currentIndex]
  const currentInstabilityPoint = instabilityPoints[currentIndex]

  // Build polyline points
  const driftPath = driftPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  const instabilityPath = instabilityPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">Replay Timeline</span>
        <span className="panel-subtitle">Structural drift and relational instability across {frames.length} frames</span>
      </div>
      <div className="chart-container">
        <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
          <rect x={0} y={0} width={W} height={H} fill="rgba(0,0,0,0.18)" />
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((y) => (
            <line
              key={`grid-${y}`}
              x1={PX}
              y1={PY + IH - y * IH}
              x2={PX + IW}
              y2={PY + IH - y * IH}
              stroke="rgba(255,255,255,0.05)"
              strokeWidth="1"
            />
          ))}

          {/* Value markers */}
          {[0, 0.25, 0.5, 0.75, 1].map((y) => (
            <text
              key={`marker-${y}`}
              x={PX - 8}
              y={PY + IH - y * IH + 4}
              fill="rgba(255,255,255,0.3)"
              fontSize="10"
              textAnchor="end"
            >
              {(y * 100).toFixed(0)}
            </text>
          ))}

          {/* Axis */}
          <line x1={PX} y1={PY} x2={PX} y2={PY + IH} stroke="rgba(255,255,255,0.2)" strokeWidth="1.5" />
          <line x1={PX} y1={PY + IH} x2={PX + IW} y2={PY + IH} stroke="rgba(255,255,255,0.2)" strokeWidth="1.5" />

          {/* Instability line */}
          {instabilityPath && (
            <polyline
              points={instabilityPath}
              fill="none"
              stroke="#FF8A5C"
              strokeWidth="2.5"
              opacity="0.95"
              filter="url(#glow)"
            />
          )}

          {/* Drift line */}
          {driftPath && (
            <polyline
              points={driftPath}
              fill="none"
              stroke="#3B82F6"
              strokeWidth="2.5"
              opacity="1"
              filter="url(#glow)"
            />
          )}

          {/* Current position indicator */}
          <line
            x1={currentX}
            y1={PY}
            x2={currentX}
            y2={PY + IH}
            stroke="#06B6D4"
            strokeWidth="2"
            strokeDasharray="5,4"
            opacity="0.6"
          />
          {currentDriftPoint && (
            <circle
              cx={currentX}
              cy={currentDriftPoint.y}
              r="7"
              fill="#3B82F6"
              stroke="#FFFFFF"
              strokeWidth="2"
              opacity="0.95"
            />
          )}
          {currentInstabilityPoint && (
            <circle
              cx={currentX}
              cy={currentInstabilityPoint.y}
              r="7"
              fill="#FF8A5C"
              stroke="#FFFFFF"
              strokeWidth="2"
              opacity="0.95"
            />
          )}

          {/* Labels */}
          <text
            x={PX - 16}
            y={PY - 12}
            fill="rgba(255,255,255,0.7)"
            fontSize="12"
            fontWeight="600"
            textAnchor="end"
          >
            Value
          </text>
          <text
            x={PX + IW - 16}
            y={PY + IH + 24}
            fill="rgba(255,255,255,0.7)"
            fontSize="12"
            fontWeight="600"
            textAnchor="end"
          >
            Cycle
          </text>

          {/* Legend */}
          <g>
            <rect x={PX + 16} y={PY + 8} width={230} height={48} fill="rgba(0,0,0,0.45)" rx="4" />
            <line x1={PX + 24} y1={PY + 18} x2={PX + 44} y2={PY + 18} stroke="#3B82F6" strokeWidth="2.5" />
            <text x={PX + 52} y={PY + 22} fill="rgba(255,255,255,0.85)" fontSize="11" fontWeight="600">
              structural_drift_score
            </text>

            <line x1={PX + 24} y1={PY + 38} x2={PX + 44} y2={PY + 38} stroke="#FF8A5C" strokeWidth="2.5" />
            <text x={PX + 52} y={PY + 42} fill="rgba(255,255,255,0.85)" fontSize="11" fontWeight="600">
              relational_instability_score
            </text>
          </g>
        </svg>
      </div>
    </div>
  )
}
