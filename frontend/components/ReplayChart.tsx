interface ReplayChartProps {
  frames: any[]
  currentIndex: number
}

export default function ReplayChart({ frames, currentIndex }: ReplayChartProps) {
  if (frames.length === 0) return null

  const W = 900
  const H = 250
  const PX = 56
  const PY = 30
  const IW = W - 2 * PX
  const IH = H - 2 * PY

  const getDriftValue = (frame: any) => frame.structural_drift_score || 0
  const getStabilityValue = (frame: any) => frame.relational_stability_score || 0

  // Create data points for drift
  const driftPoints = frames.map((frame, idx) => ({
    x: (idx / Math.max(frames.length - 1, 1)) * IW + PX,
    y: PY + IH - getDriftValue(frame) * IH,
    value: getDriftValue(frame),
  }))

  // Create data points for stability
  const stabilityPoints = frames.map((frame, idx) => ({
    x: (idx / Math.max(frames.length - 1, 1)) * IW + PX,
    y: PY + IH - getStabilityValue(frame) * IH,
    value: getStabilityValue(frame),
  }))

  // Current frame position
  const currentX = (currentIndex / Math.max(frames.length - 1, 1)) * IW + PX

  // Build polyline points
  const driftPath = driftPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  const stabilityPath = stabilityPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">Replay Timeline</span>
        <span className="panel-subtitle">Structural drift and relational stability across {frames.length} frames</span>
      </div>
      <div className="chart-container">
        <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`}>
          {/* Background */}
          <defs>
            <linearGradient id="driftGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(59, 130, 246, 0.3)" />
              <stop offset="100%" stopColor="rgba(59, 130, 246, 0.05)" />
            </linearGradient>
            <linearGradient id="stabilityGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(34, 197, 94, 0.3)" />
              <stop offset="100%" stopColor="rgba(34, 197, 94, 0.05)" />
            </linearGradient>
          </defs>

          {/* Grid */}
          {[0.25, 0.5, 0.75].map((y) => (
            <line
              key={`grid-${y}`}
              x1={PX}
              y1={PY + IH - y * IH}
              x2={PX + IW}
              y2={PY + IH - y * IH}
              stroke="rgba(255,255,255,0.08)"
              strokeWidth="1"
            />
          ))}

          {/* Axis */}
          <line x1={PX} y1={PY} x2={PX} y2={PY + IH} stroke="rgba(255,255,255,0.3)" strokeWidth="1.5" />
          <line x1={PX} y1={PY + IH} x2={PX + IW} y2={PY + IH} stroke="rgba(255,255,255,0.3)" strokeWidth="1.5" />

          {/* Stability line */}
          <polyline points={stabilityPath} fill="none" stroke="#22C55E" strokeWidth="2" opacity="0.7" />

          {/* Drift line */}
          <polyline points={driftPath} fill="none" stroke="#3B82F6" strokeWidth="2" opacity="0.9" />

          {/* Current position indicator */}
          <line x1={currentX} y1={PY} x2={currentX} y2={PY + IH} stroke="#06B6D4" strokeWidth="2" strokeDasharray="4,3" opacity="0.8" />
          <circle cx={currentX} cy={driftPoints[currentIndex]?.y || PY + IH / 2} r="6" fill="#06B6D4" stroke="#FFFFFF" strokeWidth="2" />

          {/* Labels */}
          <text x={PX - 12} y={PY - 8} fill="rgba(255,255,255,0.8)" fontSize="11" fontWeight="600" textAnchor="end">
            Signal
          </text>
          <text x={PX + IW + 4} y={PY + IH + 20} fill="rgba(255,255,255,0.8)" fontSize="11" fontWeight="600">
            Timeline →
          </text>

          {/* Legend */}
          <line x1={PX + 12} y1={PY + 12} x2={PX + 32} y2={PY + 12} stroke="#22C55E" strokeWidth="2" />
          <text x={PX + 40} y={PY + 16} fill="rgba(255,255,255,0.8)" fontSize="10">
            Stability
          </text>

          <line x1={PX + 12} y1={PY + 28} x2={PX + 32} y2={PY + 28} stroke="#3B82F6" strokeWidth="2" />
          <text x={PX + 40} y={PY + 32} fill="rgba(255,255,255,0.8)" fontSize="10">
            Drift
          </text>
        </svg>
      </div>
    </div>
  )
}
