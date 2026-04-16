interface FrameData {
  structural_drift_score: number
  composite_instability?: number
  risk_level?: string
  [key: string]: any
}

interface ReplayChartProps {
  frames: FrameData[]
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

  const getDriftValue = (frame: FrameData) => frame.structural_drift_score || 0
  const getInstabilityValue = (frame: FrameData) => frame.composite_instability || 0

  // Only show data up to current frame (live effect)
  const visibleFrames = frames.slice(0, currentIndex + 1)

  // Create data points for drift (only up to current index)
  const driftPoints = visibleFrames.map((frame, idx) => ({
    x: (idx / Math.max(frames.length - 1, 1)) * IW + PX,
    y: PY + IH - getDriftValue(frame) * IH,
    value: getDriftValue(frame),
  }))

  // Create data points for instability (only up to current index)
  const instabilityPoints = visibleFrames.map((frame, idx) => ({
    x: (idx / Math.max(frames.length - 1, 1)) * IW + PX,
    y: PY + IH - getInstabilityValue(frame) * IH,
    value: getInstabilityValue(frame),
  }))

  // Current frame position
  const currentX = (currentIndex / Math.max(frames.length - 1, 1)) * IW + PX

  // Build polyline points (only visible frames)
  const driftPath = driftPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  const instabilityPath = instabilityPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')

  // Get current values for status display
  const currentDrift = getDriftValue(frames[currentIndex] || frames[0])
  const currentInstability = getInstabilityValue(frames[currentIndex] || frames[0])
  const riskLevel = (frames[currentIndex] as any)?.risk_level || 'LOW'

  // Determine color based on instability and drift (0-1 scale)
  const instabilityColor = currentInstability > 0.5 ? '#EF4444' : currentInstability > 0.3 ? '#F97316' : '#22C55E'
  const driftColor = currentDrift > 0.5 ? '#EF4444' : currentDrift > 0.3 ? '#F97316' : '#22C55E'

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
              <stop offset="0%" stopColor="rgba(59, 130, 246, 0.2)" />
              <stop offset="100%" stopColor="rgba(59, 130, 246, 0.02)" />
            </linearGradient>
            <linearGradient id="stabilityGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(34, 197, 94, 0.2)" />
              <stop offset="100%" stopColor="rgba(34, 197, 94, 0.02)" />
            </linearGradient>
          </defs>

          {/* Zone backgrounds for better visibility */}
          <rect x={PX} y={PY} width={IW} height={IH * 0.33} fill="rgba(239, 68, 68, 0.08)" />
          <rect x={PX} y={PY + IH * 0.33} width={IW} height={IH * 0.33} fill="rgba(249, 115, 22, 0.08)" />
          <rect x={PX} y={PY + IH * 0.66} width={IW} height={IH * 0.34} fill="rgba(34, 197, 94, 0.08)" />

          {/* Grid */}
          {[0.25, 0.5, 0.75].map((y) => (
            <line
              key={`grid-${y}`}
              x1={PX}
              y1={PY + IH - y * IH}
              x2={PX + IW}
              y2={PY + IH - y * IH}
              stroke="rgba(255,255,255,0.12)"
              strokeWidth="1"
            />
          ))}

          {/* Axis */}
          <line x1={PX} y1={PY} x2={PX} y2={PY + IH} stroke="rgba(255,255,255,0.4)" strokeWidth="2" />
          <line x1={PX} y1={PY + IH} x2={PX + IW} y2={PY + IH} stroke="rgba(255,255,255,0.4)" strokeWidth="2" />

          {/* Instability line */}
          <polyline points={instabilityPath} fill="none" stroke="#EF4444" strokeWidth="3" opacity="0.85" />

          {/* Drift line */}
          <polyline points={driftPath} fill="none" stroke="#3B82F6" strokeWidth="3" opacity="0.95" />

          {/* Current position indicator */}
          <line x1={currentX} y1={PY} x2={currentX} y2={PY + IH} stroke="#06B6D4" strokeWidth="2" strokeDasharray="5,4" opacity="0.9" />
          <circle cx={currentX} cy={driftPoints[currentIndex]?.y || PY + IH / 2} r="7" fill="#06B6D4" stroke="#FFFFFF" strokeWidth="2.5" />

          {/* Labels */}
          <text x={PX - 12} y={PY - 8} fill="rgba(255,255,255,0.8)" fontSize="11" fontWeight="600" textAnchor="end">
            Signal
          </text>
          <text x={PX + IW + 4} y={PY + IH + 20} fill="rgba(255,255,255,0.8)" fontSize="11" fontWeight="600">
            Timeline →
          </text>

          {/* Legend with current values and risk */}
          <line x1={PX + 12} y1={PY + 12} x2={PX + 32} y2={PY + 12} stroke="#3B82F6" strokeWidth="3" />
          <text x={PX + 40} y={PY + 16} fill="rgba(255,255,255,0.9)" fontSize="11" fontWeight="600">
            Structural Drift: {(currentDrift * 100).toFixed(0)}%
          </text>

          <line x1={PX + 12} y1={PY + 30} x2={PX + 32} y2={PY + 30} stroke="#EF4444" strokeWidth="3" />
          <text x={PX + 40} y={PY + 34} fill="rgba(255,255,255,0.9)" fontSize="11" fontWeight="600">
            Instability: {(currentInstability * 100).toFixed(0)}% • Risk: {riskLevel}
          </text>

          {/* Status indicators */}
          <circle cx={PX + IW - 50} cy={PY + 16} r="4" fill={driftColor} opacity="0.8" />
          <circle cx={PX + IW - 50} cy={PY + 34} r="4" fill={instabilityColor} opacity="0.8" />
        </svg>
      </div>
    </div>
  )
}
